#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Map Animations and Surface Impact Analysis for SSW Events.

Creates:
  1. Animated North Polar Stereographic maps of stratospheric u-wind and
     temperature at one pressure level (ERA5 Target | Prediction |
     Pred − Target per row).
  2. Time series (zonal mean at 60°N) of one variable showing
     stratosphere-troposphere coupling after the SSW onset.

Multi-stream layout
-------------------
A strato_s2s forecast store spreads the atmospheric column over several
streams (``ERA5pl``: surface + 50–1000 hPa, ``S2Spredictors``: 1–70 hPa;
channel suffix = hPa).  Levels are therefore requested in **hPa**
(``--level 10``) and located with ``discover_column(...).nearest(p)``, which
merges all streams.  ``--variable`` accepts surface names (``2t``, ``msl``,
``10u``) or ``<var>_<hPa>`` (``u_10``, ``t_850``); on legacy single-stream
ERA5ml stores pass ``--level-type model`` so that ``u_29`` is read as L137
model level 29 (≈ 10 hPa).

Requires ``ffmpeg`` on PATH for animation output.

Usage::

    ssw-analyze polar-maps \\
        --validations-config eval_config/ssw_feb2018.yml \\
        --data-dir /path/to/validation/data \\
        --output-dir plots/polar_maps \\
        --level 10 --variable u_10

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_maps \\
        --run-ids qq8xsoeh j7ns9146 \\
        --data-dir ./data --level 50 --variable 2t
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.io import (
    LevelRef,
    convert_times_to_datetime,
    discover_column,
    find_latitude_indices,
    get_coords,
    get_forecast_steps,
    load_level,
    open_validation,
    surface_channel_ref,
    zarr_store_path,
)
from weathergen.stratosphere.levels import (
    SURFACE_CHANNELS,
    LevelType,
    channel_pressure,
    format_pressure,
    parse_channel,
)

_logger = logging.getLogger(__name__)

# Default SSW reference date (Feb 2018 event)
SSW_DATE = datetime(2018, 2, 12)

# Default pressure level (hPa) for the polar maps and the 60°N time series.
DEFAULT_LEVEL_HPA = 10.0
DEFAULT_VARIABLE = "u_10"


# ---------------------------------------------------------------------------
# Variable / level resolution
# ---------------------------------------------------------------------------


def _level_tag(ref: LevelRef) -> str:
    """Filename-safe label: ``'u10hPa'`` for levels, ``'2t'`` for surface channels."""
    if np.isnan(ref.pressure):
        return ref.channel
    return f"{ref.variable}{ref.pressure:.3g}hPa"


def _level_title(ref: LevelRef) -> str:
    """Human-readable label: ``'u 10 hPa (S2Spredictors/u_10)'``."""
    if np.isnan(ref.pressure):
        return f"{ref.channel} ({ref.stream})"
    return f"{ref.variable} {ref.pressure:.3g} hPa ({ref.stream}/{ref.channel})"


def _resolve_variable(
    zio: Any,
    variable: str,
    level_type: LevelType = "pressure",
    sample: int = 0,
) -> LevelRef | None:
    """
    Locate *variable* in the store.

    Surface names (``2t``, ``msl``, ``10u`` — anything without a numeric
    suffix or listed in ``SURFACE_CHANNELS``) are looked up directly in every
    stream.  ``<var>_<n>`` names are interpreted by *level_type*: the suffix
    is hPa (default) or an L137 model level (``--level-type model``,
    ``u_29`` → 9.8 hPa) and the nearest available level of ``var`` across
    all streams is returned.  ``None`` (with a warning) if nothing matches.
    """
    var, lv = parse_channel(variable)
    if lv is None or variable in SURFACE_CHANNELS:
        ref = surface_channel_ref(zio, variable, sample)
        if ref is None:
            _logger.warning("surface channel %r not found in streams %s", variable, zio.streams)
        return ref

    p = channel_pressure(variable, level_type)
    if p is None:
        _logger.warning("%r: cannot convert level %s (%s levels) to hPa", variable, lv, level_type)
        return None
    col = discover_column(zio, var, sample)
    try:
        ref = col.nearest(p)
    except KeyError as e:
        _logger.warning("%r (%s): %s", variable, format_pressure(p), e)
        return None
    if abs(ref.pressure - p) / p > 0.02:
        _logger.info("%r → using %s", variable, _level_title(ref))
    return ref


def _resolve_level(
    zio: Any, variable: str, pressure_hpa: float, sample: int = 0
) -> LevelRef | None:
    """Nearest level of *variable* to *pressure_hpa* across all streams, or ``None``."""
    col = discover_column(zio, variable, sample)
    if not col:
        _logger.warning("no %s levels in streams %s", variable, zio.streams)
        return None
    try:
        return col.nearest(pressure_hpa)
    except KeyError as e:
        _logger.warning("%s", e)
        return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _load_series(zio: Any, ref: LevelRef, sample: int) -> tuple[Any, Any, list[Any]]:
    """Ensemble-mean prediction / target ``(n_steps, n_pts)`` of one level."""
    steps = get_forecast_steps(zio)
    preds, tgts, raw_times = [], [], []
    for step in steps:
        pred, tgt, times = load_level(zio, ref, step, sample)
        preds.append(np.asarray(pred, dtype=np.float32).mean(axis=-1))
        tgts.append(np.asarray(tgt, dtype=np.float32).mean(axis=-1))
        raw_times.append(times[0])
    return np.stack(preds), np.stack(tgts), raw_times


def extract_polar_maps(
    zarr_path: Path,
    label: str,
    pressure_hpa: float = DEFAULT_LEVEL_HPA,
    variable: str = "u",
    sample: int = 0,
) -> dict[str, Any] | None:
    """
    Extract one level of *variable* for the polar map animation.

    Returns a dict with keys ``label``, ``predictions``, ``targets``,
    ``coords``, ``datetimes``, ``channel``, ``pressure``, ``tag``,
    ``title``; or ``None`` when the level is missing.

    Args:
        zarr_path:    Path to the validation zarr store.
        label:        Display label (e.g. validation config key).
        pressure_hpa: Requested pressure (hPa); the nearest available level
                      of *variable* in any stream is used.
        variable:     Variable letter: ``'u'``, ``'t'``, ``'v'``, or ``'z'``.
        sample:       Sample index.
    """
    with open_validation(zarr_path) as zio:
        ref = _resolve_level(zio, variable, pressure_hpa, sample)
        if ref is None:
            _logger.warning("%s: no %s level near %s — skipping", label, variable, pressure_hpa)
            return None
        _logger.info("%s: extracting %s…", label, _level_title(ref))
        coords = get_coords(zio, ref.stream, sample)  # (n_pts, 2): [lat, lon]
        predictions, targets, raw_times = _load_series(zio, ref, sample)

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info("%s: %d frames  %s → %s", label, len(datetimes), datetimes[0], datetimes[-1])

    return {
        "label": label,
        "predictions": predictions,
        "targets": targets,
        "coords": coords,
        "datetimes": datetimes,
        "channel": ref.channel,
        "pressure": ref.pressure,
        "tag": _level_tag(ref),
        "title": _level_title(ref),
    }


def extract_surface_variable(
    zarr_path: Path,
    label: str,
    variable: str = DEFAULT_VARIABLE,
    target_latitude: float = 60.0,
    sample: int = 0,
    level_type: LevelType = "pressure",
) -> dict[str, Any] | None:
    """
    Extract the zonal mean of one variable at *target_latitude*.

    Args:
        zarr_path:        Path to the validation zarr store.
        label:            Display label.
        variable:         Surface channel (``'2t'``, ``'msl'``, ``'10u'``) or
                          ``<var>_<level>`` (``'u_10'``, ``'t_850'``).
        target_latitude:  Latitude for the zonal mean (default 60°N).
        sample:           Sample index.
        level_type:       How to read the numeric suffix of *variable*:
                          ``'pressure'`` (hPa) or ``'model'`` (L137 level).

    Returns:
        Dict with ``label``, ``predictions``, ``targets``, ``datetimes``,
        ``channel``, ``tag``, ``title``; or ``None`` when unavailable.
    """
    with open_validation(zarr_path) as zio:
        ref = _resolve_variable(zio, variable, level_type, sample)
        if ref is None:
            _logger.warning("%s: %s not available — skipping", label, variable)
            return None
        _logger.info("%s: extracting %s…", label, _level_title(ref))
        coords = get_coords(zio, ref.stream, sample)
        lat_indices = find_latitude_indices(coords, target_latitude)
        pred_all, tgt_all, raw_times = _load_series(zio, ref, sample)
        predictions = pred_all[:, lat_indices].mean(axis=1)
        targets = tgt_all[:, lat_indices].mean(axis=1)

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info("%s: %d steps for %s", label, len(datetimes), _level_tag(ref))

    return {
        "label": label,
        "predictions": predictions,
        "targets": targets,
        "datetimes": datetimes,
        "channel": ref.channel,
        "tag": _level_tag(ref),
        "title": _level_title(ref),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def create_polar_animation(
    data_list: list[dict[str, Any]],
    variable: str,
    output_dir: Path,
    fps: int = 5,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Render an animated North Polar Stereographic map.

    Layout per experiment row: ERA5 Target | Prediction | Pred − Target.
    Colorbars are built once and survive ``ax.clear()`` across frames.
    Output is saved as ``polar_map_animation_<channel>.mp4`` (requires ffmpeg).
    """
    if not CARTOPY_AVAILABLE:
        _logger.warning("cartopy not available — skipping animation (install cartopy)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_list:
        return

    var_channel = data_list[0]["tag"]
    _logger.info("Creating polar map animation for %s…", data_list[0]["title"])

    # Common time axis: use the run with the most frames
    reference_data = max(data_list, key=lambda d: len(d["datetimes"]))
    common_times = reference_data["datetimes"]
    _logger.info("  %d frames  %s → %s", len(common_times), common_times[0], common_times[-1])

    def _ts(t: Any) -> int:
        if isinstance(t, datetime):
            return int(t.timestamp())
        return int(np.datetime64(t, "s").astype("int64"))

    # Pad runs that start later by repeating their first frame
    plot_data: list[dict[str, Any]] = []
    for data in data_list:
        pred = data["predictions"]
        tgt = data["targets"]
        start_ts = _ts(data["datetimes"][0])
        pad_frames = sum(1 for t in common_times if _ts(t) < start_ts)
        if pad_frames > 0:
            pred = np.concatenate([np.repeat(pred[:1], pad_frames, axis=0), pred], axis=0)
            tgt = np.concatenate([np.repeat(tgt[:1], pad_frames, axis=0), tgt], axis=0)
        plot_data.append(
            {
                "label": data["label"],
                "lats": data["coords"][:, 0],
                "lons": data["coords"][:, 1],
                "predictions": pred,
                "targets": tgt,
            }
        )

    # Fixed colour normalisation from full data range
    all_vals = np.concatenate(
        [np.concatenate([d["predictions"].ravel(), d["targets"].ravel()]) for d in plot_data]
    )
    if variable == "t":
        cmap = "RdYlBu_r"
        norm: Any = Normalize(
            vmin=float(np.percentile(all_vals, 2)),
            vmax=float(np.percentile(all_vals, 98)),
        )
        cb_label = "Temperature (K)"
    elif variable in ("u", "v"):
        cmap = "RdBu_r"
        vabs = float(np.percentile(np.abs(all_vals), 99))
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        cb_label = "Zonal Wind (m/s)" if variable == "u" else "Meridional Wind (m/s)"
    else:
        cmap = "viridis"
        norm = Normalize(
            vmin=float(np.percentile(all_vals, 1)),
            vmax=float(np.percentile(all_vals, 99)),
        )
        cb_label = "Geopotential Height (m)"

    all_diffs = np.concatenate([(d["predictions"] - d["targets"]).ravel() for d in plot_data])
    diff_abs = max(float(np.percentile(np.abs(all_diffs), 98)), 0.5)
    norm_diff = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    diff_label = (
        "Bias (m/s)" if variable in ("u", "v") else ("Bias (K)" if variable == "t" else "Bias")
    )

    # Figure: n_exps map rows + 1 thin colorbar row
    n_exps = len(plot_data)
    fig = plt.figure(figsize=(15, 6.5 * n_exps + 1.0), facecolor="white")
    gs = gridspec.GridSpec(
        n_exps + 1,
        3,
        height_ratios=[6.5] * n_exps + [0.4],
        hspace=0.18,
        wspace=0.04,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.05,
    )

    proj_stereo = ccrs.NorthPolarStereo()
    proj_pc = ccrs.PlateCarree()

    map_axes: list[list[Any]] = []
    for row in range(n_exps):
        row_ax = [fig.add_subplot(gs[row, col], projection=proj_stereo) for col in range(3)]
        for ax in row_ax:
            ax.set_extent([-180, 180, 20, 90], proj_pc)
        map_axes.append(row_ax)

    # Persistent colorbars (survive ax.clear())
    cax_main = fig.add_subplot(gs[n_exps, :2])
    cax_diff = fig.add_subplot(gs[n_exps, 2])
    sm_main = ScalarMappable(cmap=cmap, norm=norm)
    sm_main.set_array([])
    sm_diff = ScalarMappable(cmap="RdBu_r", norm=norm_diff)
    sm_diff.set_array([])
    cb_main = fig.colorbar(sm_main, cax=cax_main, orientation="horizontal")
    cb_diff = fig.colorbar(sm_diff, cax=cax_diff, orientation="horizontal")
    cb_main.set_label(cb_label, fontsize=9)
    cb_diff.set_label(diff_label, fontsize=9)

    title_obj = fig.suptitle("", fontsize=12, fontweight="bold", y=0.975)
    col_headers = ["ERA5 Target", "Prediction", "Pred \u2212 Target"]
    sc_kw = dict(transform=proj_pc, s=18, linewidths=0, rasterized=True)

    def update(frame: int) -> list:
        for row_idx, row_ax in enumerate(map_axes):
            for ax in row_ax:
                ax.clear()
                ax.set_extent([-180, 180, 20, 90], proj_pc)
                ax.coastlines(linewidth=0.6, color="#333333", zorder=5)
                ax.gridlines(linestyle="--", alpha=0.3, linewidth=0.4, color="gray", zorder=4)

            data = plot_data[row_idx]
            tgt_f = data["targets"][frame]
            pred_f = data["predictions"][frame]
            lats, lons = data["lats"], data["lons"]

            row_ax[0].scatter(lons, lats, c=tgt_f, cmap=cmap, norm=norm, **sc_kw)
            row_ax[1].scatter(lons, lats, c=pred_f, cmap=cmap, norm=norm, **sc_kw)
            row_ax[2].scatter(
                lons, lats, c=(pred_f - tgt_f), cmap="RdBu_r", norm=norm_diff, **sc_kw
            )

            if row_idx == 0:
                for col_idx, hdr in enumerate(col_headers):
                    row_ax[col_idx].set_title(hdr, fontsize=10, fontweight="bold", pad=3)

            if n_exps > 1:
                row_ax[0].text(
                    0.02,
                    0.97,
                    data["label"],
                    transform=row_ax[0].transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="none"),
                )

        t = common_times[frame]
        days = (t - common_times[0]).total_seconds() / 86400
        ssw_flag = "  \u2605 SSW onset" if t.date() == ssw_date.date() else ""
        title_obj.set_text(
            f"{data_list[0]['title']}"
            f"  \u00b7  {t.strftime('%Y-%m-%d %H:%M')}  (+{days:.0f}\u2009d){ssw_flag}"
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(common_times), interval=1000 // fps, blit=False
    )

    if animation.FFMpegWriter.isAvailable():
        output_file = output_dir / f"polar_map_animation_{var_channel}.mp4"
        _logger.info("Saving animation (ffmpeg) to %s…", output_file)
        anim.save(output_file, writer="ffmpeg", fps=fps, dpi=110)
    else:
        _logger.warning("ffmpeg not available — saving as GIF instead (larger file, lower quality)")
        output_file = output_dir / f"polar_map_animation_{var_channel}.gif"
        _logger.info("Saving animation (Pillow/GIF) to %s…", output_file)
        anim.save(output_file, writer="pillow", fps=fps, dpi=80)

    _logger.info("Saved: %s", output_file)
    plt.close()


def plot_surface_impact(data_list: list[dict[str, Any]], output_dir: Path) -> None:
    """
    Plot time series of a surface variable showing post-SSW stratosphere-
    troposphere coupling.

    Two panels:
      - Top: predictions (solid) vs ERA5 targets (dashed) per experiment.
      - Bottom: bias (Pred − Target) per experiment.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_list:
        return

    tag = data_list[0]["tag"]
    title = data_list[0]["title"]
    _logger.info("Plotting surface impact for %s…", title)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, data in enumerate(data_list):
        datetimes = data["datetimes"]
        days = np.array([(dt - datetimes[0]).total_seconds() / 86400 for dt in datetimes])
        color = colors[idx % len(colors)]

        ax1.plot(
            days,
            data["predictions"],
            color=color,
            linewidth=2,
            label=f"{data['label']} Pred",
        )
        ax1.plot(
            days,
            data["targets"],
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"{data['label']} Target",
        )
        ax2.plot(
            days,
            data["predictions"] - data["targets"],
            color=color,
            linewidth=2,
            label=data["label"],
        )

    ssw_offset = (SSW_DATE - data_list[0]["datetimes"][0]).total_seconds() / 86400
    ax1.axvline(
        ssw_offset,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="SSW Event",
    )
    ax2.axvline(ssw_offset, color="red", linestyle="--", linewidth=2, alpha=0.7)

    ax1.set_ylabel(_variable_label(tag), fontsize=12)
    ax1.set_title(
        f"Surface Impact of SSW Event: {title} at 60°N",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Lead Time (days)", fontsize=12)
    ax2.set_ylabel("Bias (Pred \u2212 Target)", fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    max_days = max(
        max((dt - data["datetimes"][0]).total_seconds() / 86400 for dt in data["datetimes"])
        for data in data_list
    )
    ax2.set_xticks(np.arange(0, max_days + 1, 7))

    plt.tight_layout()
    output_file = output_dir / f"surface_impact_{tag}_60N.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    _logger.info("Saved: %s", output_file)
    plt.close()


_SURFACE_LABELS = {
    "2t": "2m Temperature (K)",
    "2d": "2m Dewpoint (K)",
    "10u": "10m Zonal Wind (m/s)",
    "10v": "10m Meridional Wind (m/s)",
    "sp": "Surface Pressure (Pa)",
    "msl": "Mean Sea-Level Pressure (Pa)",
    "skt": "Skin Temperature (K)",
}
_LEVEL_LABELS = {
    "u": "Zonal Wind (m/s)",
    "v": "Meridional Wind (m/s)",
    "t": "Temperature (K)",
    "z": "Geopotential (m² s⁻²)",
    "q": "Specific Humidity (kg/kg)",
}


def _variable_label(tag: str) -> str:
    """Axis label from a ``_level_tag`` (``'u10hPa'`` → ``'10 hPa Zonal Wind (m/s)'``)."""
    if tag in _SURFACE_LABELS:
        return _SURFACE_LABELS[tag]
    m = re.fullmatch(r"([A-Za-z]+)([\d.]+)hPa", tag)
    if m:
        var, p = m.groups()
        return f"{p} hPa {_LEVEL_LABELS.get(var, var)}"
    return tag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Create polar cap map animations and surface impact plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing validation zarr stores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/polar_maps"),
        help="Output directory for plots and animations.",
    )

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        "--run-ids",
        nargs="+",
        help="Validation IDs used directly as zarr path components.",
    )
    id_group.add_argument(
        "--validations-config",
        type=Path,
        help="YAML validations config file.",
    )

    parser.add_argument("--sample", type=int, default=0, help="Sample index (default: 0).")
    parser.add_argument(
        "--level",
        type=float,
        default=None,
        metavar="HPA",
        help=f"Pressure level (hPa) for the u/t polar maps; the nearest available level in "
        f"any stream is used (default: {DEFAULT_LEVEL_HPA:g}; 50 for the lower stratosphere).",
    )
    parser.add_argument(
        "--model-level",
        type=int,
        default=None,
        help="Legacy alias of --level: L137 model level (e.g. 29 ≈ 10 hPa, 55 ≈ 75 hPa), "
        "converted to its full-level pressure.",
    )
    parser.add_argument(
        "--variable",
        "--surface-var",
        dest="variable",
        type=str,
        default=DEFAULT_VARIABLE,
        help="Variable for the 60°N impact time series: a surface channel (2t, msl, 10u, sp) "
        "or <var>_<hPa> such as u_10, t_850, z_500 (default: %(default)s).",
    )
    parser.add_argument(
        "--level-type",
        choices=("pressure", "model"),
        default="pressure",
        help="How to read the numeric suffix of --variable: hPa (default) or L137 model level "
        "(legacy ERA5ml stores, e.g. --variable u_29 --level-type model).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Animation frames per second.")
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip polar map animations; still creates the surface impact plot.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.level is not None and args.model_level is not None:
        parser.error("--level and --model-level are mutually exclusive")
    if args.model_level is not None:
        level_hpa = channel_pressure(f"u_{args.model_level}", "model")
        if level_hpa is None:
            parser.error(f"--model-level {args.model_level} is not a valid L137 level")
        _logger.info("--model-level %d → %s", args.model_level, format_pressure(level_hpa))
    else:
        level_hpa = args.level if args.level is not None else DEFAULT_LEVEL_HPA

    # Resolve run specs from config or CLI
    if args.validations_config:
        cfg = load_validations_config(args.validations_config)
        # Deduplicate (zarr ID, rank, sample) while keeping first label; a single
        # run_id can now hold many samples spread across ranks (date_ranges
        # inference), so dedup must key on more than just the run_id.
        seen: set[tuple[str, int, int]] = set()
        run_specs = []
        for label, v in cfg.items():
            rank = v.get("rank", 0)
            sample = v.get("sample", args.sample)
            key = (v["id"], rank, sample)
            if key not in seen:
                seen.add(key)
                run_specs.append(
                    {
                        "id": v["id"],
                        "label": label,
                        "sample": sample,
                        "rank": rank,
                    }
                )
    else:
        run_specs = [
            {"id": rid, "label": rid, "sample": args.sample, "rank": 0} for rid in args.run_ids
        ]

    # -----------------------------------------------------------------------
    # Polar map animations
    # -----------------------------------------------------------------------
    if not args.skip_animation:
        u_data: list[dict[str, Any]] = []
        t_data: list[dict[str, Any]] = []

        for spec in run_specs:
            zarr_path = zarr_store_path(args.data_dir, spec["id"], spec["rank"])
            if not zarr_path.exists():
                _logger.warning("zarr not found, skipping: %s", zarr_path)
                continue
            for var, bucket in (("u", u_data), ("t", t_data)):
                result = extract_polar_maps(
                    zarr_path, spec["label"], level_hpa, var, spec["sample"]
                )
                if result is not None:
                    bucket.append(result)

        # Runs may resolve to slightly different nearest levels; animate only
        # those matching the first run so one figure shows one pressure.
        for var, bucket in (("u", u_data), ("t", t_data)):
            if len({d["tag"] for d in bucket}) > 1:
                keep = bucket[0]["tag"]
                dropped = [d["label"] for d in bucket if d["tag"] != keep]
                _logger.warning(
                    "%s: runs %s resolved to a different level than %s — dropped",
                    var,
                    dropped,
                    keep,
                )
                bucket[:] = [d for d in bucket if d["tag"] == keep]

        if u_data:
            create_polar_animation(u_data, "u", args.output_dir, args.fps)
        else:
            _logger.warning("No u-wind data extracted for animation.")
        if t_data:
            create_polar_animation(t_data, "t", args.output_dir, args.fps)
        else:
            _logger.warning("No temperature data extracted for animation.")

    # -----------------------------------------------------------------------
    # Surface impact time series
    # -----------------------------------------------------------------------
    surface_data: list[dict[str, Any]] = []
    for spec in run_specs:
        zarr_path = zarr_store_path(args.data_dir, spec["id"], spec["rank"])
        if not zarr_path.exists():
            continue
        result = extract_surface_variable(
            zarr_path,
            spec["label"],
            args.variable,
            sample=spec["sample"],
            level_type=args.level_type,
        )
        if result is not None:
            surface_data.append(result)

    if surface_data:
        plot_surface_impact(surface_data, args.output_dir)
    else:
        _logger.warning("No surface data extracted.")

    _logger.info("Done.")


if __name__ == "__main__":
    main()
