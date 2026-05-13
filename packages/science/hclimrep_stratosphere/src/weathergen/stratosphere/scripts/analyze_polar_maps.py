#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Map Animations and Surface Impact Analysis for SSW Events.

Creates:
  1. Animated North Polar Stereographic maps of stratospheric u-wind and
     temperature (ERA5 Target | Prediction | Pred − Target per row).
  2. Time series of a surface/tropospheric variable showing
     stratosphere-troposphere coupling after the SSW onset.

Requires ``ffmpeg`` on PATH for animation output.

Usage::

    ssw-analyze polar-maps \\
        --validations-config eval_config/ssw_feb2018.yml \\
        --data-dir /path/to/validation/data \\
        --output-dir plots/polar_maps

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_maps \\
        --run-ids qq8xsoeh j7ns9146 \\
        --data-dir ./data
"""

from __future__ import annotations

import argparse
import logging
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
    convert_times_to_datetime,
    find_latitude_indices,
    get_channels,
    get_coords,
    get_forecast_steps,
    load_step,
    open_validation,
)

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zarr"

# Default SSW reference date (Feb 2018 event)
SSW_DATE = datetime(2018, 2, 12)


# ---------------------------------------------------------------------------
# Stream selection helper
# ---------------------------------------------------------------------------


def _stream_for_variable(zio: Any, variable: str) -> str | None:
    """Return the appropriate stream name for *variable*, or ``None`` if absent.

    Pressure-level variables (``z_500``, ``t_850``, etc.) live in ``ERA5pl``;
    surface variables (``2t``, ``10u``, ``10v``) and model-level variables
    live in ``ERA5ml``.
    """
    need_pl = variable.startswith(("z_", "t_", "u_", "v_")) and variable not in (
        "2t",
        "10u",
        "10v",
    )
    prefer = "ERA5pl" if need_pl else "ERA5ml"
    return prefer if prefer in zio.streams else None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_polar_maps(
    zarr_path: Path,
    label: str,
    model_level: int = 55,
    variable: str = "u",
    sample: int = 0,
) -> dict[str, Any] | None:
    """
    Extract ERA5ml model-level data for polar map animation.

    Returns a dict with keys ``label``, ``predictions``, ``targets``,
    ``coords``, ``datetimes``, ``channel``, ``model_level``; or ``None``
    on error.

    Args:
        zarr_path:   Path to the validation zarr store.
        label:       Display label (e.g. validation config key).
        model_level: Model level number (30 ≈ 12 hPa, 55 ≈ 77 hPa).
        variable:    Variable letter: ``'u'``, ``'t'``, ``'v'``, or ``'z'``.
        sample:      Sample index.
    """
    target_channel = f"{variable}_{model_level}"
    _logger.info("%s: extracting %s from ERA5ml…", label, target_channel)

    with open_validation(zarr_path) as zio:
        if "ERA5ml" not in zio.streams:
            _logger.warning(
                "%s: ERA5ml stream not available (streams: %s)", label, zio.streams
            )
            return None

        stream = "ERA5ml"
        channels = get_channels(zio, stream, sample)
        if target_channel not in channels:
            avail = [c for c in channels if c.startswith(variable + "_")][:10]
            _logger.warning(
                "%s: channel %s not found — available %s channels: %s…",
                label,
                target_channel,
                variable,
                avail,
            )
            return None

        ch_idx = channels.index(target_channel)
        coords = get_coords(zio, stream, sample)  # (n_pts, 2): [lat, lon]
        steps = get_forecast_steps(zio)

        n_steps = len(steps)
        n_pts = coords.shape[0]
        predictions = np.zeros((n_steps, n_pts), dtype=np.float32)
        targets = np.zeros((n_steps, n_pts), dtype=np.float32)
        raw_times: list[Any] = []

        for i, step in enumerate(steps):
            pred, tgt, times = load_step(zio, stream, step, sample)
            # pred/tgt: (n_pts, n_channels, n_ens) → ensemble mean → (n_pts,)
            predictions[i] = pred[:, ch_idx, :].mean(axis=-1)
            targets[i] = tgt[:, ch_idx, 0]
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info("%s: %d frames  %s → %s", label, n_steps, datetimes[0], datetimes[-1])

    return {
        "label": label,
        "predictions": predictions,
        "targets": targets,
        "coords": coords,
        "datetimes": datetimes,
        "channel": target_channel,
        "model_level": model_level,
    }


def extract_surface_variable(
    zarr_path: Path,
    label: str,
    variable: str = "z_500",
    target_latitude: float = 60.0,
    sample: int = 0,
) -> dict[str, Any] | None:
    """
    Extract the zonal mean of a surface or tropospheric variable.

    Args:
        zarr_path:        Path to the validation zarr store.
        label:            Display label.
        variable:         Channel name (e.g. ``'z_500'``, ``'2t'``, ``'10u'``).
        target_latitude:  Latitude for the zonal mean (default 60°N).
        sample:           Sample index.

    Returns:
        Dict with ``label``, ``predictions``, ``targets``, ``datetimes``,
        ``channel``; or ``None`` on error.
    """
    _logger.info("%s: extracting %s…", label, variable)

    with open_validation(zarr_path) as zio:
        stream = _stream_for_variable(zio, variable)
        if stream is None:
            _logger.warning(
                "%s: required stream for %s not found (available: %s)",
                label,
                variable,
                list(zio.streams),
            )
            return None

        channels = get_channels(zio, stream, sample)
        if variable not in channels:
            _logger.warning(
                "%s: %s not found in %s — available: %s…",
                label,
                variable,
                stream,
                channels[:20],
            )
            return None

        ch_idx = channels.index(variable)
        coords = get_coords(zio, stream, sample)
        lat_indices = find_latitude_indices(coords, target_latitude)
        steps = get_forecast_steps(zio)

        n_steps = len(steps)
        predictions = np.zeros(n_steps, dtype=np.float32)
        targets = np.zeros(n_steps, dtype=np.float32)
        raw_times: list[Any] = []

        for i, step in enumerate(steps):
            pred, tgt, times = load_step(zio, stream, step, sample)
            # ensemble mean → zonal mean at target latitude
            predictions[i] = pred[lat_indices, ch_idx, :].mean(axis=-1).mean()
            targets[i] = tgt[lat_indices, ch_idx, 0].mean()
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info("%s: %d steps for %s", label, n_steps, variable)

    return {
        "label": label,
        "predictions": predictions,
        "targets": targets,
        "datetimes": datetimes,
        "channel": variable,
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

    var_channel = data_list[0]["channel"]
    _logger.info("Creating polar map animation for %s…", var_channel)

    # Common time axis: use the run with the most frames
    reference_data = max(data_list, key=lambda d: len(d["datetimes"]))
    common_times = reference_data["datetimes"]
    _logger.info(
        "  %d frames  %s → %s", len(common_times), common_times[0], common_times[-1]
    )

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
            pred = np.concatenate(
                [np.repeat(pred[:1], pad_frames, axis=0), pred], axis=0
            )
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
        [
            np.concatenate([d["predictions"].ravel(), d["targets"].ravel()])
            for d in plot_data
        ]
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

    all_diffs = np.concatenate(
        [(d["predictions"] - d["targets"]).ravel() for d in plot_data]
    )
    diff_abs = max(float(np.percentile(np.abs(all_diffs), 98)), 0.5)
    norm_diff = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)
    diff_label = (
        "Bias (m/s)"
        if variable in ("u", "v")
        else ("Bias (K)" if variable == "t" else "Bias")
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
        row_ax = [
            fig.add_subplot(gs[row, col], projection=proj_stereo) for col in range(3)
        ]
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
                ax.gridlines(
                    linestyle="--", alpha=0.3, linewidth=0.4, color="gray", zorder=4
                )

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
                    row_ax[col_idx].set_title(
                        hdr, fontsize=10, fontweight="bold", pad=3
                    )

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
                    bbox=dict(
                        boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="none"
                    ),
                )

        t = common_times[frame]
        days = (t - common_times[0]).total_seconds() / 86400
        ssw_flag = "  \u2605 SSW onset" if t.date() == ssw_date.date() else ""
        title_obj.set_text(
            f"{var_channel} (model level {data_list[0]['model_level']})"
            f"  \u00b7  {t.strftime('%Y-%m-%d %H:%M')}  (+{days:.0f}\u2009d){ssw_flag}"
        )
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(common_times), interval=1000 // fps, blit=False
    )

    output_file = output_dir / f"polar_map_animation_{var_channel}.mp4"
    _logger.info("Saving animation to %s…", output_file)
    anim.save(output_file, writer="ffmpeg", fps=fps, dpi=110)
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

    channel = data_list[0]["channel"]
    _logger.info("Plotting surface impact for %s…", channel)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, data in enumerate(data_list):
        datetimes = data["datetimes"]
        days = np.array(
            [(dt - datetimes[0]).total_seconds() / 86400 for dt in datetimes]
        )
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

    ax1.set_ylabel(_variable_label(channel), fontsize=12)
    ax1.set_title(
        f"Surface Impact of SSW Event: {channel} at 60°N",
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
        max(
            (dt - data["datetimes"][0]).total_seconds() / 86400
            for dt in data["datetimes"]
        )
        for data in data_list
    )
    ax2.set_xticks(np.arange(0, max_days + 1, 7))

    plt.tight_layout()
    output_file = output_dir / f"surface_impact_{channel}_60N.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    _logger.info("Saved: %s", output_file)
    plt.close()


def _variable_label(channel: str) -> str:
    _labels = {
        "z_500": "500 hPa Geopotential Height (m)",
        "z_850": "850 hPa Geopotential Height (m)",
        "2t": "2m Temperature (K)",
        "10u": "10m Zonal Wind (m/s)",
        "10v": "10m Meridional Wind (m/s)",
        "sp": "Surface Pressure (Pa)",
        "t_850": "850 hPa Temperature (K)",
        "u_850": "850 hPa Zonal Wind (m/s)",
    }
    return _labels.get(channel, channel)


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

    parser.add_argument(
        "--sample", type=int, default=0, help="Sample index (default: 0)."
    )
    parser.add_argument(
        "--model-level",
        type=int,
        default=55,
        help="Model level for stratospheric maps (default: 55 ≈ 77 hPa; use 30 for ~12 hPa).",
    )
    parser.add_argument(
        "--surface-var",
        type=str,
        default="z_500",
        help="Surface/tropospheric variable for impact plot (default: z_500).",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Animation frames per second."
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip polar map animations; still creates the surface impact plot.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Resolve run specs from config or CLI
    if args.validations_config:
        cfg = load_validations_config(args.validations_config)
        # Deduplicate zarr IDs while keeping first label
        seen: set[str] = set()
        run_specs = []
        for label, v in cfg.items():
            if v["id"] not in seen:
                seen.add(v["id"])
                run_specs.append(
                    {
                        "id": v["id"],
                        "label": label,
                        "sample": v.get("sample", args.sample),
                    }
                )
    else:
        run_specs = [
            {"id": rid, "label": rid, "sample": args.sample} for rid in args.run_ids
        ]

    # -----------------------------------------------------------------------
    # Polar map animations
    # -----------------------------------------------------------------------
    if not args.skip_animation:
        u_data: list[dict[str, Any]] = []
        t_data: list[dict[str, Any]] = []

        for spec in run_specs:
            zarr_path = args.data_dir / spec["id"] / _ZARR_FNAME
            if not zarr_path.exists():
                _logger.warning("zarr not found, skipping: %s", zarr_path)
                continue
            for var, bucket in (("u", u_data), ("t", t_data)):
                result = extract_polar_maps(
                    zarr_path, spec["label"], args.model_level, var, spec["sample"]
                )
                if result is not None:
                    bucket.append(result)

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
        zarr_path = args.data_dir / spec["id"] / _ZARR_FNAME
        if not zarr_path.exists():
            continue
        result = extract_surface_variable(
            zarr_path, spec["label"], args.surface_var, sample=spec["sample"]
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
