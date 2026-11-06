#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Cap Temperature Analysis for SSW Event Prediction.

Analyses area-weighted mean temperature poleward of 60°N to quantify
stratospheric warming during SSW events.  Produces one time-series panel
per pressure level, comparing model predictions against ERA5 targets
across multiple experiments.

Multi-stream layout
-------------------
A forecast store may spread the atmospheric column over several streams
(e.g. ``ERA5pl`` with 50–1000 hPa and ``S2Spredictors`` with 1–70 hPa, or a
single legacy ``ERA5ml`` model-level stream).  Levels are therefore selected
in hPa via :func:`weathergen.stratosphere.io.discover_column`, which merges
all streams into one pressure axis; the stream and channel name of each
level are resolved automatically.  Default levels: 2, 10, 30 and 50 hPa
(falls back to every available level ≤ 100 hPa, e.g. only 50 hPa for the
``memory`` arm).

Usage::

    ssw-analyze polar-cap-temperature \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/polar_cap_temperature \\
        --levels 2 10 30 50

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_cap_temperature \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results

``--channels t_19 t_29`` (explicit channel names) is still accepted as a
legacy alias of ``--levels``.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from weathergen.stratosphere.config import (
    event_reference_date,
    group_validations_by_event,
    load_validations_config,
)
from weathergen.stratosphere.diagnostics import detect_warming_event, polar_cap_mean
from weathergen.stratosphere.io import (
    LevelRef,
    convert_times_to_datetime,
    discover_column,
    find_polar_cap_indices,
    get_coords,
    get_forecast_steps,
    load_levels,
    open_validation,
    resolve_channel_arg,
    zarr_store_path,
)
from weathergen.stratosphere.levels import format_pressure

_logger = logging.getLogger(__name__)

# Default pressure levels (hPa) for polar-cap temperature.
# Override with --levels (hPa) or the legacy --channels (channel names).
_DEFAULT_T_LEVELS_HPA = [2.0, 10.0, 30.0, 50.0]

# If none of the defaults exist, fall back to all levels at or above this pressure.
_FALLBACK_P_BOTTOM_HPA = 100.0

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def level_key(pressure_hpa: float) -> str:
    """Stable key / filename fragment for a level, e.g. ``'t_10hPa'``."""
    return f"t_{pressure_hpa:g}hPa"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _resolve_levels(
    zio: Any,
    label: str,
    sample: int,
    channels: list[str] | None,
    levels_hpa: list[float] | None,
) -> list[LevelRef]:
    """Resolve requested (or default) temperature levels to :class:`LevelRef` objects."""
    if channels or levels_hpa:
        return resolve_channel_arg(zio, "t", channels, levels_hpa, _DEFAULT_T_LEVELS_HPA, sample)

    col = discover_column(zio, "t", sample)
    refs = col.select(_DEFAULT_T_LEVELS_HPA)
    if not refs:
        refs = col.between(0.0, _FALLBACK_P_BOTTOM_HPA)
        if refs:
            _logger.warning(
                "%s: none of the default levels %s hPa available; using all levels ≤ %g hPa: %s",
                label,
                _DEFAULT_T_LEVELS_HPA,
                _FALLBACK_P_BOTTOM_HPA,
                [r.pressure for r in refs],
            )
    return refs


def extract_polar_cap_temperature(
    zarr_path: Path,
    label: str,
    min_latitude: float = 60.0,
    sample: int = 0,
    channels: list[str] | None = None,
    levels_hpa: list[float] | None = None,
    color: str | None = None,
) -> dict[str, Any] | None:
    """
    Extract area-weighted polar cap mean temperature poleward of *min_latitude*.

    Resolves the requested pressure levels across all streams of the store
    and returns one (ensemble-mean) time series per level.

    Parameters
    ----------
    zarr_path:
        Path to the ``.zip`` zarr validation store.
    label:
        Human-readable experiment label (used in plots).
    min_latitude:
        Southern boundary of the polar cap (°N).  Default 60°N.
    sample:
        Ensemble/sample index to extract.
    channels:
        Legacy explicit channel names (e.g. ``["t_19", "t_29"]``).
    levels_hpa:
        Pressure levels in hPa; ``None`` → :data:`_DEFAULT_T_LEVELS_HPA`.
    color:
        Optional plot colour (from the validations config), carried through
        to the returned dict for :func:`plot_polar_cap_temperature`.

    Returns
    -------
    Dict with keys ``label``, ``levels`` (pressure → ``{"predictions",
    "targets", "channel", "stream"}``), ``times``, ``datetimes``,
    ``min_latitude``, ``n_points``, ``color``; or ``None`` if no temperature
    level is available.
    """
    _logger.info("%s: extracting polar cap temperature (lat≥%.1f°N)…", label, min_latitude)

    with open_validation(zarr_path) as zio:
        refs = _resolve_levels(zio, label, sample, channels, levels_hpa)
        if not refs:
            _logger.warning("%s: no temperature levels found in streams %s", label, zio.streams)
            return None

        _logger.info(
            "%s: using levels %s", label, [f"{r.label} ({r.stream}/{r.channel})" for r in refs]
        )

        steps = get_forecast_steps(zio, skip_source_step=True)

        # Group levels by stream: one read per stream and step, each stream
        # reduced with its own coordinates.
        by_stream: dict[str, list[LevelRef]] = {}
        for ref in refs:
            by_stream.setdefault(ref.stream, []).append(ref)
        coords = {s: get_coords(zio, s, sample) for s in by_stream}
        n_points = len(find_polar_cap_indices(coords[refs[0].stream], min_latitude))

        preds: dict[float, list[float]] = {r.pressure: [] for r in refs}
        tgts: dict[float, list[float]] = {r.pressure: [] for r in refs}
        times_list: list = []

        for step in steps:
            step_time = None
            for stream, s_refs in by_stream.items():
                pred, tgt, times = load_levels(zio, s_refs, step, sample)  # (pts, lev, ens)
                step_time = times[0] if step_time is None else step_time
                for j, ref in enumerate(s_refs):
                    # polar_cap_mean over (n_ens, n_pts) → per-member means → ensemble mean
                    pm = polar_cap_mean(pred[:, j, :].T, coords[stream], min_latitude)
                    tm = polar_cap_mean(tgt[:, j, :].T, coords[stream], min_latitude)
                    preds[ref.pressure].append(float(np.mean(pm)))
                    tgts[ref.pressure].append(float(np.mean(tm)))
            times_list.append(step_time)

    times_arr = np.array(times_list)
    datetimes = convert_times_to_datetime(times_arr)

    levels_data: dict[float, dict[str, Any]] = {
        r.pressure: {
            "predictions": np.array(preds[r.pressure]),
            "targets": np.array(tgts[r.pressure]),
            "channel": r.channel,
            "stream": r.stream,
        }
        for r in refs
    }

    _logger.info(
        "%s: extracted %d steps, %s → %s",
        label,
        len(times_arr),
        datetimes[0],
        datetimes[-1],
    )

    return {
        "label": label,
        "levels": levels_data,
        "times": times_arr,
        "datetimes": datetimes,
        "min_latitude": min_latitude,
        "n_points": int(n_points),
        "color": color,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _merge_targets(
    data_list: list[dict[str, Any]], pressure: float
) -> tuple[list[datetime], list[float]]:
    """
    Merge ERA5 target time series from multiple runs into one, analogous to
    :func:`analyze_polar_vortex._merge_targets`.

    Runs sharing overlapping init windows report (near-)identical ERA5
    values for the same valid time, so a single merged line is a faithful
    (and far less cluttered) stand-in for plotting each run's own target.
    """
    seen: dict[datetime, float] = {}
    for d in data_list:
        if pressure not in d["levels"]:
            continue
        lv_data = d["levels"][pressure]
        for dt, val in zip(d["datetimes"], lv_data["targets"], strict=True):
            if dt not in seen:
                seen[dt] = val
    if not seen:
        return [], []
    sorted_items = sorted(seen.items())
    return [t for t, _ in sorted_items], [v for _, v in sorted_items]


def plot_polar_cap_temperature(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Plot polar cap mean temperature time series, one panel per pressure level.

    Each panel shows one prediction line per experiment (coloured per the
    validations config, same as :func:`analyze_polar_vortex.plot_zonal_wind_comparison`)
    and a single merged ERA5 target line.  A vertical reference line marks
    *ssw_date* if it falls within the plotted period.

    Parameters
    ----------
    data_list:
        List of dicts returned by :func:`extract_polar_cap_temperature`.
    output_dir:
        Directory where the PNG is written.
    ssw_date:
        Reference date for the SSW onset marker.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # All pressure levels that appear in any experiment, top → bottom
    all_pressures: list[float] = sorted({p for data in data_list for p in data["levels"]})
    if not all_pressures:
        _logger.warning("No levels to plot.")
        return

    n_panels = len(all_pressures)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), squeeze=False)
    axes = axes.flatten()

    for ax, pressure in zip(axes, all_pressures, strict=True):
        level_label = f"t {format_pressure(pressure)}"

        # One prediction line per experiment, using the experiment's own
        # config colour when set (falls back to a cyclic palette otherwise).
        for color_idx, data in enumerate(data_list):
            if pressure not in data["levels"]:
                continue

            datetimes = data["datetimes"]
            lv_data = data["levels"][pressure]
            color = data.get("color") or _COLORS[color_idx % len(_COLORS)]

            ax.plot(
                datetimes,
                lv_data["predictions"],
                color=color,
                linestyle="-",
                linewidth=2.5,
                alpha=0.85,
            )

        # Single merged ERA5 target line
        tgt_times, tgt_vals = _merge_targets(data_list, pressure)
        if tgt_times:
            ax.plot(tgt_times, tgt_vals, color="k", lw=2, ls="--", alpha=0.8)

        # SSW reference line (only if date falls in range)
        all_dts = [dt for data in data_list for dt in data["datetimes"]]
        if all_dts and min(all_dts) <= ssw_date <= max(all_dts):
            ax.axvline(
                x=ssw_date,
                color="red",
                linestyle="-.",
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Temperature (K)", fontsize=12)
        ax.set_title(
            f"Polar Cap Temperature (≥{data_list[0]['min_latitude']:.0f}°N) — {level_label}",
            fontsize=13,
            fontweight="bold",
        )
        # Legend disabled for now — too many experiments make it unreadable
        # (see analyze_polar_vortex.plot_zonal_wind_comparison for the same
        # change). Re-enable once a more compact scheme (e.g. colorbar by
        # init date) is in place.
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_file = output_dir / "polar_cap_temperature_by_level.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out_file)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Polar cap temperature analysis for SSW events.")
    parser.add_argument(
        "--validations-config",
        type=Path,
        required=True,
        help="YAML validations config (see config/evaluate/).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing zarr result stores.  [default: results]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/polar_cap_temperature"),
        help="Directory for output plots and JSON.  [default: plots/polar_cap_temperature]",
    )
    parser.add_argument(
        "--min-latitude",
        type=float,
        default=60.0,
        help="Southern boundary of polar cap (°N).  [default: 60.0]",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        metavar="HPA",
        default=None,
        help=(
            "Pressure levels (hPa) to plot; the nearest available level in any "
            "stream is used.  [default: 2 10 30 50, falling back to all levels "
            "≤ 100 hPa]"
        ),
    )
    parser.add_argument(
        "--channels",
        "--t-channels",
        dest="channels",
        nargs="+",
        metavar="CHANNEL",
        default=None,
        help="Legacy alias of --levels: explicit channel names (e.g. t_19 t_29 or t_10 t_50).",
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference line (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    args = parser.parse_args(argv)

    _logger.info("Polar cap temperature analysis")

    cfg = load_validations_config(args.validations_config)
    events = group_validations_by_event(cfg)
    single_event = len(events) == 1

    results: dict[str, Any] = {}
    any_data = False

    for event_name, event_cfg in events.items():
        event_ssw_date = event_reference_date(event_cfg, args.ssw_date)
        # Keep the flat (un-nested) output layout when there is only one
        # event, so existing single-event configs are unaffected.
        event_output_dir = args.output_dir if single_event else args.output_dir / event_name

        run_specs = [
            (label, spec["id"], spec["sample"], spec.get("rank", 0), spec.get("color"))
            for label, spec in event_cfg.items()
        ]

        data_list: list[dict] = []

        for label, run_id, sample, rank, color in run_specs:
            zarr_path = zarr_store_path(args.data_dir, run_id, rank)

            if not zarr_path.exists():
                _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
                continue

            data = extract_polar_cap_temperature(
                zarr_path,
                label,
                args.min_latitude,
                sample,
                channels=args.channels,
                levels_hpa=args.levels,
                color=color,
            )

            if data is None:
                _logger.warning("Skipping %s — no suitable temperature levels.", label)
                continue

            data_list.append(data)

            # Detect warming events per level
            results.setdefault(event_name, {})[label] = {}
            for pressure, lv_data in data["levels"].items():
                pred_event = detect_warming_event(lv_data["predictions"], data["datetimes"])
                tgt_event = detect_warming_event(lv_data["targets"], data["datetimes"])

                _logger.info(
                    "[%s] %s  %s  pred=%s  tgt=%s",
                    event_name,
                    label,
                    format_pressure(pressure),
                    pred_event,
                    tgt_event,
                )
                results[event_name][label][level_key(pressure)] = {
                    "pressure_hpa": pressure,
                    "channel": lv_data["channel"],
                    "stream": lv_data["stream"],
                    "prediction": pred_event,
                    "target": tgt_event,
                }

        if not data_list:
            _logger.warning("[%s] No data extracted — skipping event.", event_name)
            continue

        any_data = True
        event_output_dir.mkdir(parents=True, exist_ok=True)
        plot_polar_cap_temperature(data_list, event_output_dir, event_ssw_date)

    if not any_data:
        _logger.warning("No data extracted — check data-dir and validations config.")
        return

    # Save warming-event summary (keyed by event when there is more than one)
    results_out: Any = results[next(iter(results))] if single_event and results else results
    json_path = args.output_dir / "polar_cap_temperature_results.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(results_out, fh, indent=2, default=str)
    _logger.info("Saved %s", json_path)

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
