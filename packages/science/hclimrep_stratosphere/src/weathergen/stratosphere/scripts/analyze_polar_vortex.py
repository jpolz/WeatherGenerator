#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Vortex Analysis for SSW Event Prediction.

Analyses zonal mean zonal wind at 60°N to detect Sudden Stratospheric Warming
events and compares model predictions against ERA5 targets.

Usage::

    ssw-analyze polar-vortex \\
        --validations-config eval_config/validations.yml \\
        --data-dir /path/to/validation/data \\
        --output-dir plots/polar_vortex \\
        --channels u_29 u_30

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_vortex \\
        --run-ids qq8xsoeh j7ns9146 \\
        --data-dir ./data
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from weathergen.stratosphere.diagnostics import detect_ssw_reversal
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    find_latitude_indices,
    get_channels,
    get_coords,
    get_forecast_steps,
    get_stream,
    load_step,
    open_validation,
)

_logger = logging.getLogger(__name__)

# Default zarr filename produced by WeatherGenerator inference
_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# Default channels to extract (priority order; first match per variable is used)
_DEFAULT_U_CHANNELS = ["u_29", "u_30", "u_55"]  # ERA5ml
_DEFAULT_U_CHANNELS_PL = ["u_10", "u_50"]  # ERA5pl

# Known SSW event date for reference lines
SSW_DATE = datetime(2018, 2, 12)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_zonal_wind(
    zarr_path: Path,
    label: str,
    channels_override: list[str] | None = None,
    sample: int = 0,
    target_latitude: float = 60.0,
) -> dict[str, Any] | None:
    """
    Extract zonal mean u-wind at *target_latitude* from a validation zarr store.

    Args:
        zarr_path:         Path to the ``.zarr`` store.
        label:             Human-readable label for logging/plotting.
        channels_override: Explicit list of channel names. Auto-detected when ``None``.
        sample:            Ensemble member index (default 0).
        target_latitude:   Latitude for zonal mean (default 60°N).

    Returns:
        Dict with keys ``label``, ``channels`` (dict per channel with
        ``predictions`` / ``targets``), ``times``, ``datetimes``, ``latitude``;
        or ``None`` on failure.
    """
    _logger.info("Extracting zonal wind for %s (sample %d) …", label, sample)

    with open_validation(zarr_path) as zio:
        stream_name = get_stream(zio)
        steps = get_forecast_steps(zio)
        channels = get_channels(zio, stream_name, sample)
        coords = get_coords(zio, stream_name, sample)

        # Select channels
        defaults = _DEFAULT_U_CHANNELS if stream_name == "ERA5ml" else _DEFAULT_U_CHANNELS_PL
        wanted = channels_override or defaults
        available = {ch: channels.index(ch) for ch in wanted if ch in channels}

        if not available:
            _logger.warning(
                "None of %s found in %s. Available u-channels: %s",
                wanted,
                label,
                [c for c in channels if c.startswith("u_")],
            )
            return None

        lat_indices = find_latitude_indices(coords, target_latitude)

        preds_by_ch: dict[str, list[float]] = {ch: [] for ch in available}
        targets_by_ch: dict[str, list[float]] = {ch: [] for ch in available}
        times_list: list[Any] = []

        for step in steps:
            pred, tgt, times = load_step(zio, stream_name, step, sample)
            for ch, ch_idx in available.items():
                preds_by_ch[ch].append(float(np.mean(pred[lat_indices, ch_idx, 0])))
                # target may have no ens dim in older stores — handle both shapes
                tgt_slice = (
                    tgt[lat_indices, ch_idx, 0] if tgt.ndim == 3 else tgt[lat_indices, ch_idx]
                )
                targets_by_ch[ch].append(float(np.mean(tgt_slice)))
            times_list.append(times[0])

    times_arr = np.array(times_list)
    datetimes = convert_times_to_datetime(times_arr)

    _logger.info("  %d forecast steps, %s → %s", len(steps), datetimes[0], datetimes[-1])

    return {
        "label": label,
        "sample": sample,
        "color": None,  # filled by caller
        "channels": {
            ch: {
                "predictions": np.array(preds_by_ch[ch]),
                "targets": np.array(targets_by_ch[ch]),
            }
            for ch in available
        },
        "times": times_arr,
        "datetimes": datetimes,
        "latitude": target_latitude,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _merge_targets(data_list: list[dict[str, Any]], ch: str) -> tuple[list[datetime], list[float]]:
    """
    Merge ERA5 target time series from multiple runs into one.

    Runs share the same underlying ERA5 data where their time windows overlap,
    so duplicates are simply deduplicated (first-seen value is kept).
    """
    seen: dict[datetime, float] = {}
    for d in data_list:
        if ch not in d["channels"]:
            continue
        for dt, val in zip(d["datetimes"], d["channels"][ch]["targets"], strict=False):
            if dt not in seen:
                seen[dt] = val
    if not seen:
        return [], []
    sorted_items = sorted(seen.items())
    return [t for t, _ in sorted_items], [v for _, v in sorted_items]


def plot_zonal_wind_comparison(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
    event_tag: str = "",
) -> None:
    """
    Plot zonal mean u-wind time series: one panel with all forecasts and a
    single merged ERA5 target line.

    Args:
        data_list:  List of dicts returned by :func:`extract_zonal_wind`.
        output_dir: Directory for saved figures.
        ssw_date:   Reference SSW date for vertical line.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_channels: set[str] = set()
    for d in data_list:
        all_channels.update(d["channels"].keys())
    sorted_channels = sorted(all_channels)

    fallback_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for ch in sorted_channels:
        fig, ax = plt.subplots(figsize=(14, 5))

        # One prediction line per run
        for i, d in enumerate(data_list):
            if ch not in d["channels"]:
                continue
            color = d.get("color") or fallback_colors[i % len(fallback_colors)]
            ax.plot(
                d["datetimes"],
                d["channels"][ch]["predictions"],
                color=color,
                lw=2,
                label=d["label"],
            )

        # Single merged ERA5 target line
        tgt_times, tgt_vals = _merge_targets(data_list, ch)
        if tgt_times:
            ax.plot(tgt_times, tgt_vals, color="k", lw=2, ls="--", label="ERA5")

        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.7)
        all_dates = [dt for d in data_list for dt in d["datetimes"]]
        if all_dates and min(all_dates) <= ssw_date <= max(all_dates):
            ax.axvline(ssw_date, color="red", ls="-.", lw=1.5, alpha=0.7, label="SSW date")

        ax.set_ylabel("u-wind (m/s)")
        ax.set_title(f"Zonal mean u at 60°N — {ch}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        tag = event_tag or ssw_date.strftime("%b%Y").lower()
        out = output_dir / f"zonal_wind_60N_{ch}_{tag}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_run_specs(
    args: argparse.Namespace,
) -> list[tuple[str, str, int, str | None]]:
    """Return list of (label, validation_id, sample, color) tuples."""
    if args.validations_config is not None:
        from weathergen.stratosphere.config import load_validations_config

        cfg = load_validations_config(args.validations_config)
        return [(lbl, spec["id"], spec["sample"], spec.get("color")) for lbl, spec in cfg.items()]
    if args.run_ids:
        return [(vid, vid, 0, None) for vid in args.run_ids]
    raise SystemExit("Provide --run-ids or --validations-config")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/polar_vortex"))
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--validations-config", type=Path, default=None)
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--latitude", type=float, default=60.0)
    args = parser.parse_args(argv)

    run_specs = _build_run_specs(args)
    all_data: list[dict[str, Any]] = []

    for label, vid, sample, color in run_specs:
        zarr_path = args.data_dir / vid / _ZARR_FNAME
        if not zarr_path.exists():
            _logger.warning("Zarr not found: %s", zarr_path)
            continue

        data = extract_zonal_wind(
            zarr_path,
            label,
            channels_override=args.channels,
            sample=sample,
            target_latitude=args.latitude,
        )
        if data is None:
            continue
        data["color"] = color
        all_data.append(data)

        for ch, ch_data in data["channels"].items():
            pred_ssw = detect_ssw_reversal(ch_data["predictions"], data["datetimes"])
            tgt_ssw = detect_ssw_reversal(ch_data["targets"], data["datetimes"])
            _logger.info("%s / %s  pred=%s  target=%s", label, ch, pred_ssw, tgt_ssw)

    if all_data:
        event_tag = ""
        if args.validations_config is not None:
            stem = args.validations_config.stem  # e.g. "ssw_feb2018"
            parts = stem.split("_", 1)
            event_tag = parts[1] if len(parts) == 2 else stem
        plot_zonal_wind_comparison(all_data, args.output_dir, event_tag=event_tag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
