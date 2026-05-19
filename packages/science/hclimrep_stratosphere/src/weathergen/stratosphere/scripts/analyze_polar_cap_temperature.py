#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Cap Temperature Analysis for SSW Event Prediction.

Analyses area-weighted mean temperature poleward of 60°N to quantify
stratospheric warming during SSW events.  Produces one time-series panel
per available temperature level, comparing model predictions against ERA5
targets across multiple experiments.

Usage::

    ssw-analyze polar-cap-temperature \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/polar_cap_temperature

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_cap_temperature \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results
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
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.diagnostics import detect_warming_event
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    find_polar_cap_indices,
    get_area_weights,
    get_channels,
    get_coords,
    get_forecast_steps,
    get_stream,
    open_validation,
)
from weathergen.stratosphere.levels import channel_pressure

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# Model level range considered "stratospheric" (B_k = 0 for levels ≤ 60)
_STRAT_LEVEL_MAX = 60

# Fallback for ERA5pl streams
_T_CHANNELS_PL = ["t_10", "t_50"]

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_polar_cap_temperature(
    zarr_path: Path,
    label: str,
    min_latitude: float = 60.0,
    sample: int = 0,
    t_channels_override: list[str] | None = None,
) -> dict[str, Any] | None:
    """
    Extract area-weighted polar cap mean temperature poleward of *min_latitude*.

    Searches for available temperature channels in the data stream and
    returns a time series for each found channel.

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

    Returns
    -------
    Dict with keys ``label``, ``channels``, ``times``, ``datetimes``,
    ``min_latitude``, ``n_points``; or ``None`` if no temperature channels
    are available.
    """
    _logger.info("%s: extracting polar cap temperature (lat≥%.1f°N)…", label, min_latitude)

    with open_validation(zarr_path) as zio:
        stream_name = get_stream(zio)
        steps = get_forecast_steps(zio, skip_source_step=True)
        channels = get_channels(zio, stream_name, sample)
        coords = get_coords(zio, stream_name, sample)

        # Select candidate channels based on stream type
        if t_channels_override is not None:
            candidates = t_channels_override
        elif stream_name == "ERA5ml":
            # Auto-detect all t channels at stratospheric model levels (B_k ≈ 0)
            def _is_strat_t(ch: str) -> bool:
                if not ch.startswith("t_"):
                    return False
                try:
                    return int(ch.split("_", 1)[1]) <= _STRAT_LEVEL_MAX
                except ValueError:
                    return False

            # Sort by ascending model level number (descending pressure → top-down)
            strat_t_channels = sorted(
                [ch for ch in channels if _is_strat_t(ch)],
                key=lambda c: int(c.split("_", 1)[1]),
            )
            candidates = strat_t_channels if strat_t_channels else ["t_29", "t_30"]
        else:
            candidates = _T_CHANNELS_PL

        available_channels: dict[str, int] = {
            ch: channels.index(ch) for ch in candidates if ch in channels
        }

        if not available_channels:
            _logger.warning(
                "%s: no stratospheric t channels found in %s.  Available t channels: %s",
                label,
                stream_name,
                [c for c in channels if c.startswith("t_")],
            )
            return None

        _logger.info("%s: using channels %s", label, list(available_channels))

        # Spatial weights for polar cap
        polar_indices = find_polar_cap_indices(coords, min_latitude)
        weights = get_area_weights(coords, polar_indices)

        # Accumulate weighted-mean time series per channel
        preds: dict[str, list[float]] = {ch: [] for ch in available_channels}
        tgts: dict[str, list[float]] = {ch: [] for ch in available_channels}
        times_list: list = []

        for step in steps:
            pred_arr, tgt_arr, times = _load_step_data(zio, stream_name, step, sample)

            for ch, ch_idx in available_channels.items():
                pred_polar = pred_arr[polar_indices, ch_idx, 0]
                tgt_polar = tgt_arr[polar_indices, ch_idx, 0]
                preds[ch].append(float(np.sum(pred_polar * weights)))
                tgts[ch].append(float(np.sum(tgt_polar * weights)))

            times_list.append(times[0])

    times_arr = np.array(times_list)
    datetimes = convert_times_to_datetime(times_arr)

    channels_data: dict[str, dict[str, np.ndarray]] = {
        ch: {
            "predictions": np.array(preds[ch]),
            "targets": np.array(tgts[ch]),
        }
        for ch in available_channels
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
        "channels": channels_data,
        "times": times_arr,
        "datetimes": datetimes,
        "min_latitude": min_latitude,
        "n_points": int(len(polar_indices)),
    }


def _load_step_data(
    zio: Any,
    stream: str,
    step: int,
    sample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(pred, tgt, times)`` with pred/tgt always 3-D (pts, channels, ens)."""
    from weathergen.stratosphere.io import load_step

    pred, tgt, times = load_step(zio, stream, step, sample)
    pred = np.atleast_3d(pred)
    tgt = np.atleast_3d(tgt)
    return pred, tgt, times


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_polar_cap_temperature(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Plot polar cap mean temperature time series, one panel per pressure level.

    Each panel shows predictions (solid) and ERA5 targets (dashed) for every
    experiment in *data_list*.  A vertical reference line marks *ssw_date* if
    it falls within the plotted period.

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

    # Collect all unique channels that appear in any experiment
    all_channels: list[str] = sorted(
        {ch for data in data_list for ch in data["channels"]}
    )
    if not all_channels:
        _logger.warning("No channels to plot.")
        return

    n_panels = len(all_channels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels), squeeze=False)
    axes = axes.flatten()

    for ax, channel in zip(axes, all_channels):
        # Build a human-readable pressure label
        p_hpa = channel_pressure(channel)
        level_label = f"{channel}  ({p_hpa:.0f} hPa)" if p_hpa is not None else channel

        # Plot each experiment that contains this channel
        for color_idx, data in enumerate(data_list):
            if channel not in data["channels"]:
                continue

            label = data["label"]
            datetimes = data["datetimes"]
            ch_data = data["channels"][channel]
            color = _COLORS[color_idx % len(_COLORS)]

            ax.plot(
                datetimes,
                ch_data["predictions"],
                color=color,
                linestyle="-",
                linewidth=2.5,
                label=f"{label} prediction",
                alpha=0.85,
            )
            ax.plot(
                datetimes,
                ch_data["targets"],
                color=color,
                linestyle="--",
                linewidth=2.0,
                label=f"{label} ERA5",
                alpha=0.6,
            )

        # SSW reference line (only if date falls in range)
        all_dts = [dt for data in data_list for dt in data["datetimes"]]
        if all_dts and min(all_dts) <= ssw_date <= max(all_dts):
            ax.axvline(
                x=ssw_date,
                color="red",
                linestyle="-.",
                linewidth=2,
                alpha=0.7,
                label=f"SSW onset ({ssw_date.strftime('%b %d, %Y')})",
                zorder=1,
            )

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Temperature (K)", fontsize=12)
        ax.set_title(
            f"Polar Cap Temperature (≥{data_list[0]['min_latitude']:.0f}°N) — {level_label}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9, ncol=2)
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

    parser = argparse.ArgumentParser(
        description="Polar cap temperature analysis for SSW events."
    )
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
        "--t-channels",
        nargs="+",
        metavar="CHANNEL",
        default=None,
        help=(
            "Explicit temperature channels to plot (e.g. t_29 t_51 t_5).  "
            "Default: all stratospheric t channels auto-detected from the zarr stream."
        ),
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference line (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("POLAR CAP TEMPERATURE ANALYSIS")
    print("=" * 60)

    cfg = load_validations_config(args.validations_config)
    run_specs = [(label, spec["id"], spec["sample"]) for label, spec in cfg.items()]

    data_list: list[dict] = []
    results: dict[str, Any] = {}

    for label, run_id, sample in run_specs:
        zarr_path = args.data_dir / run_id / _ZARR_FNAME

        if not zarr_path.exists():
            _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
            continue

        data = extract_polar_cap_temperature(
            zarr_path, label, args.min_latitude, sample,
            t_channels_override=args.t_channels,
        )

        if data is None:
            _logger.warning("Skipping %s — no suitable temperature channels.", label)
            continue

        data_list.append(data)

        # Detect warming events per channel
        results[label] = {}
        for ch, ch_data in data["channels"].items():
            pred_event = detect_warming_event(ch_data["predictions"], data["datetimes"])
            tgt_event = detect_warming_event(ch_data["targets"], data["datetimes"])

            _logger.info(
                "%s  %s  pred=%s  tgt=%s",
                label,
                ch,
                pred_event,
                tgt_event,
            )
            results[label][ch] = {
                "prediction": pred_event,
                "target": tgt_event,
            }

    if not data_list:
        print("No data extracted — check data-dir and validations config.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_polar_cap_temperature(data_list, args.output_dir, args.ssw_date)

    # Save warming-event summary
    json_path = args.output_dir / "polar_cap_temperature_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _logger.info("Saved %s", json_path)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
