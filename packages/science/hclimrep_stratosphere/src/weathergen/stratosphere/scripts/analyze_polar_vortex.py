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
import pandas as pd

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
# Climatology
# ---------------------------------------------------------------------------


def load_climatology_zonal_mean(
    climatology_path: Path,
    channels: list[str],
    datetimes: list[datetime],
    target_latitude: float = 60.0,
) -> dict[str, np.ndarray]:
    """
    Load zonal mean climatology at *target_latitude* for each datetime.

    Matches by day-of-year + hour so the result is independent of forecast year.
    Returns a dict mapping channel -> float array (one value per datetime).
    """
    import xarray as xr

    _logger.info("Loading climatology from %s …", climatology_path)
    clim = xr.open_zarr(climatology_path)

    # Grid-point indices near target latitude
    clim_coords = np.column_stack([clim.latitude.values, clim.longitude.values])
    lat_indices = find_latitude_indices(clim_coords, target_latitude)

    clim_channels = list(clim.channels.values)
    available_chs = [ch for ch in channels if ch in clim_channels]
    missing_chs = [ch for ch in channels if ch not in clim_channels]
    if missing_chs:
        _logger.warning("Channels not in climatology: %s", missing_chs)

    # Match each datetime to climatology time by DOY + hour
    clim_times = pd.to_datetime(clim.time.values)
    clim_doys = clim_times.dayofyear.values
    clim_hours = clim_times.hour.values

    time_indices: list[int] = []
    for dt in datetimes:
        ts = pd.Timestamp(dt)
        mask = (clim_doys == ts.dayofyear) & (clim_hours == ts.hour)
        idx = np.where(mask)[0]
        time_indices.append(int(idx[0]) if len(idx) > 0 else -1)

    # Bulk-load: unique time steps × needed channels × lat-band points
    unique_t = sorted({t for t in time_indices if t >= 0})
    ch_indices = [clim_channels.index(ch) for ch in available_chs]

    # clim.data shape: (time, channels, grid_points)
    clim_block = (
        clim.data
        .isel(time=unique_t, channels=ch_indices)
        .values[:, :, lat_indices]  # (n_t, n_ch, n_latpts)
    )
    zonal_means = clim_block.mean(axis=2)  # (n_t, n_ch)
    t_pos = {t: i for i, t in enumerate(unique_t)}

    result: dict[str, np.ndarray] = {}
    for ci, ch in enumerate(available_chs):
        vals = np.full(len(datetimes), np.nan, dtype=np.float64)
        for ti, t in enumerate(time_indices):
            if t >= 0:
                vals[ti] = zonal_means[t_pos[t], ci]
        result[ch] = vals
    for ch in missing_chs:
        result[ch] = np.full(len(datetimes), np.nan, dtype=np.float64)

    return result


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_zonal_wind(
    zarr_path: Path,
    label: str,
    channels_override: list[str] | None = None,
    sample: int = 0,
    target_latitude: float = 60.0,
    climatology_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Extract zonal mean u-wind at *target_latitude* from a validation zarr store.

    Args:
        zarr_path:         Path to the ``.zarr`` store.
        label:             Human-readable label for logging/plotting.
        channels_override: Explicit list of channel names. Auto-detected when ``None``.
        sample:            Ensemble member index (default 0).
        target_latitude:   Latitude for zonal mean (default 60°N).
        climatology_path:  If provided, subtract the DOY climatology mean to
                           return anomalies instead of absolute values.

    Returns:
        Dict with keys ``label``, ``channels`` (dict per channel with
        ``predictions`` / ``targets``), ``times``, ``datetimes``, ``latitude``,
        ``is_anomaly``; or ``None`` on failure.
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

        preds_by_ch: dict[str, list[np.ndarray]] = {ch: [] for ch in available}
        targets_by_ch: dict[str, list[float]] = {ch: [] for ch in available}
        times_list: list[Any] = []

        for step in steps:
            pred, tgt, times = load_step(zio, stream_name, step, sample)
            pred3 = np.atleast_3d(pred)  # (n_pts, n_ch, n_ens)
            for ch, ch_idx in available.items():
                # Mean over lat band, keep all ensemble members → (n_ens,)
                preds_by_ch[ch].append(pred3[lat_indices, ch_idx, :].mean(axis=0))
                # target may have no ens dim in older stores — handle both shapes
                tgt_slice = (
                    tgt[lat_indices, ch_idx, 0] if tgt.ndim == 3 else tgt[lat_indices, ch_idx]
                )
                targets_by_ch[ch].append(float(np.mean(tgt_slice)))
            times_list.append(times[0])

    times_arr = np.array(times_list)
    datetimes = convert_times_to_datetime(times_arr)

    _logger.info("  %d forecast steps, %s → %s", len(steps), datetimes[0], datetimes[-1])

    # preds_np[ch]: (n_steps, n_ens) — one row per forecast step, one col per ensemble member
    preds_np = {ch: np.stack(preds_by_ch[ch], axis=0) for ch in available}
    targets_np = {ch: np.array(targets_by_ch[ch]) for ch in available}

    # Always store absolute values so that detect_ssw_reversal (u < 0 criterion)
    # works correctly.  Anomalies are kept separately for optional plotting.
    clim_means: dict[str, np.ndarray] = {}
    if climatology_path is not None:
        clim_means = load_climatology_zonal_mean(
            climatology_path, list(available.keys()), datetimes, target_latitude
        )

    return {
        "label": label,
        "sample": sample,
        "color": None,  # filled by caller
        "is_anomaly": climatology_path is not None,
        "channels": {
            ch: {
                "predictions": preds_np[ch],
                "targets": targets_np[ch],
                # anomaly arrays present only when climatology was supplied
                **(
                    {
                        # clim_means[ch] is (n_steps,); broadcast over ens dim
                        "pred_anomaly": preds_np[ch] - clim_means[ch][:, None],
                        "tgt_anomaly": targets_np[ch] - clim_means[ch],
                    }
                    if ch in clim_means
                    else {}
                ),
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


def _merge_targets(
    data_list: list[dict[str, Any]], ch: str, use_anomaly: bool = False
) -> tuple[list[datetime], list[float]]:
    """
    Merge ERA5 target time series from multiple runs into one.

    Uses anomaly arrays when *use_anomaly* is True, otherwise absolute values.
    """
    seen: dict[datetime, float] = {}
    for d in data_list:
        if ch not in d["channels"]:
            continue
        ch_data = d["channels"][ch]
        vals = ch_data.get("tgt_anomaly", ch_data["targets"]) if use_anomaly else ch_data["targets"]
        for dt, val in zip(d["datetimes"], vals, strict=False):
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
    use_anomaly: bool = False,
) -> None:
    """
    Plot zonal mean u-wind time series: one panel with all forecasts and a
    single merged ERA5 target line.

    Args:
        data_list:   List of dicts returned by :func:`extract_zonal_wind`.
        output_dir:  Directory for saved figures.
        ssw_date:    Reference SSW date for vertical line.
        use_anomaly: Plot climatology anomalies instead of absolute values.
                     Requires that ``extract_zonal_wind`` was called with a
                     ``climatology_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_channels: set[str] = set()
    for d in data_list:
        all_channels.update(d["channels"].keys())
    sorted_channels = sorted(all_channels)

    fallback_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

    for ch in sorted_channels:
        fig, ax = plt.subplots(figsize=(14, 5))

        # One prediction line per run (ensemble mean + spread shading when n_ens > 1)
        for i, d in enumerate(data_list):
            if ch not in d["channels"]:
                continue
            color = d.get("color") or fallback_colors[i % len(fallback_colors)]
            ch_data = d["channels"][ch]
            plot_vals = (
                ch_data.get("pred_anomaly", ch_data["predictions"])
                if use_anomaly
                else ch_data["predictions"]
            )
            # plot_vals may be (n_steps,) or (n_steps, n_ens)
            plot_vals = np.asarray(plot_vals)
            if plot_vals.ndim == 2 and plot_vals.shape[1] > 1:
                # ens_mean = plot_vals.mean(axis=1)
                # ens_min = plot_vals.min(axis=1)
                # ens_max = plot_vals.max(axis=1)
                # ax.fill_between(
                #     d["datetimes"], ens_min, ens_max, color=color, alpha=0.2
                # )
                # ax.plot(
                #     d["datetimes"], ens_mean, color=color, lw=2, label=d["label"]
                # )
                for ens_idx in range(plot_vals.shape[1]):
                    ax.plot(
                        d["datetimes"],
                        plot_vals[:, ens_idx],
                        color=color,
                        lw=1,
                        alpha=0.5,
                        label=f"{d['label']} (ens {ens_idx})" if ens_idx == 0 else None,
                    )
            else:
                ax.plot(
                    d["datetimes"],
                    plot_vals.squeeze(),
                    color=color,
                    lw=2,
                    label=d["label"],
                )

        # Single merged ERA5 target line
        tgt_times, tgt_vals = _merge_targets(data_list, ch, use_anomaly=use_anomaly)
        if tgt_times:
            ax.plot(tgt_times, tgt_vals, color="k", lw=2, ls="--", label="ERA5")

        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.7)
        all_dates = [dt for d in data_list for dt in d["datetimes"]]
        if all_dates and min(all_dates) <= ssw_date <= max(all_dates):
            ax.axvline(ssw_date, color="red", ls="-.", lw=1.5, alpha=0.7, label="SSW date")

        ylabel = "u-wind anomaly (m/s)" if use_anomaly else "u-wind (m/s)"
        title_suffix = " anomaly" if use_anomaly else ""
        ax.set_ylabel(ylabel)
        ax.set_title(f"Zonal mean u{title_suffix} at 60°N — {ch}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        tag = event_tag or ssw_date.strftime("%b%Y").lower()
        anom_suffix = "_anomaly" if use_anomaly else ""
        out = output_dir / f"zonal_wind_60N_{ch}_{tag}{anom_suffix}.png"
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
    parser.add_argument("--climatology", type=Path, default=None,
                        help="Path to climatology zarr for anomaly computation.")
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
            climatology_path=args.climatology,
        )
        if data is None:
            continue
        data["color"] = color
        all_data.append(data)

        for ch, ch_data in data["channels"].items():
            pred_arr = np.asarray(ch_data["predictions"])
            # Use ensemble mean for SSW detection when multiple members are present
            pred_for_ssw = pred_arr.mean(axis=1) if pred_arr.ndim == 2 else pred_arr
            pred_ssw = detect_ssw_reversal(pred_for_ssw, data["datetimes"])
            tgt_ssw = detect_ssw_reversal(ch_data["targets"], data["datetimes"])
            _logger.info("%s / %s  pred=%s  target=%s", label, ch, pred_ssw, tgt_ssw)

    if all_data:
        event_tag = ""
        if args.validations_config is not None:
            stem = args.validations_config.stem  # e.g. "ssw_feb2018"
            parts = stem.split("_", 1)
            event_tag = parts[1] if len(parts) == 2 else stem
        plot_zonal_wind_comparison(all_data, args.output_dir, event_tag=event_tag)
        if any(d.get("is_anomaly") for d in all_data):
            plot_zonal_wind_comparison(
                all_data, args.output_dir, event_tag=event_tag, use_anomaly=True
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
