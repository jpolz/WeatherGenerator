#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Vertical Structure Analysis for SSW Event Prediction.

Produces height–time cross-sections showing:
  - Absolute zonal wind / polar-cap temperature vs time and pressure level.
  - Anomalies relative to an initial reference period (or a separate reference run).
  - Downward propagation comparison across experiments.

Two output types are generated for each variable:
  1. Per-experiment 2×2 panel (abs prediction | abs target | pred anomaly | tgt anomaly).
  2. Multi-experiment 2-column comparison (pred anomaly | target anomaly per row).

Usage::

    ssw-analyze vertical-structure \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/vertical_structure

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_vertical_structure \\
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
from matplotlib.colors import TwoSlopeNorm
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    find_latitude_indices,
    find_polar_cap_indices,
    get_area_weights,
    get_channels,
    get_coords,
    get_forecast_steps,
    load_step,
    open_validation,
)
from weathergen.stratosphere.levels import build_level_map

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# Default SSW reference date (Feb 2018 event)
SSW_DATE = datetime(2018, 2, 12)

# Standard pressure ticks for log-pressure y-axis
_P_TICKS = [10, 30, 50, 100, 200, 300, 500, 700, 1000]
_P_TICK_LABELS = [str(p) for p in _P_TICKS]


# ---------------------------------------------------------------------------
# Axis helper
# ---------------------------------------------------------------------------


def _configure_pressure_axis(
    ax: Any,
    datetimes: list[datetime],
    ssw_date: datetime = SSW_DATE,
    tick_interval_steps: int = 28,
) -> None:
    """Apply log-pressure y-axis formatting and date x-ticks to *ax*."""
    ax.invert_yaxis()
    ax.set_yscale("log")
    ax.set_yticks(_P_TICKS)
    ax.set_yticklabels(_P_TICK_LABELS)
    ax.set_ylabel("Pressure (hPa)", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    n = len(datetimes)
    indices = np.arange(0, n, tick_interval_steps)
    ax.set_xticks(indices)
    ax.set_xticklabels([datetimes[i].strftime("%m-%d") for i in indices], rotation=45)
    ax.set_xlabel("Date (MM-DD)", fontsize=12)

    if datetimes[0] <= ssw_date <= datetimes[-1]:
        ssw_idx = int(
            np.argmin([abs((dt - ssw_date).total_seconds()) for dt in datetimes])
        )
        ax.axvline(
            ssw_idx,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="SSW onset",
        )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_vertical_profile(
    zarr_path: Path,
    label: str,
    target_latitude: float = 60.0,
    sample: int = 0,
) -> dict[str, Any] | None:
    """
    Extract zonal mean zonal wind at *target_latitude* across all ERA5ml
    model levels for a height–time cross-section.

    Returns a dict with keys ``label``, ``predictions``, ``targets``,
    ``datetimes``, ``pressures``, ``channels``; or ``None`` on error.
    """
    _logger.info(
        "%s: extracting wind vertical profile (lat=%.1f°N)…", label, target_latitude
    )

    with open_validation(zarr_path) as zio:
        if "ERA5ml" not in zio.streams:
            _logger.warning(
                "%s: ERA5ml not available (streams: %s) — skipping wind profile",
                label,
                list(zio.streams),
            )
            return None

        stream = "ERA5ml"
        channels = get_channels(zio, stream, sample)
        coords = get_coords(zio, stream, sample)

        level_map = build_level_map("u", channels)
        # Exclude levels with pressure ≤ 0 (top-of-atmosphere levels in L137)
        # to avoid log(0) on the pressure axis.
        level_map = {ch: p for ch, p in level_map.items() if p > 0.1}
        if not level_map:
            _logger.warning("%s: no u-wind channels found in ERA5ml — skipping", label)
            return None

        # Sort channels by ascending pressure (= descending altitude)
        sorted_channels = sorted(level_map, key=lambda ch: level_map[ch])
        ch_indices = [channels.index(ch) for ch in sorted_channels]
        pressures = np.array([level_map[ch] for ch in sorted_channels])

        lat_indices = find_latitude_indices(coords, target_latitude)
        steps = get_forecast_steps(zio)

        n_steps = len(steps)
        n_levels = len(sorted_channels)
        predictions = np.zeros((n_steps, n_levels), dtype=np.float32)
        targets = np.zeros((n_steps, n_levels), dtype=np.float32)
        raw_times: list[Any] = []

        for i, step in enumerate(steps):
            pred, tgt, times = load_step(zio, stream, step, sample)
            # pred/tgt: (n_pts, n_channels) or (n_pts, n_channels, n_ens)
            pred3 = np.atleast_3d(pred)  # (n_pts, n_ch, n_ens)
            tgt3 = np.atleast_3d(tgt)
            for j, ch_idx in enumerate(ch_indices):
                predictions[i, j] = pred3[lat_indices, ch_idx, :].mean()
                targets[i, j] = tgt3[lat_indices, ch_idx, :].mean()
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info(
        "%s: %d steps × %d levels  (%.0f–%.0f hPa)",
        label,
        n_steps,
        n_levels,
        pressures[0],
        pressures[-1],
    )

    return {
        "label": label,
        "latitude": target_latitude,
        "predictions": predictions,
        "targets": targets,
        "datetimes": datetimes,
        "pressures": pressures,
        "channels": sorted_channels,
    }


def extract_temperature_vertical_profile(
    zarr_path: Path,
    label: str,
    min_latitude: float = 60.0,
    sample: int = 0,
) -> dict[str, Any] | None:
    """
    Extract area-weighted polar-cap mean temperature across all ERA5ml model
    levels for a height–time cross-section.

    Returns a dict with keys ``label``, ``predictions``, ``targets``,
    ``datetimes``, ``pressures``, ``channels``; or ``None`` on error.
    """
    _logger.info(
        "%s: extracting temperature vertical profile (polar cap ≥%.1f°N)…",
        label,
        min_latitude,
    )

    with open_validation(zarr_path) as zio:
        if "ERA5ml" not in zio.streams:
            _logger.warning(
                "%s: ERA5ml not available (streams: %s) — skipping temperature profile",
                label,
                list(zio.streams),
            )
            return None

        stream = "ERA5ml"
        channels = get_channels(zio, stream, sample)
        coords = get_coords(zio, stream, sample)

        level_map = build_level_map("t", channels)
        # Exclude levels with pressure ≤ 0 (top-of-atmosphere levels in L137)
        # to avoid log(0) on the pressure axis.
        level_map = {ch: p for ch, p in level_map.items() if p > 0.1}
        if not level_map:
            _logger.warning(
                "%s: no temperature channels found in ERA5ml — skipping", label
            )
            return None

        sorted_channels = sorted(level_map, key=lambda ch: level_map[ch])
        ch_indices = [channels.index(ch) for ch in sorted_channels]
        pressures = np.array([level_map[ch] for ch in sorted_channels])

        polar_indices = find_polar_cap_indices(coords, min_latitude)
        weights = get_area_weights(coords, polar_indices)  # normalised
        steps = get_forecast_steps(zio)

        n_steps = len(steps)
        n_levels = len(sorted_channels)
        predictions = np.zeros((n_steps, n_levels), dtype=np.float32)
        targets = np.zeros((n_steps, n_levels), dtype=np.float32)
        raw_times: list[Any] = []

        for i, step in enumerate(steps):
            pred, tgt, times = load_step(zio, stream, step, sample)
            pred3 = np.atleast_3d(pred)  # (n_pts, n_ch, n_ens)
            tgt3 = np.atleast_3d(tgt)
            for j, ch_idx in enumerate(ch_indices):
                # Area-weighted polar-cap mean, then ensemble mean
                predictions[i, j] = (
                    (pred3[polar_indices, ch_idx, :] * weights[:, None])
                    .sum(axis=0)
                    .mean()
                )
                targets[i, j] = (
                    (tgt3[polar_indices, ch_idx, :] * weights[:, None])
                    .sum(axis=0)
                    .mean()
                )
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info(
        "%s: %d steps × %d levels  (%.0f–%.0f hPa)",
        label,
        n_steps,
        n_levels,
        pressures[0],
        pressures[-1],
    )

    return {
        "label": label,
        "min_latitude": min_latitude,
        "predictions": predictions,
        "targets": targets,
        "datetimes": datetimes,
        "pressures": pressures,
        "channels": sorted_channels,
    }


# ---------------------------------------------------------------------------
# Anomaly calculation
# ---------------------------------------------------------------------------


def calculate_anomalies(
    data: np.ndarray,
    reference_data: np.ndarray | None = None,
    reference_period: int = 40,
    data_pressures: np.ndarray | None = None,
    reference_pressures: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute anomalies relative to a reference mean.

    If *reference_data* is provided the mean over its first *reference_period*
    steps is used as the baseline.  When pressure-level arrays are given and
    don't fully overlap, common levels are matched; non-overlapping levels fall
    back to the per-experiment reference period mean.

    Args:
        data:               ``(time, level)`` array.
        reference_data:     Optional ``(time, level)`` baseline array.
        reference_period:   Number of initial timesteps used for the baseline
                            (default 40 = 10 days at 6-hourly).
        data_pressures:     Pressure levels for *data* (hPa).
        reference_pressures: Pressure levels for *reference_data* (hPa).

    Returns:
        Anomaly array with the same shape as *data*.
    """
    if reference_data is not None:
        if (
            data_pressures is not None
            and reference_pressures is not None
            and not np.array_equal(data_pressures, reference_pressures)
        ):
            common = np.intersect1d(data_pressures, reference_pressures)
            if len(common) == 0:
                reference_mean = data[:reference_period].mean(axis=0, keepdims=True)
            else:
                d_idx = np.array([np.where(data_pressures == p)[0][0] for p in common])
                r_idx = np.array(
                    [np.where(reference_pressures == p)[0][0] for p in common]
                )
                reference_mean = data[:reference_period].mean(axis=0, keepdims=True)
                reference_mean[:, d_idx] = reference_data[
                    :reference_period, r_idx
                ].mean(axis=0, keepdims=True)
        else:
            reference_mean = reference_data[:reference_period].mean(
                axis=0, keepdims=True
            )
    else:
        reference_mean = data[:reference_period].mean(axis=0, keepdims=True)

    return data - reference_mean


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _contourf_panel(
    ax: Any,
    T: np.ndarray,
    P: np.ndarray,
    field: np.ndarray,
    levels: np.ndarray,
    cmap: str,
    norm: Any,
    cb_label: str,
    title: str,
    datetimes: list[datetime],
    contour_levels: list[float] | None = None,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """Render one contourf panel with consistent formatting."""
    cf = ax.contourf(T, P, field.T, levels=levels, cmap=cmap, norm=norm, extend="both")
    if contour_levels is not None:
        cs = ax.contour(
            T,
            P,
            field.T,
            levels=contour_levels,
            colors="black",
            linewidths=1.5,
            alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=8)
    plt.colorbar(cf, ax=ax, label=cb_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    _configure_pressure_axis(ax, datetimes, ssw_date)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_vertical_structure(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    reference_dict: dict[str, Any] | None = None,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Save a 2×2 panel per experiment showing absolute and anomaly zonal wind.

    Panels: Prediction (abs) | Target (abs) | Prediction (anomaly) | Target (anomaly).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_data = reference_dict["data"] if reference_dict else None
    ref_pressures = reference_dict["pressures"] if reference_dict else None

    for data in data_list:
        label = data["label"]
        datetimes = data["datetimes"]
        pressures = data["pressures"]
        predictions = data["predictions"]
        targets = data["targets"]

        pred_anom = calculate_anomalies(
            predictions,
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )
        tgt_anom = calculate_anomalies(
            targets,
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )

        T, P = np.meshgrid(np.arange(len(datetimes)), pressures)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Absolute wind — shared diverging scale
        all_abs = np.concatenate([predictions.ravel(), targets.ravel()])
        vabs = max(float(np.percentile(np.abs(all_abs), 99)), 5.0)
        norm_abs = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
        levels_abs = np.linspace(-vabs, vabs, 37)

        for ax, field, ttl in (
            (axes[0, 0], predictions, f"{label} — Prediction"),
            (axes[0, 1], targets, f"{label} — Target (ERA5)"),
        ):
            _contourf_panel(
                ax,
                T,
                P,
                field,
                levels_abs,
                "RdBu_r",
                norm_abs,
                "Zonal Wind (m/s)",
                f"{ttl}\nZonal Wind at {data['latitude']:.0f}°N",
                datetimes,
                contour_levels=[0.0],
                ssw_date=ssw_date,
            )

        # Anomaly — fixed ±30 m/s scale
        norm_anom = TwoSlopeNorm(vmin=-30, vcenter=0.0, vmax=30)
        levels_anom = np.linspace(-30, 30, 31)

        for ax, field, ttl in (
            (axes[1, 0], pred_anom, f"{label} — Prediction Anomaly"),
            (axes[1, 1], tgt_anom, f"{label} — Target Anomaly"),
        ):
            _contourf_panel(
                ax,
                T,
                P,
                field,
                levels_anom,
                "RdBu_r",
                norm_anom,
                "Wind Anomaly (m/s)",
                ttl,
                datetimes,
                contour_levels=[0.0],
                ssw_date=ssw_date,
            )

        plt.tight_layout()
        out = output_dir / f"vertical_structure_{label}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        _logger.info("Saved: %s", out)
        plt.close()


def plot_downward_propagation(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    reference_dict: dict[str, Any] | None = None,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Save a multi-experiment comparison of zonal wind anomaly cross-sections.

    Layout: one row per experiment, two columns (Prediction | Target).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_data = reference_dict["data"] if reference_dict else None
    ref_pressures = reference_dict["pressures"] if reference_dict else None

    n_exps = len(data_list)
    fig, axes = plt.subplots(n_exps, 2, figsize=(16, 6 * n_exps), squeeze=False)

    norm_anom = TwoSlopeNorm(vmin=-30, vcenter=0.0, vmax=30)
    levels_anom = np.linspace(-30, 30, 31)

    for idx, data in enumerate(data_list):
        label = data["label"]
        datetimes = data["datetimes"]
        pressures = data["pressures"]

        pred_anom = calculate_anomalies(
            data["predictions"],
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )
        tgt_anom = calculate_anomalies(
            data["targets"],
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )

        T, P = np.meshgrid(np.arange(len(datetimes)), pressures)

        for ax, field, ttl in (
            (axes[idx, 0], pred_anom, f"{label} — Prediction"),
            (axes[idx, 1], tgt_anom, f"{label} — Target (ERA5)"),
        ):
            _contourf_panel(
                ax,
                T,
                P,
                field,
                levels_anom,
                "RdBu_r",
                norm_anom,
                "Wind Anomaly (m/s)",
                f"{ttl}\nDownward Propagation",
                datetimes,
                contour_levels=[-10.0, -5.0, 5.0, 10.0],
                ssw_date=ssw_date,
            )

    fig.suptitle(
        "Downward Propagation of SSW Wind Anomaly", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    out = output_dir / "downward_propagation_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    _logger.info("Saved: %s", out)
    plt.close()


def plot_temperature_vertical_structure(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    reference_dict: dict[str, Any] | None = None,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Save a 2×2 panel per experiment showing absolute and anomaly polar-cap temperature.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_data = reference_dict["data"] if reference_dict else None
    ref_pressures = reference_dict["pressures"] if reference_dict else None

    for data in data_list:
        label = data["label"]
        datetimes = data["datetimes"]
        pressures = data["pressures"]
        predictions = data["predictions"]
        targets = data["targets"]

        pred_anom = calculate_anomalies(
            predictions,
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )
        tgt_anom = calculate_anomalies(
            targets,
            ref_data,
            data_pressures=pressures,
            reference_pressures=ref_pressures,
        )

        T, P = np.meshgrid(np.arange(len(datetimes)), pressures)

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Absolute temperature — 180–270 K
        levels_abs = np.linspace(180, 270, 37)

        for ax, field, ttl in (
            (axes[0, 0], predictions, f"{label} — Prediction"),
            (axes[0, 1], targets, f"{label} — Target (ERA5)"),
        ):
            _contourf_panel(
                ax,
                T,
                P,
                field,
                levels_abs,
                "RdYlBu_r",
                None,
                "Temperature (K)",
                f"{ttl}\nPolar Cap Temperature (≥{data['min_latitude']:.0f}°N)",
                datetimes,
                ssw_date=ssw_date,
            )

        # Anomaly — fixed ±20 K scale
        norm_anom = TwoSlopeNorm(vmin=-20, vcenter=0.0, vmax=20)
        levels_anom = np.linspace(-20, 20, 41)

        for ax, field, ttl in (
            (axes[1, 0], pred_anom, f"{label} — Prediction Anomaly"),
            (axes[1, 1], tgt_anom, f"{label} — Target Anomaly"),
        ):
            _contourf_panel(
                ax,
                T,
                P,
                field,
                levels_anom,
                "RdBu_r",
                norm_anom,
                "Temperature Anomaly (K)",
                ttl,
                datetimes,
                contour_levels=[0.0],
                ssw_date=ssw_date,
            )

        plt.tight_layout()
        out = output_dir / f"temperature_vertical_structure_{label}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        _logger.info("Saved: %s", out)
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Height–time cross-sections of zonal wind and polar-cap temperature.",
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
        default=Path("plots/vertical_structure"),
        help="Output directory for plots.",
    )

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument(
        "--run-ids", nargs="+", help="Validation IDs used as zarr path components."
    )
    id_group.add_argument(
        "--validations-config", type=Path, help="YAML validations config file."
    )

    parser.add_argument(
        "--sample", type=int, default=0, help="Sample index (default: 0)."
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=60.0,
        help="Target latitude for zonal wind cross-section (default: 60°N).",
    )
    parser.add_argument(
        "--min-latitude",
        type=float,
        default=60.0,
        help="Minimum latitude for polar-cap temperature average (default: 60°N).",
    )
    parser.add_argument(
        "--reference-id",
        type=str,
        default=None,
        help="Validation ID to use as anomaly reference baseline (optional; "
        "defaults to the per-experiment initial-period mean).",
    )
    parser.add_argument(
        "--skip-wind",
        action="store_true",
        help="Skip zonal wind profiles.",
    )
    parser.add_argument(
        "--skip-temperature",
        action="store_true",
        help="Skip polar-cap temperature profiles.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Resolve run specs
    if args.validations_config:
        cfg = load_validations_config(args.validations_config)
        run_specs = [
            {"id": v["id"], "label": label, "sample": v.get("sample", args.sample)}
            for label, v in cfg.items()
        ]
    else:
        run_specs = [
            {"id": rid, "label": rid, "sample": args.sample} for rid in args.run_ids
        ]

    # Optional reference baseline
    wind_reference: dict[str, Any] | None = None
    temp_reference: dict[str, Any] | None = None

    if args.reference_id:
        ref_zarr = args.data_dir / args.reference_id / _ZARR_FNAME
        if ref_zarr.exists():
            _logger.info("Extracting anomaly reference from %s…", args.reference_id)
            if not args.skip_wind:
                ref_w = extract_vertical_profile(
                    ref_zarr, args.reference_id, args.latitude, args.sample
                )
                if ref_w is not None:
                    wind_reference = {
                        "data": ref_w["predictions"],
                        "pressures": ref_w["pressures"],
                    }
            if not args.skip_temperature:
                ref_t = extract_temperature_vertical_profile(
                    ref_zarr, args.reference_id, args.min_latitude, args.sample
                )
                if ref_t is not None:
                    temp_reference = {
                        "data": ref_t["predictions"],
                        "pressures": ref_t["pressures"],
                    }
        else:
            _logger.warning(
                "Reference zarr not found: %s — using per-experiment baseline", ref_zarr
            )

    # Extract profiles for all experiments
    wind_data: list[dict[str, Any]] = []
    temp_data: list[dict[str, Any]] = []

    for spec in run_specs:
        zarr_path = args.data_dir / spec["id"] / _ZARR_FNAME
        if not zarr_path.exists():
            _logger.warning("zarr not found, skipping: %s", zarr_path)
            continue
        if not args.skip_wind:
            d = extract_vertical_profile(
                zarr_path, spec["label"], args.latitude, spec["sample"]
            )
            if d is not None:
                wind_data.append(d)
        if not args.skip_temperature:
            d = extract_temperature_vertical_profile(
                zarr_path, spec["label"], args.min_latitude, spec["sample"]
            )
            if d is not None:
                temp_data.append(d)

    # Generate plots
    if wind_data:
        plot_vertical_structure(wind_data, args.output_dir, wind_reference)
        if len(wind_data) > 1:
            plot_downward_propagation(wind_data, args.output_dir, wind_reference)
    elif not args.skip_wind:
        _logger.warning("No wind data extracted.")

    if temp_data:
        plot_temperature_vertical_structure(temp_data, args.output_dir, temp_reference)
    elif not args.skip_temperature:
        _logger.warning("No temperature data extracted.")

    _logger.info("Done.")


if __name__ == "__main__":
    main()
