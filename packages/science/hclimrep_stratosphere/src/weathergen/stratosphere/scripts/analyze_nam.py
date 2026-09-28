#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Coarse 3-level Northern Annular Mode (NAM) index.

Computes a leading-EOF NAM/AO-style index at each of the three ERA5pl
pressure levels available in the ``era5_strato_final`` stream
(50 / 500 / 850 hPa), using ``z_50``, ``z_500``, ``z_850`` geopotential.

This is deliberately a coarse proxy, NOT a full stratospheric NAM: with only
3 pressure levels there is no continuous height-time cross-section, just
three independent per-level time series (one panel each).

Anomalies are computed either against an external climatology zarr store
(``--climatology``, matched by day-of-year + hour, as in
``analyze_polar_vortex.py``) or, if omitted, against the run's own
time-mean over the plotted window — a much coarser approximation that
should not be over-interpreted for short windows.

Usage::

    ssw-analyze nam \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/nam

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_nam \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results
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
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.diagnostics import (
    nam_index_eof_reference,
    nam_index_eof_project,
)
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    find_polar_cap_indices,
    get_channels,
    get_coords,
    get_forecast_steps,
    load_step,
    open_validation,
)

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# The only 3 pressure levels available in the era5_strato_final ERA5pl stream.
_NAM_LEVELS = [50, 500, 850]

# Default EOF domain: poleward of this latitude.
_DEFAULT_MIN_LAT = 20.0

SSW_DATE = datetime(2018, 2, 12)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------


def load_climatology_domain_field(
    climatology_path: Path,
    channel: str,
    datetimes: list[datetime],
    domain_indices: np.ndarray,
) -> np.ndarray | None:
    """
    Load per-point climatology values for *channel* at *domain_indices*,
    matched to *datetimes* by day-of-year + hour (as in
    ``analyze_polar_vortex.load_climatology_zonal_mean``).

    Returns ``(n_time, n_domain_pts)`` or ``None`` if *channel* is absent.
    """
    import xarray as xr

    clim = xr.open_zarr(climatology_path)
    clim_channels = list(clim.channels.values)
    if channel not in clim_channels:
        _logger.warning("Climatology: channel %s not found", channel)
        return None
    ch_idx = clim_channels.index(channel)

    clim_times = pd.to_datetime(clim.time.values)
    clim_doys = clim_times.dayofyear.values
    clim_hours = clim_times.hour.values

    time_indices: list[int] = []
    for dt in datetimes:
        ts = pd.Timestamp(dt)
        mask = (clim_doys == ts.dayofyear) & (clim_hours == ts.hour)
        idx = np.where(mask)[0]
        time_indices.append(int(idx[0]) if len(idx) > 0 else -1)

    unique_t = sorted({t for t in time_indices if t >= 0})
    if not unique_t:
        return None

    # clim.data shape: (time, channels, grid_points)
    block = clim.data.isel(time=unique_t, channels=[ch_idx]).values[:, 0, domain_indices]
    t_pos = {t: i for i, t in enumerate(unique_t)}

    out = np.full((len(datetimes), len(domain_indices)), np.nan, dtype=np.float64)
    for ti, t in enumerate(time_indices):
        if t >= 0:
            out[ti] = block[t_pos[t]]
    return out


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_nam(
    zarr_path: Path,
    label: str,
    sample: int = 0,
    min_lat: float = _DEFAULT_MIN_LAT,
    climatology_path: Path | None = None,
    lead_days: int = 0,
    color: str = "C0",
    ssw_date: datetime = SSW_DATE,
) -> dict[str, Any] | None:
    """
    Extract the coarse 3-level NAM index for one validation run.

    Returns a dict with keys ``label``, ``lead_days``, ``color``,
    ``datetimes``, ``days_rel``, and ``pred``/``tgt`` dicts mapping each
    available level (int) to its ``(n_time,)`` NAM index; or ``None`` on
    failure.
    """
    with open_validation(zarr_path) as zio:
        if "ERA5pl" not in zio.streams:
            _logger.warning("%s: ERA5pl stream not present — skipping NAM", label)
            return None

        channels = get_channels(zio, "ERA5pl", sample)
        coords = get_coords(zio, "ERA5pl", sample)
        domain_idx = find_polar_cap_indices(coords, min_lat, "north")
        domain_coords = coords[domain_idx]

        levels = [lv for lv in _NAM_LEVELS if f"z_{lv}" in channels]
        if not levels:
            _logger.warning(
                "%s: none of z_%s found in ERA5pl (available z channels: %s)",
                label,
                _NAM_LEVELS,
                [c for c in channels if c.startswith("z_")],
            )
            return None

        ch_idx = {lv: channels.index(f"z_{lv}") for lv in levels}
        steps = get_forecast_steps(zio, skip_source_step=True)

        pred_fields: dict[int, list[np.ndarray]] = {lv: [] for lv in levels}
        tgt_fields: dict[int, list[np.ndarray]] = {lv: [] for lv in levels}
        raw_times: list = []

        for step in steps:
            pred, tgt, times = load_step(zio, "ERA5pl", step, sample)
            pred = np.atleast_3d(pred)
            tgt = np.atleast_3d(tgt)
            for lv in levels:
                pred_fields[lv].append(pred[domain_idx, ch_idx[lv], 0].astype(np.float64))
                tgt_fields[lv].append(tgt[domain_idx, ch_idx[lv], 0].astype(np.float64))
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))

    pred_nam: dict[int, np.ndarray] = {}
    tgt_nam: dict[int, np.ndarray] = {}
    for lv in levels:
        pred_arr = np.stack(pred_fields[lv])  # (n_time, n_domain)
        tgt_arr = np.stack(tgt_fields[lv])

        clim = None
        if climatology_path is not None:
            clim = load_climatology_domain_field(
                climatology_path, f"z_{lv}", datetimes, domain_idx
            )
        if clim is not None:
            pred_anom = pred_arr - clim
            tgt_anom = tgt_arr - clim
        else:
            # Coarse fallback: subtract the run's own time-mean.
            pred_anom = pred_arr - pred_arr.mean(axis=0, keepdims=True)
            tgt_anom = tgt_arr - tgt_arr.mean(axis=0, keepdims=True)

        # Compute reference EOF pattern from ERA5 (target), then project both
        eof1_ref, _sv = nam_index_eof_reference(
            tgt_anom.astype(np.float64),
            domain_coords,
            full_coords=coords,
            domain_indices=domain_idx,
        )
        pred_nam[lv] = nam_index_eof_project(
            pred_anom.astype(np.float64),
            eof1_ref,
            domain_coords,
            full_coords=coords,
            domain_indices=domain_idx,
        )
        tgt_nam[lv] = nam_index_eof_project(
            tgt_anom.astype(np.float64),
            eof1_ref,
            domain_coords,
            full_coords=coords,
            domain_indices=domain_idx,
        )

    days_rel = np.array(
        [(dt - ssw_date).total_seconds() / 86400.0 for dt in datetimes],
        dtype=np.float64,
    )

    _logger.info("%s: computed NAM index for levels %s", label, levels)

    return {
        "label": label,
        "lead_days": lead_days,
        "color": color,
        "datetimes": datetimes,
        "days_rel": days_rel,
        "levels": levels,
        "pred": pred_nam,
        "tgt": tgt_nam,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_nam(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    One panel per available pressure level; each panel shows the NAM index
    time series (predictions colored by lead time, ERA5 target in black),
    x-axis = days relative to *ssw_date*.
    """
    if not data_list:
        _logger.warning("No NAM data to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_levels = sorted({lv for data in data_list for lv in data["levels"]})
    if not all_levels:
        return

    fig, axes = plt.subplots(len(all_levels), 1, figsize=(13, 4 * len(all_levels)))
    axes = np.atleast_1d(axes)
    data_sorted = sorted(data_list, key=lambda d: d["days_rel"][0])

    for ax, lv in zip(axes, all_levels):
        era5_drawn = False
        for data in data_sorted:
            if lv not in data["pred"]:
                continue
            days_rel = data["days_rel"]
            ax.plot(
                days_rel,
                data["pred"][lv],
                color=data["color"],
                linestyle="-",
                linewidth=1.5,
                alpha=0.85,
                label=f"t{data['lead_days']:02d}d pred",
            )
            if not era5_drawn:
                ax.plot(
                    days_rel,
                    data["tgt"][lv],
                    color="k",
                    linestyle="-",
                    linewidth=2.5,
                    alpha=0.9,
                    label="ERA5",
                    zorder=10,
                )
                era5_drawn = True

        if days_rel[0] <= 0.0 <= days_rel[-1]:
            ax.axvline(
                0.0,
                color="red",
                linestyle="-.",
                linewidth=1.8,
                alpha=0.7,
                label="SSW onset (day 0)",
            )
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(f"NAM index (z_{lv})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=4)

    axes[-1].set_xlabel("Days relative to SSW onset", fontsize=12)
    fig.suptitle(
        "Coarse 3-level NAM index (50/500/850 hPa proxy, not a full "
        "stratospheric NAM)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / "nam_index_by_level.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
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
        description=(
            "Coarse 3-level (50/500/850 hPa) NAM index — NOT a full "
            "stratospheric NAM, limited by ERA5pl's available levels."
        )
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
        default=Path("plots/nam"),
        help="Directory for output plots.  [default: plots/nam]",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        default=_DEFAULT_MIN_LAT,
        help="Southern boundary of the EOF domain (°N).  [default: 20]",
    )
    parser.add_argument(
        "--climatology",
        type=Path,
        default=None,
        help=(
            "Optional climatology zarr store for anomaly computation "
            "(matched by day-of-year + hour). If omitted, anomalies are "
            "computed against each run's own time-mean (coarser)."
        ),
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference lines (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("COARSE 3-LEVEL NAM INDEX")
    print("=" * 60)

    cfg = load_validations_config(args.validations_config)
    run_specs = [
        (
            label,
            spec["id"],
            spec.get("sample", 0),
            spec.get("lead_days") or 0,
            spec.get("color") or _COLORS[i % len(_COLORS)],
        )
        for i, (label, spec) in enumerate(cfg.items())
    ]

    data_list: list[dict[str, Any]] = []
    for label, run_id, sample, lead_days, color in run_specs:
        zarr_path = args.data_dir / run_id / _ZARR_FNAME
        if not zarr_path.exists():
            _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
            continue

        data = extract_nam(
            zarr_path,
            label,
            sample,
            min_lat=args.min_lat,
            climatology_path=args.climatology,
            lead_days=lead_days,
            color=color,
            ssw_date=args.ssw_date,
        )
        if data is None:
            _logger.warning("Skipping %s — extraction failed.", label)
            continue
        data_list.append(data)

    plot_nam(data_list, args.output_dir, args.ssw_date)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
