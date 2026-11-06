#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Multi-level Northern Annular Mode (NAM) index.

Computes a leading-EOF NAM/AO-style index of geopotential anomalies poleward
of ``--min-lat`` at each requested pressure level (default 10, 50, 100, 500
and 1000 hPa), one independent time series per level (one panel each).

Multi-stream layout: a forecast store may hold several streams, e.g.
``ERA5pl`` (z at 50–1000 hPa) and ``S2Spredictors`` (z at 1–70 hPa). Levels
are addressed in hPa and located with
:func:`weathergen.stratosphere.io.discover_column`, which merges all streams
into one pressure axis, so 10 hPa (S2Spredictors) and 500 hPa (ERA5pl) are
handled uniformly. Levels that are not available in a store (e.g. 10 hPa in
the ``memory`` arm) are skipped with a warning.

Anomalies are computed either against an external climatology zarr store
(``--climatology``, matched by day-of-year + hour, as in
``analyze_polar_vortex.py``) or, if omitted, against the run's own
time-mean over the plotted window — a much coarser approximation that
should not be over-interpreted for short windows.

Usage::

    ssw-analyze nam \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/nam \\
        --levels 10 50 100 500 1000

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
from numpy.typing import NDArray

from weathergen.stratosphere.config import (
    event_reference_date,
    group_validations_by_event,
    load_validations_config,
)
from weathergen.stratosphere.diagnostics import (
    nam_index_eof_project,
    nam_index_eof_reference,
)
from weathergen.stratosphere.io import (
    LevelRef,
    convert_times_to_datetime,
    find_polar_cap_indices,
    get_coords,
    get_forecast_steps,
    load_level,
    open_validation,
    resolve_channel_arg,
    zarr_store_path,
)
from weathergen.stratosphere.levels import format_pressure

_logger = logging.getLogger(__name__)

# Default pressure levels (hPa) for the per-level NAM index; missing levels
# are skipped.
_DEFAULT_NAM_LEVELS = [10.0, 50.0, 100.0, 500.0, 1000.0]

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
    domain_indices: NDArray,
) -> NDArray | None:
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
    levels_hpa: list[float] | None = None,
    channels: list[str] | None = None,
    lead_days: int = 0,
    color: str = "C0",
    ssw_date: datetime = SSW_DATE,
) -> dict[str, Any] | None:
    """
    Extract the per-level NAM index for one validation run.

    *levels_hpa* (hPa) or legacy *channels* (channel names) select the
    geopotential levels; defaults to :data:`_DEFAULT_NAM_LEVELS`, skipping
    levels that are not present in any stream.

    Returns a dict with keys ``label``, ``lead_days``, ``color``,
    ``datetimes``, ``days_rel``, ``levels`` (sorted pressures, hPa) and
    ``pred``/``tgt`` dicts mapping each available pressure (float, hPa) to
    its ``(n_time,)`` NAM index; or ``None`` on failure.
    """
    with open_validation(zarr_path) as zio:
        try:
            refs: list[LevelRef] = resolve_channel_arg(
                zio, "z", channels, levels_hpa, _DEFAULT_NAM_LEVELS, sample
            )
        except KeyError as e:
            _logger.warning("%s: %s — skipping NAM", label, e)
            return None
        if not refs:
            _logger.warning("%s: no geopotential levels found in any stream — skipping", label)
            return None
        refs = sorted(refs, key=lambda r: r.pressure)
        levels = [r.pressure for r in refs]

        # Domain (polar cap poleward of min_lat) per stream; grids may differ.
        domain: dict[str, tuple[NDArray, NDArray, NDArray]] = {}
        for ref in refs:
            if ref.stream not in domain:
                # float64: in float32 cos(90°) is slightly negative, which
                # would give NaN sqrt-weights in the EOF for a polar point.
                coords = get_coords(zio, ref.stream, sample).astype(np.float64)
                idx = find_polar_cap_indices(coords, min_lat, "north")
                domain[ref.stream] = (coords, idx, coords[idx])

        steps = get_forecast_steps(zio, skip_source_step=True)

        pred_fields: dict[float, list[NDArray]] = {lv: [] for lv in levels}
        tgt_fields: dict[float, list[NDArray]] = {lv: [] for lv in levels}
        raw_times: list = []

        for step in steps:
            times = None
            for ref in refs:
                pred, tgt, times = load_level(zio, ref, step, sample)
                _, idx, _ = domain[ref.stream]
                pred_fields[ref.pressure].append(pred[idx, 0].astype(np.float64))
                tgt_fields[ref.pressure].append(tgt[idx, 0].astype(np.float64))
            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))

    pred_nam: dict[float, NDArray] = {}
    tgt_nam: dict[float, NDArray] = {}
    for ref in refs:
        lv = ref.pressure
        coords, domain_idx, domain_coords = domain[ref.stream]
        pred_arr = np.stack(pred_fields[lv])  # (n_time, n_domain)
        tgt_arr = np.stack(tgt_fields[lv])

        clim = None
        if climatology_path is not None:
            clim = load_climatology_domain_field(
                climatology_path, ref.channel, datetimes, domain_idx
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

    _logger.info("%s: computed NAM index at %s", label, [format_pressure(lv) for lv in levels])

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
    One panel per available pressure level (hPa); each panel shows the NAM index
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

    for ax, lv in zip(axes, all_levels, strict=True):
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
        ax.set_ylabel(f"NAM index (z {format_pressure(lv)})", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=4)

    axes[-1].set_xlabel("Days relative to SSW onset", fontsize=12)
    fig.suptitle(
        "NAM index by pressure level (leading EOF of z anomalies, "
        + "/".join(f"{lv:g}" for lv in all_levels)
        + " hPa)",
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
            "Per-level NAM index (leading EOF of geopotential anomalies "
            "poleward of --min-lat) at the requested pressure levels."
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
        "--levels",
        type=float,
        nargs="+",
        metavar="HPA",
        default=None,
        help=(
            "Pressure levels (hPa) of geopotential for the NAM index; levels "
            "missing from the store are skipped.  "
            f"[default: {' '.join(f'{lv:g}' for lv in _DEFAULT_NAM_LEVELS)}]"
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Legacy alias for --levels: explicit z channel names (e.g. z_500).",
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

    _logger.info("=== NAM INDEX BY PRESSURE LEVEL ===")

    cfg = load_validations_config(args.validations_config)
    events = group_validations_by_event(cfg)
    single_event = len(events) == 1

    for event_name, event_cfg in events.items():
        event_ssw_date = event_reference_date(event_cfg, args.ssw_date)
        event_output_dir = args.output_dir if single_event else args.output_dir / event_name

        run_specs = [
            (
                label,
                spec["id"],
                spec.get("sample", 0),
                spec.get("rank", 0),
                spec.get("lead_days") or 0,
                spec.get("color") or _COLORS[i % len(_COLORS)],
            )
            for i, (label, spec) in enumerate(event_cfg.items())
        ]

        data_list: list[dict[str, Any]] = []
        for label, run_id, sample, rank, lead_days, color in run_specs:
            zarr_path = zarr_store_path(args.data_dir, run_id, rank)
            if not zarr_path.exists():
                _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
                continue

            data = extract_nam(
                zarr_path,
                label,
                sample,
                min_lat=args.min_lat,
                climatology_path=args.climatology,
                levels_hpa=args.levels,
                channels=args.channels,
                lead_days=lead_days,
                color=color,
                ssw_date=event_ssw_date,
            )
            if data is None:
                _logger.warning("Skipping %s — extraction failed.", label)
                continue
            data_list.append(data)

        if not data_list:
            _logger.warning("[%s] No data extracted — skipping event.", event_name)
            continue

        plot_nam(data_list, event_output_dir, event_ssw_date)

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
