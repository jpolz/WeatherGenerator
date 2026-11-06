#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Polar Vortex Analysis for SSW Event Prediction.

Analyses zonal mean zonal wind at 60°N to detect Sudden Stratospheric Warming
events and compares model predictions against ERA5 targets.

Levels are addressed by pressure (hPa).  A forecast store of the strato_s2s
arms spreads the atmospheric column over several streams (``ERA5pl`` for
50–1000 hPa, ``S2Spredictors`` for 1–70 hPa, ``TEM`` for model-level TEM
diagnostics); the level-aware I/O layer merges them into one pressure axis
per variable, so 10 hPa and 50 hPa are found regardless of which stream holds
them.  Levels missing from a store (e.g. nothing above 50 hPa in the
``memory`` arm) are skipped with a warning.  SSW wind-reversal detection is
performed on the 10 hPa level only.

Usage::

    ssw-analyze polar-vortex \\
        --validations-config eval_config/validations.yml \\
        --data-dir /path/to/validation/data \\
        --output-dir plots/polar_vortex \\
        --levels 10 50

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_polar_vortex \\
        --run-ids qq8xsoeh j7ns9146 \\
        --data-dir ./data

``--channels u_29 u_55`` (legacy model-level channel names) is still accepted
and resolved against every stream in the store.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from weathergen.stratosphere.diagnostics import detect_ssw_reversal, zonal_mean
from weathergen.stratosphere.io import (
    LevelRef,
    convert_times_to_datetime,
    get_coords,
    get_forecast_steps,
    load_level,
    open_validation,
    resolve_channel_arg,
    zarr_store_path,
)
from weathergen.stratosphere.levels import (
    build_level_map,
    format_pressure,
    stream_level_type,
)

_logger = logging.getLogger(__name__)

# Default pressure levels (hPa) for the zonal-mean u-wind at 60°N
_DEFAULT_U_LEVELS_HPA = [10.0, 50.0]

# Pressure level (hPa) on which SSW wind reversals are diagnosed (WMO definition)
SSW_LEVEL_HPA = 10.0

# Latitude tolerance (degrees) for the zonal mean band around --latitude
_LAT_TOLERANCE = 2.5

# Known SSW event date for reference lines
SSW_DATE = datetime(2018, 2, 12)


def level_tag(pressure_hpa: float, variable: str = "u") -> str:
    """Filename-safe tag for a level, e.g. ``'u10hPa'``."""
    return f"{variable}{pressure_hpa:g}hPa"


def is_ssw_level(pressure_hpa: float, tolerance_frac: float = 0.25) -> bool:
    """``True`` if *pressure_hpa* is the (nearest-available) 10 hPa level."""
    return abs(pressure_hpa - SSW_LEVEL_HPA) / SSW_LEVEL_HPA <= tolerance_frac


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------


def _match_climatology_channels(
    refs: list[LevelRef], clim_channels: list[str], tolerance_frac: float = 0.25
) -> dict[float, str | None]:
    """
    Map each level's pressure → channel name in the climatology store.

    The climatology may be written on model levels (``u_29``) or pressure
    levels (``u_10``); match by pressure first, then by exact channel name.
    """
    out: dict[float, str | None] = {}
    lt = stream_level_type("climatology", clim_channels)
    for ref in refs:
        clim_map = build_level_map(ref.variable, clim_channels, lt)  # {channel: hPa}
        best: str | None = None
        if clim_map:
            ch, p = min(clim_map.items(), key=lambda kv: abs(np.log(kv[1]) - np.log(ref.pressure)))
            if abs(p - ref.pressure) / ref.pressure <= tolerance_frac:
                best = ch
        if best is None and ref.channel in clim_channels:
            best = ref.channel
        out[ref.pressure] = best
    return out


def load_climatology_zonal_mean(
    climatology_path: Path,
    refs: list[LevelRef],
    datetimes: list[datetime],
    target_latitude: float = 60.0,
) -> dict[float, NDArray]:
    """
    Load zonal mean climatology at *target_latitude* for each datetime.

    Matches by day-of-year + hour so the result is independent of forecast year.
    Returns a dict mapping pressure (hPa) -> float array (one value per datetime).
    """
    import xarray as xr

    from weathergen.stratosphere.io import find_latitude_indices

    _logger.info("Loading climatology from %s …", climatology_path)
    clim = xr.open_zarr(climatology_path)

    # Grid-point indices near target latitude
    clim_coords = np.column_stack([clim.latitude.values, clim.longitude.values])
    lat_indices = find_latitude_indices(clim_coords, target_latitude, _LAT_TOLERANCE)

    clim_channels = list(clim.channels.values)
    p_to_ch = _match_climatology_channels(refs, clim_channels)
    available = {p: ch for p, ch in p_to_ch.items() if ch is not None}
    missing = [p for p, ch in p_to_ch.items() if ch is None]
    if missing:
        _logger.warning("Levels not in climatology: %s", [format_pressure(p) for p in missing])

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

    result: dict[float, NDArray] = {
        p: np.full(len(datetimes), np.nan, dtype=np.float64) for p in p_to_ch
    }
    unique_t = sorted({t for t in time_indices if t >= 0})
    if not available or not unique_t:
        return result

    # Bulk-load: unique time steps × needed channels × lat-band points
    pressures = list(available)
    ch_indices = [clim_channels.index(available[p]) for p in pressures]

    # clim.data shape: (time, channels, grid_points)
    clim_block = clim.data.isel(time=unique_t, channels=ch_indices).values[
        :, :, lat_indices
    ]  # (n_t, n_ch, n_latpts)
    zonal_means = clim_block.mean(axis=2)  # (n_t, n_ch)
    t_pos = {t: i for i, t in enumerate(unique_t)}

    for ci, p in enumerate(pressures):
        for ti, t in enumerate(time_indices):
            if t >= 0:
                result[p][ti] = zonal_means[t_pos[t], ci]
    return result


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_zonal_wind_from_store(
    zio: Any,
    label: str,
    refs: list[LevelRef],
    sample: int = 0,
    target_latitude: float = 60.0,
    climatology_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Extract zonal mean u-wind at *target_latitude* for the given levels from an
    already-open validation store.

    See :func:`extract_zonal_wind` for the returned structure.
    """
    if not refs:
        _logger.warning("%s: no u-wind levels to extract.", label)
        return None

    steps = get_forecast_steps(zio)
    coords_by_stream = {s: get_coords(zio, s, sample) for s in {r.stream for r in refs}}

    preds_by_p: dict[float, list[NDArray]] = {r.pressure: [] for r in refs}
    targets_by_p: dict[float, list[float]] = {r.pressure: [] for r in refs}
    times_list: list[Any] = []

    for step in steps:
        step_time = None
        for ref in refs:
            pred, tgt, times = load_level(zio, ref, step, sample)  # (n_pts, n_ens)
            coords = coords_by_stream[ref.stream]
            # Zonal mean over the ±tolerance band, keep all ensemble members → (n_ens,)
            preds_by_p[ref.pressure].append(
                zonal_mean(pred.T, coords, target_latitude, _LAT_TOLERANCE)
            )
            # Target: single value (first member — targets carry no real ensemble)
            targets_by_p[ref.pressure].append(
                float(zonal_mean(tgt[:, 0], coords, target_latitude, _LAT_TOLERANCE))
            )
            if step_time is None:
                step_time = times[0]
        times_list.append(step_time)

    times_arr = np.array(times_list)
    datetimes = convert_times_to_datetime(times_arr)

    _logger.info(
        "  %s: %d forecast steps, %s → %s, levels %s",
        label,
        len(steps),
        datetimes[0],
        datetimes[-1],
        [f"{r.pressure:g} hPa ({r.stream}/{r.channel})" for r in refs],
    )

    # preds_np[p]: (n_steps, n_ens) — one row per forecast step, one col per member
    preds_np = {p: np.stack(v, axis=0) for p, v in preds_by_p.items()}
    targets_np = {p: np.array(v) for p, v in targets_by_p.items()}

    # Always store absolute values so that detect_ssw_reversal (u < 0 criterion)
    # works correctly.  Anomalies are kept separately for optional plotting.
    clim_means: dict[float, NDArray] = {}
    if climatology_path is not None:
        clim_means = load_climatology_zonal_mean(climatology_path, refs, datetimes, target_latitude)

    return {
        "label": label,
        "sample": sample,
        "color": None,  # filled by caller
        "is_anomaly": climatology_path is not None,
        "levels": {
            ref.pressure: {
                "ref": ref,
                "predictions": preds_np[ref.pressure],
                "targets": targets_np[ref.pressure],
                # anomaly arrays present only when climatology was supplied
                **(
                    {
                        # clim_means[p] is (n_steps,); broadcast over ens dim
                        "pred_anomaly": preds_np[ref.pressure] - clim_means[ref.pressure][:, None],
                        "tgt_anomaly": targets_np[ref.pressure] - clim_means[ref.pressure],
                    }
                    if ref.pressure in clim_means
                    else {}
                ),
            }
            for ref in refs
        },
        "times": times_arr,
        "datetimes": datetimes,
        "latitude": target_latitude,
    }


def extract_zonal_wind(
    zarr_path: Path,
    label: str,
    channels_override: list[str] | None = None,
    levels_hpa: list[float] | None = None,
    sample: int = 0,
    target_latitude: float = 60.0,
    climatology_path: Path | None = None,
) -> dict[str, Any] | None:
    """
    Extract zonal mean u-wind at *target_latitude* from a validation zarr store.

    Args:
        zarr_path:         Path to the ``.zarr`` store.
        label:             Human-readable label for logging/plotting.
        channels_override: Legacy explicit channel names (``u_29`` ...), looked
                           up in every stream.  Takes precedence over *levels_hpa*.
        levels_hpa:        Pressure levels in hPa; defaults to 10 and 50 hPa.
                           Levels missing from the store are skipped with a warning.
        sample:            Ensemble member index (default 0).
        target_latitude:   Latitude for zonal mean (default 60°N).
        climatology_path:  If provided, subtract the DOY climatology mean to
                           return anomalies instead of absolute values.

    Returns:
        Dict with keys ``label``, ``levels`` (dict keyed by pressure in hPa,
        each with ``ref`` / ``predictions`` / ``targets``), ``times``,
        ``datetimes``, ``latitude``, ``is_anomaly``; or ``None`` on failure.
    """
    _logger.info("Extracting zonal wind for %s (sample %d) …", label, sample)

    with open_validation(zarr_path) as zio:
        try:
            refs = resolve_channel_arg(
                zio, "u", channels_override, levels_hpa, _DEFAULT_U_LEVELS_HPA, sample
            )
        except KeyError as e:
            _logger.warning("%s: %s — skipping.", label, e)
            return None
        return extract_zonal_wind_from_store(
            zio,
            label,
            refs,
            sample=sample,
            target_latitude=target_latitude,
            climatology_path=climatology_path,
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _merge_targets(
    data_list: list[dict[str, Any]], pressure: float, use_anomaly: bool = False
) -> tuple[list[datetime], list[float]]:
    """
    Merge ERA5 target time series from multiple runs into one.

    Uses anomaly arrays when *use_anomaly* is True, otherwise absolute values.
    """
    seen: dict[datetime, float] = {}
    for d in data_list:
        if pressure not in d["levels"]:
            continue
        lv = d["levels"][pressure]
        vals = lv.get("tgt_anomaly", lv["targets"]) if use_anomaly else lv["targets"]
        for dt, val in zip(d["datetimes"], vals, strict=True):
            if dt not in seen:
                seen[dt] = val
    if not seen:
        return [], []
    sorted_items = sorted(seen.items())
    return [t for t, _ in sorted_items], [v for _, v in sorted_items]


# ---------------------------------------------------------------------------
# Hit/miss-by-init-time (for later grouping into lead-time hit rates)
# ---------------------------------------------------------------------------


def compute_hit_miss(
    data_list: list[dict[str, Any]],
    pressure: float,
    event_date: datetime,
    window_days: float = 1.0,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Per-run hit/miss of the u-wind reversal around *event_date*.

    A run "hits" if its (ensemble-mean) prediction at *pressure* (hPa) drops
    below *threshold* at any verifying time within
    ``[event_date - window_days, event_date + window_days]``.  Each run's
    init time is simply its first datetime — one record per run, keyed by
    init time, with no grouping applied (left to downstream analysis, e.g.
    bucketing by nominal IFS S2S cycle or by lead time).

    Returns a list of dicts::

        {
            "label": str,
            "init_time": "YYYY-MM-DDTHH:MM:SS",
            "hit": bool | None,  # None when the window isn't covered at all
        }
    """
    window_lo = event_date - timedelta(days=window_days)
    window_hi = event_date + timedelta(days=window_days)

    records: list[dict[str, Any]] = []
    for data in data_list:
        if pressure not in data["levels"]:
            continue

        datetimes = data["datetimes"]
        pred_arr = np.asarray(data["levels"][pressure]["predictions"])
        pred_for_ssw = pred_arr.mean(axis=1) if pred_arr.ndim == 2 else pred_arr

        mask = np.array([window_lo <= dt <= window_hi for dt in datetimes])
        hit = bool(np.any(pred_for_ssw[mask] < threshold)) if mask.any() else None

        records.append(
            {
                "label": data["label"],
                "init_time": datetimes[0].isoformat(),
                "hit": hit,
            }
        )
    return sorted(records, key=lambda r: r["init_time"])


def plot_zonal_wind_comparison(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
    event_tag: str = "",
    use_anomaly: bool = False,
) -> None:
    """
    Plot zonal mean u-wind time series: one figure per pressure level with all
    forecasts and a single merged ERA5 target line.

    Args:
        data_list:   List of dicts returned by :func:`extract_zonal_wind`.
        output_dir:  Directory for saved figures.
        ssw_date:    Reference SSW date for vertical line.
        event_tag:   Tag used in output filenames.
        use_anomaly: Plot climatology anomalies instead of absolute values.
                     Requires that ``extract_zonal_wind`` was called with a
                     ``climatology_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_levels: set[float] = set()
    for d in data_list:
        all_levels.update(d["levels"].keys())
    sorted_levels = sorted(all_levels)

    fallback_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    latitude = data_list[0]["latitude"] if data_list else 60.0

    for p in sorted_levels:
        fig, ax = plt.subplots(figsize=(14, 5))

        # One prediction line per run (one thin line per ensemble member when n_ens > 1)
        for i, d in enumerate(data_list):
            if p not in d["levels"]:
                continue
            color = d.get("color") or fallback_colors[i % len(fallback_colors)]
            lv = d["levels"][p]
            plot_vals = (
                lv.get("pred_anomaly", lv["predictions"]) if use_anomaly else lv["predictions"]
            )
            # plot_vals may be (n_steps,) or (n_steps, n_ens)
            plot_vals = np.asarray(plot_vals)
            if plot_vals.ndim == 2 and plot_vals.shape[1] > 1:
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
        tgt_times, tgt_vals = _merge_targets(data_list, p, use_anomaly=use_anomaly)
        if tgt_times:
            ax.plot(tgt_times, tgt_vals, color="k", lw=2, ls="--", label="ERA5")

        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.7)
        all_dates = [dt for d in data_list for dt in d["datetimes"]]
        if all_dates and min(all_dates) <= ssw_date <= max(all_dates):
            ax.axvline(ssw_date, color="red", ls="-.", lw=1.5, alpha=0.7, label="SSW date")

        ylabel = "u-wind anomaly (m/s)" if use_anomaly else "u-wind (m/s)"
        title_suffix = " anomaly" if use_anomaly else ""
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Zonal mean u{title_suffix} at {latitude:g}°N — {format_pressure(p)}",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        # Legend disabled for now — too many experiments make it unreadable
        # (see analyze_polar_cap_temperature.plot_polar_cap_temperature for the
        # same change). Re-enable once a more compact scheme (e.g. colorbar by
        # init date) is in place.
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        tag = event_tag or ssw_date.strftime("%b%Y").lower()
        anom_suffix = "_anomaly" if use_anomaly else ""
        out = output_dir / f"zonal_wind_{latitude:g}N_{level_tag(p)}_{tag}{anom_suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_event_groups(
    args: argparse.Namespace,
) -> dict[str, tuple[list[tuple[str, str, int, int, str | None]], datetime]]:
    """
    Return event name → (list of (label, validation_id, sample, rank, color),
    event SSW reference date).

    When ``--validations-config`` is given, entries are split by their
    ``event``/``group`` field (see
    :func:`weathergen.stratosphere.config.group_validations_by_event`) so
    configs bundling several real-world SSW events produce one set of plots
    per event instead of one unreadable combined plot.  ``--run-ids`` (no
    config) always yields a single ``"default"`` event using ``args.ssw_date``.
    """
    if args.validations_config is not None:
        from weathergen.stratosphere.config import (
            event_reference_date,
            group_validations_by_event,
            load_validations_config,
        )

        cfg = load_validations_config(args.validations_config)
        events = group_validations_by_event(cfg)
        return {
            event_name: (
                [
                    (lbl, spec["id"], spec["sample"], spec.get("rank", 0), spec.get("color"))
                    for lbl, spec in event_cfg.items()
                ],
                event_reference_date(event_cfg, args.ssw_date),
            )
            for event_name, event_cfg in events.items()
        }
    if args.run_ids:
        return {"default": ([(vid, vid, 0, 0, None) for vid in args.run_ids], args.ssw_date)}
    raise SystemExit("Provide --run-ids or --validations-config")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/polar_vortex"))
    parser.add_argument("--run-ids", nargs="+", default=None)
    parser.add_argument("--validations-config", type=Path, default=None)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=None,
        metavar="HPA",
        help=(
            "Pressure levels (hPa) of the zonal-mean u-wind to analyse; the nearest "
            "available level in any stream is used, missing levels are skipped.  "
            "[default: 10 50]"
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Legacy alias: explicit channel names (e.g. u_29 u_55); overrides --levels.",
    )
    parser.add_argument("--latitude", type=float, default=60.0)
    parser.add_argument(
        "--climatology",
        type=Path,
        default=None,
        help="Path to climatology zarr for anomaly computation.",
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference line (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    parser.add_argument(
        "--hit-miss-window-days",
        type=float,
        default=1.0,
        help=(
            "Half-width (days) of the window around each event's reference "
            "date used for the per-init-time hit/miss JSON.  [default: 1.0]"
        ),
    )
    parser.add_argument(
        "--hit-miss-threshold",
        type=float,
        default=0.0,
        help="u-wind threshold (m/s) below which a reversal counts as a hit.  [default: 0.0]",
    )
    args = parser.parse_args(argv)

    event_groups = _build_event_groups(args)
    single_event = len(event_groups) == 1

    for event_name, (run_specs, event_ssw_date) in event_groups.items():
        event_output_dir = args.output_dir if single_event else args.output_dir / event_name
        all_data: list[dict[str, Any]] = []

        for label, vid, sample, rank, color in run_specs:
            zarr_path = zarr_store_path(args.data_dir, vid, rank)
            if not zarr_path.exists():
                _logger.warning("Zarr not found: %s", zarr_path)
                continue

            data = extract_zonal_wind(
                zarr_path,
                label,
                channels_override=args.channels,
                levels_hpa=args.levels,
                sample=sample,
                target_latitude=args.latitude,
                climatology_path=args.climatology,
            )
            if data is None:
                continue
            data["color"] = color
            all_data.append(data)

            # SSW wind-reversal detection on the 10 hPa level only
            ssw_levels = [p for p in data["levels"] if is_ssw_level(p)]
            if not ssw_levels:
                _logger.warning(
                    "[%s] %s: no %s level available (have %s) — SSW detection skipped.",
                    event_name,
                    label,
                    format_pressure(SSW_LEVEL_HPA),
                    [format_pressure(p) for p in data["levels"]],
                )
            for p in ssw_levels:
                lv = data["levels"][p]
                pred_arr = np.asarray(lv["predictions"])
                # Use ensemble mean for SSW detection when multiple members are present
                pred_for_ssw = pred_arr.mean(axis=1) if pred_arr.ndim == 2 else pred_arr
                pred_ssw = detect_ssw_reversal(pred_for_ssw, data["datetimes"])
                tgt_ssw = detect_ssw_reversal(lv["targets"], data["datetimes"])
                _logger.info(
                    "[%s] %s / %s  pred=%s  target=%s",
                    event_name,
                    label,
                    format_pressure(p),
                    pred_ssw,
                    tgt_ssw,
                )

        if not all_data:
            _logger.warning("[%s] No data extracted — skipping event.", event_name)
            continue

        # Keep the historical filename tag (derived from the config's
        # filename stem) when there is only one event, for backward
        # compatibility; otherwise tag filenames with the actual event name.
        if single_event and args.validations_config is not None:
            stem = args.validations_config.stem  # e.g. "ssw_feb2018"
            parts = stem.split("_", 1)
            event_tag = parts[1] if len(parts) == 2 else stem
        else:
            event_tag = event_name

        plot_zonal_wind_comparison(
            all_data, event_output_dir, ssw_date=event_ssw_date, event_tag=event_tag
        )
        if any(d.get("is_anomaly") for d in all_data):
            plot_zonal_wind_comparison(
                all_data,
                event_output_dir,
                ssw_date=event_ssw_date,
                event_tag=event_tag,
                use_anomaly=True,
            )

        # Per-init-time hit/miss JSON (one list per pressure level) — a raw,
        # ungrouped record of which runs' predictions reversed near
        # event_ssw_date, left for downstream grouping (e.g. by nominal IFS
        # S2S cycle or by lead time) rather than aggregated here.
        all_levels = sorted({p for d in all_data for p in d["levels"]})
        hit_miss: dict[str, Any] = {
            "event": event_name,
            "event_date": event_ssw_date.isoformat(),
            "window_days": args.hit_miss_window_days,
            "threshold": args.hit_miss_threshold,
            "latitude": args.latitude,
            "levels": {
                format_pressure(p): compute_hit_miss(
                    all_data,
                    p,
                    event_ssw_date,
                    window_days=args.hit_miss_window_days,
                    threshold=args.hit_miss_threshold,
                )
                for p in all_levels
            },
        }
        event_output_dir.mkdir(parents=True, exist_ok=True)
        hit_miss_path = event_output_dir / f"hit_miss_{event_tag}.json"
        with open(hit_miss_path, "w") as f:
            json.dump(hit_miss, f, indent=2)
        _logger.info("[%s] Saved %s", event_name, hit_miss_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
