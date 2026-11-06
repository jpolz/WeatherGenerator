#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
SSW Precursor Diagnostics.

Computes three tropospheric/lower-stratospheric precursor signals commonly
associated with Sudden Stratospheric Warming events, comparing model
predictions against ERA5 targets over the pre-onset window:

  * Eddy heat flux v'T'   — poleward wave-activity proxy from v and t at a
                            single pressure level (default 100 hPa; falls
                            back to 70 or 50 hPa if 100 hPa is unavailable),
                            area-averaged over 45–75°N.
  * Wave-1 / wave-2 amplitude at 60°N — planetary-wave amplitude from
                            geopotential at a stratospheric level (default
                            10 hPa) via harmonic least-squares fit.
  * Blocking index        — Tibaldi-Molteni-style meridional gradient
                            reversal index from geopotential at 500 hPa.

Multi-stream layout: a forecast store may hold several streams, e.g.
``ERA5pl`` (u/v/t/z at 50–1000 hPa + surface fields) and ``S2Spredictors``
(u/v/t/z at 1–70 hPa). Levels are addressed in hPa and located with
:func:`weathergen.stratosphere.io.discover_column`, which merges all streams
into one pressure axis per variable, so the heat flux (100 hPa, ERA5pl), the
wave amplitude (10 hPa, S2Spredictors) and the blocking index (500 hPa,
ERA5pl) can come from different streams.  A diagnostic whose level is not
available in a store (e.g. 10 hPa in the ``memory`` arm) is skipped with a
warning and its panel is omitted.

Usage::

    ssw-analyze precursors \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/precursors \\
        --heat-flux-level 100 --wave-level 10

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_precursors \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from weathergen.stratosphere.config import (
    event_reference_date,
    group_validations_by_event,
    load_validations_config,
)
from weathergen.stratosphere.diagnostics import (
    blocking_index,
    heat_flux_wave_activity,
    wave_amplitude,
)
from weathergen.stratosphere.io import (
    ColumnIndex,
    LevelRef,
    common_pressures,
    convert_times_to_datetime,
    discover_column,
    get_coords,
    get_forecast_steps,
    load_level,
    load_levels,
    open_validation,
    zarr_store_path,
)
from weathergen.stratosphere.levels import format_pressure

_logger = logging.getLogger(__name__)

# Default pressure levels (hPa).
_DEFAULT_HEAT_FLUX_LEVEL = 100.0
_HEAT_FLUX_FALLBACK_LEVELS = (70.0, 50.0)
_DEFAULT_WAVE_LEVEL = 10.0
_DEFAULT_BLOCKING_LEVEL = 500.0

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

_KEYS = ("heat_flux", "wave1", "wave2", "blocking")


# ---------------------------------------------------------------------------
# Level resolution
# ---------------------------------------------------------------------------


def _nearest_pressure(
    pressures: list[float], target: float, tolerance_frac: float = 0.25
) -> float | None:
    """Closest pressure (log-space) within *tolerance_frac* of *target*, or None."""
    if not pressures:
        return None
    p = min(pressures, key=lambda q: abs(math.log(q) - math.log(target)))
    if abs(p - target) / target > tolerance_frac:
        return None
    return p


def resolve_heat_flux_refs(
    v_col: ColumnIndex,
    t_col: ColumnIndex,
    level_hpa: float,
    label: str = "",
) -> tuple[LevelRef, LevelRef] | None:
    """
    Pick v/t :class:`LevelRef` pairs at *level_hpa*, requiring both variables
    on the same pressure; falls back to 70 / 50 hPa with a logged warning.
    """
    common = common_pressures(v_col, t_col)
    candidates = [level_hpa] + [p for p in _HEAT_FLUX_FALLBACK_LEVELS if p != level_hpa]
    for cand in candidates:
        p = _nearest_pressure(common, cand)
        if p is None:
            continue
        if cand != level_hpa:
            _logger.warning(
                "%s: v/t at %s not available — heat flux computed at %s instead",
                label,
                format_pressure(level_hpa),
                format_pressure(p),
            )
        return v_col.nearest(p), t_col.nearest(p)
    _logger.warning(
        "%s: no common v/t level near %s hPa (common levels: %s) — heat flux skipped",
        label,
        [f"{c:g}" for c in candidates],
        [f"{p:g}" for p in common],
    )
    return None


def _resolve_single(col: ColumnIndex, level_hpa: float, what: str, label: str) -> LevelRef | None:
    try:
        return col.nearest(level_hpa)
    except KeyError:
        _logger.warning(
            "%s: %s at %s not available (have %s: %s) — %s skipped",
            label,
            col.variable,
            format_pressure(level_hpa),
            col.variable,
            [f"{p:g}" for p in col.pressures],
            what,
        )
        return None


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_precursors(
    zarr_path: Path,
    label: str,
    sample: int = 0,
    heat_flux_level: float = _DEFAULT_HEAT_FLUX_LEVEL,
    heat_flux_lat_band: tuple[float, float] = (45.0, 75.0),
    wave_level: float = _DEFAULT_WAVE_LEVEL,
    wave_lat: float = 60.0,
    blocking_level: float = _DEFAULT_BLOCKING_LEVEL,
    lead_days: int = 0,
    color: str = "C0",
    ssw_date: datetime = SSW_DATE,
) -> dict[str, Any] | None:
    """
    Extract heat-flux, wave-amplitude, and blocking-index precursors.

    Returns a dict with keys ``label``, ``lead_days``, ``color``,
    ``datetimes``, ``days_rel``, ``levels`` (actual pressure used per
    diagnostic, hPa) and ``pred``/``tgt`` sub-dicts each holding
    ``heat_flux``, ``wave1``, ``wave2``, ``blocking`` arrays (``(n_time,)``,
    NaN where the required level was unavailable); or ``None`` if no
    diagnostic could be computed.
    """
    with open_validation(zarr_path) as zio:
        v_col = discover_column(zio, "v", sample)
        t_col = discover_column(zio, "t", sample)
        z_col = discover_column(zio, "z", sample)

        hf_refs = resolve_heat_flux_refs(v_col, t_col, heat_flux_level, label)
        wave_ref = _resolve_single(z_col, wave_level, "wave amplitude", label)
        blk_ref = _resolve_single(z_col, blocking_level, "blocking index", label)

        if hf_refs is None and wave_ref is None and blk_ref is None:
            _logger.warning("%s: no usable precursor levels found — skipping", label)
            return None

        coords_cache: dict[str, NDArray] = {}

        def _coords(ref: LevelRef) -> NDArray:
            if ref.stream not in coords_cache:
                coords_cache[ref.stream] = get_coords(zio, ref.stream, sample)
            return coords_cache[ref.stream]

        steps = get_forecast_steps(zio, skip_source_step=True)
        n_t = len(steps)
        pred: dict[str, NDArray] = {k: np.full(n_t, np.nan) for k in _KEYS}
        tgt: dict[str, NDArray] = {k: np.full(n_t, np.nan) for k in _KEYS}
        levels: dict[str, float] = {}
        raw_times: list = []

        for i, step in enumerate(steps):
            times = None

            if hf_refs is not None:
                v_ref, t_ref = hf_refs
                levels["heat_flux"] = v_ref.pressure
                vt_pred, vt_tgt, times = load_levels(zio, list(hf_refs), step, sample)
                hf_coords = _coords(v_ref)
                for out, arr in ((pred, vt_pred), (tgt, vt_tgt)):
                    out["heat_flux"][i] = heat_flux_wave_activity(
                        arr[:, 0, 0].astype(np.float64),
                        arr[:, 1, 0].astype(np.float64),
                        hf_coords,
                        heat_flux_lat_band,
                    )

            if wave_ref is not None:
                levels["wave"] = wave_ref.pressure
                z_pred, z_tgt, times_w = load_level(zio, wave_ref, step, sample)
                times = times_w if times is None else times
                w_coords = _coords(wave_ref)
                for wn, key in ((1, "wave1"), (2, "wave2")):
                    pred[key][i] = wave_amplitude(
                        z_pred[:, 0].astype(np.float64), w_coords, wn, wave_lat
                    )
                    tgt[key][i] = wave_amplitude(
                        z_tgt[:, 0].astype(np.float64), w_coords, wn, wave_lat
                    )

            if blk_ref is not None:
                levels["blocking"] = blk_ref.pressure
                z_pred, z_tgt, times_b = load_level(zio, blk_ref, step, sample)
                times = times_b if times is None else times
                b_coords = _coords(blk_ref)
                pred["blocking"][i] = blocking_index(z_pred[:, 0].astype(np.float64), b_coords)[
                    "index"
                ]
                tgt["blocking"][i] = blocking_index(z_tgt[:, 0].astype(np.float64), b_coords)[
                    "index"
                ]

            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    days_rel = np.array(
        [(dt - ssw_date).total_seconds() / 86400.0 for dt in datetimes],
        dtype=np.float64,
    )

    _logger.info(
        "%s: extracted %d steps of precursor diagnostics (levels: %s)",
        label,
        n_t,
        {k: f"{p:g} hPa" for k, p in levels.items()},
    )

    return {
        "label": label,
        "lead_days": lead_days,
        "color": color,
        "datetimes": datetimes,
        "days_rel": days_rel,
        "levels": levels,
        "pred": pred,
        "tgt": tgt,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _panel_label(key: str, data_list: list[dict[str, Any]]) -> str:
    level_key = "wave" if key in ("wave1", "wave2") else key
    used = sorted({d["levels"][level_key] for d in data_list if level_key in d["levels"]})
    lvl = "/".join(format_pressure(p) for p in used)
    if key == "heat_flux":
        return f"Eddy heat flux v'T' at {lvl} (K m/s)"
    if key == "wave1":
        return f"Wave-1 amplitude at 60°N (z {lvl})"
    if key == "wave2":
        return f"Wave-2 amplitude at 60°N (z {lvl})"
    return f"Blocking index (z {lvl}, fraction of blocked longitudes)"


def plot_precursors(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Plot heat-flux, wave-amplitude, and blocking-index precursor time series.

    One stacked figure with one panel per available diagnostic, x-axis = days
    relative to *ssw_date*.  ERA5 target is drawn once (black) using the run
    with the widest window; predictions are drawn per experiment, colored by
    lead time.  Panels for which no run has data are omitted.
    """
    if not data_list:
        _logger.warning("No precursor data to plot.")
        return

    panels = [k for k in _KEYS if any(np.any(np.isfinite(d["pred"][k])) for d in data_list)]
    if not panels:
        _logger.warning("No finite precursor diagnostics to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 4 * len(panels)))
    axes = np.atleast_1d(axes)
    data_sorted = sorted(data_list, key=lambda d: d["days_rel"][0])

    for ax, key in zip(axes, panels, strict=True):
        era5_drawn = False
        for data in data_sorted:
            days_rel = data["days_rel"]
            pred = data["pred"][key]
            if np.all(np.isnan(pred)):
                continue
            ax.plot(
                days_rel,
                pred,
                color=data["color"],
                linestyle="-",
                linewidth=1.5,
                alpha=0.85,
                label=f"t{data['lead_days']:02d}d pred",
            )
            if not era5_drawn:
                tgt = data["tgt"][key]
                if not np.all(np.isnan(tgt)):
                    ax.plot(
                        days_rel,
                        tgt,
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
        ax.set_ylabel(_panel_label(key, data_list), fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=4)

    axes[-1].set_xlabel("Days relative to SSW onset", fontsize=12)
    fig.suptitle("SSW Precursor Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "precursors_timeseries.png"
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
        description="SSW precursor diagnostics: heat flux, wave amplitude, blocking."
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
        default=Path("plots/precursors"),
        help="Directory for output plots.  [default: plots/precursors]",
    )
    parser.add_argument(
        "--heat-flux-level",
        type=float,
        default=_DEFAULT_HEAT_FLUX_LEVEL,
        metavar="HPA",
        help=(
            "Pressure level (hPa) for the eddy heat flux v'T'; v and t must both "
            "exist there (any stream).  Falls back to 70 then 50 hPa if missing.  "
            f"[default: {_DEFAULT_HEAT_FLUX_LEVEL:g}]"
        ),
    )
    parser.add_argument(
        "--heat-flux-lat-band",
        type=float,
        nargs=2,
        metavar=("LAT_MIN", "LAT_MAX"),
        default=[45.0, 75.0],
        help="Latitude band for heat-flux averaging (°N).  [default: 45 75]",
    )
    parser.add_argument(
        "--wave-level",
        type=float,
        default=_DEFAULT_WAVE_LEVEL,
        metavar="HPA",
        help=(
            "Pressure level (hPa) of geopotential for the wave-1/2 amplitude; "
            f"panel skipped if unavailable.  [default: {_DEFAULT_WAVE_LEVEL:g}]"
        ),
    )
    parser.add_argument(
        "--wave-lat",
        type=float,
        default=60.0,
        help="Latitude for wave-1/2 amplitude fit (°N).  [default: 60]",
    )
    parser.add_argument(
        "--blocking-level",
        type=float,
        default=_DEFAULT_BLOCKING_LEVEL,
        metavar="HPA",
        help=(
            "Pressure level (hPa) of geopotential for the blocking index.  "
            f"[default: {_DEFAULT_BLOCKING_LEVEL:g}]"
        ),
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference lines (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    args = parser.parse_args(argv)

    _logger.info("=== SSW PRECURSOR DIAGNOSTICS ===")

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

            data = extract_precursors(
                zarr_path,
                label,
                sample,
                heat_flux_level=args.heat_flux_level,
                heat_flux_lat_band=tuple(args.heat_flux_lat_band),
                wave_level=args.wave_level,
                wave_lat=args.wave_lat,
                blocking_level=args.blocking_level,
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

        plot_precursors(data_list, event_output_dir, event_ssw_date)

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
