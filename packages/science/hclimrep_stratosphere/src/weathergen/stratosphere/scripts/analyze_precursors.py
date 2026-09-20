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

  * Eddy heat flux v'T'   — poleward wave-activity proxy, from raw ERA5ml
                            u/v/t model-level channels (default level 61,
                            pf≈103.7 hPa — closest available channel to
                            100 hPa in the ``era5_strato_final`` stream).
  * Wave-1 / wave-2 amplitude at 60°N — planetary-wave amplitude from
                            ERA5pl ``z_500`` (harmonic least-squares fit;
                            ``z_100`` does not exist in this stream).
  * Blocking index        — Tibaldi-Molteni-style meridional Z500 gradient
                            reversal index, from ERA5pl ``z_500``.

Each diagnostic is skipped (with a warning) if its required stream
(ERA5ml for heat flux, ERA5pl for wave amplitude/blocking) is not present
in a given validation store.

Usage::

    ssw-analyze precursors \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/precursors

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_precursors \\
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
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.diagnostics import (
    blocking_index,
    heat_flux_wave_activity,
    wave_amplitude,
)
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    get_channels,
    get_coords,
    get_forecast_steps,
    load_step,
    open_validation,
)

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# Default ERA5ml model level for the heat-flux proxy (pf≈103.7 hPa; closest
# available channel to 100 hPa in the era5_strato_final stream).
_DEFAULT_HEAT_FLUX_LEVEL = 61

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_precursors(
    zarr_path: Path,
    label: str,
    sample: int = 0,
    heat_flux_level: int = _DEFAULT_HEAT_FLUX_LEVEL,
    heat_flux_lat_band: tuple[float, float] = (45.0, 75.0),
    wave_lat: float = 60.0,
    lead_days: int = 0,
    color: str = "C0",
    ssw_date: datetime = SSW_DATE,
) -> dict[str, Any] | None:
    """
    Extract heat-flux, wave-amplitude, and blocking-index precursors.

    Returns a dict with keys ``label``, ``lead_days``, ``color``,
    ``datetimes``, ``days_rel``, and ``pred``/``tgt`` sub-dicts each holding
    ``heat_flux``, ``wave1``, ``wave2``, ``blocking`` arrays (``(n_time,)``,
    NaN where the required stream was unavailable); or ``None`` if neither
    ERA5ml nor ERA5pl is present.
    """
    with open_validation(zarr_path) as zio:
        has_ml = "ERA5ml" in zio.streams
        has_pl = "ERA5pl" in zio.streams

        if not has_ml and not has_pl:
            _logger.warning("%s: neither ERA5ml nor ERA5pl present — skipping", label)
            return None

        steps = get_forecast_steps(zio, skip_source_step=True)

        v_idx = t_idx = None
        if has_ml:
            ml_channels = get_channels(zio, "ERA5ml", sample)
            v_ch, t_ch = f"v_{heat_flux_level}", f"t_{heat_flux_level}"
            if v_ch in ml_channels and t_ch in ml_channels:
                v_idx = ml_channels.index(v_ch)
                t_idx = ml_channels.index(t_ch)
                ml_coords = get_coords(zio, "ERA5ml", sample)
            else:
                _logger.warning(
                    "%s: %s/%s not found in ERA5ml — heat flux skipped. "
                    "Available v/t channels: %s",
                    label,
                    v_ch,
                    t_ch,
                    [c for c in ml_channels if c.startswith(("v_", "t_"))],
                )
                has_ml = False

        z_idx = None
        if has_pl:
            pl_channels = get_channels(zio, "ERA5pl", sample)
            if "z_500" in pl_channels:
                z_idx = pl_channels.index("z_500")
                pl_coords = get_coords(zio, "ERA5pl", sample)
            else:
                _logger.warning(
                    "%s: z_500 not found in ERA5pl — wave/blocking skipped. "
                    "Available z channels: %s",
                    label,
                    [c for c in pl_channels if c.startswith("z_")],
                )
                has_pl = False

        if not has_ml and not has_pl:
            _logger.warning("%s: no usable precursor channels found — skipping", label)
            return None

        n_t = len(steps)
        keys = ("heat_flux", "wave1", "wave2", "blocking")
        pred: dict[str, np.ndarray] = {k: np.full(n_t, np.nan) for k in keys}
        tgt: dict[str, np.ndarray] = {k: np.full(n_t, np.nan) for k in keys}
        raw_times: list = []

        for i, step in enumerate(steps):
            stream_for_step = "ERA5ml" if has_ml else "ERA5pl"
            _, _, times = load_step(zio, stream_for_step, step, sample)
            raw_times.append(times[0])

            if has_ml:
                ml_pred, ml_tgt, _ = load_step(zio, "ERA5ml", step, sample)
                ml_pred = np.atleast_3d(ml_pred)
                ml_tgt = np.atleast_3d(ml_tgt)
                pred["heat_flux"][i] = heat_flux_wave_activity(
                    ml_pred[:, v_idx, 0].astype(np.float64),
                    ml_pred[:, t_idx, 0].astype(np.float64),
                    ml_coords,
                    heat_flux_lat_band,
                    heat_flux_level,
                )
                tgt["heat_flux"][i] = heat_flux_wave_activity(
                    ml_tgt[:, v_idx, 0].astype(np.float64),
                    ml_tgt[:, t_idx, 0].astype(np.float64),
                    ml_coords,
                    heat_flux_lat_band,
                    heat_flux_level,
                )

            if has_pl:
                pl_pred, pl_tgt, _ = load_step(zio, "ERA5pl", step, sample)
                pl_pred = np.atleast_3d(pl_pred)
                pl_tgt = np.atleast_3d(pl_tgt)
                for wn, key in ((1, "wave1"), (2, "wave2")):
                    pred[key][i] = wave_amplitude(
                        pl_pred[:, z_idx, 0].astype(np.float64), pl_coords, wn, wave_lat
                    )
                    tgt[key][i] = wave_amplitude(
                        pl_tgt[:, z_idx, 0].astype(np.float64), pl_coords, wn, wave_lat
                    )
                pred["blocking"][i] = blocking_index(
                    pl_pred[:, z_idx, 0].astype(np.float64), pl_coords
                )["index"]
                tgt["blocking"][i] = blocking_index(
                    pl_tgt[:, z_idx, 0].astype(np.float64), pl_coords
                )["index"]

    datetimes = convert_times_to_datetime(np.array(raw_times))
    days_rel = np.array(
        [(dt - ssw_date).total_seconds() / 86400.0 for dt in datetimes],
        dtype=np.float64,
    )

    _logger.info("%s: extracted %d steps of precursor diagnostics", label, n_t)

    return {
        "label": label,
        "lead_days": lead_days,
        "color": color,
        "datetimes": datetimes,
        "days_rel": days_rel,
        "pred": pred,
        "tgt": tgt,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_PANELS = (
    ("heat_flux", "Eddy heat flux v'T' (K m/s)"),
    ("wave1", "Wave-1 amplitude at 60°N (z_500)"),
    ("wave2", "Wave-2 amplitude at 60°N (z_500)"),
    ("blocking", "Blocking index (fraction of blocked longitudes)"),
)


def plot_precursors(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
) -> None:
    """
    Plot heat-flux, wave-amplitude, and blocking-index precursor time series.

    One 4-panel figure (stacked), x-axis = days relative to *ssw_date*.
    ERA5 target is drawn once (black) using the run with the widest window;
    predictions are drawn per experiment, colored by lead time.
    """
    if not data_list:
        _logger.warning("No precursor data to plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(_PANELS), 1, figsize=(13, 4 * len(_PANELS)))
    data_sorted = sorted(data_list, key=lambda d: d["days_rel"][0])

    for ax, (key, ylabel) in zip(axes, _PANELS):
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
        ax.set_ylabel(ylabel, fontsize=11)
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
        type=int,
        default=_DEFAULT_HEAT_FLUX_LEVEL,
        help=(
            "ERA5ml model level for the heat-flux proxy.  "
            f"[default: {_DEFAULT_HEAT_FLUX_LEVEL} (pf≈103.7 hPa)]"
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
        "--wave-lat",
        type=float,
        default=60.0,
        help="Latitude for wave-1/2 amplitude fit (°N).  [default: 60]",
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference lines (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("SSW PRECURSOR DIAGNOSTICS")
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

        data = extract_precursors(
            zarr_path,
            label,
            sample,
            heat_flux_level=args.heat_flux_level,
            heat_flux_lat_band=tuple(args.heat_flux_lat_band),
            wave_lat=args.wave_lat,
            lead_days=lead_days,
            color=color,
            ssw_date=args.ssw_date,
        )
        if data is None:
            _logger.warning("Skipping %s — extraction failed.", label)
            continue
        data_list.append(data)

    plot_precursors(data_list, args.output_dir, args.ssw_date)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
