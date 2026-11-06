#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Forecast-climatology diagnostics: lead-dependent bias, RMSE and blurring.

For every validation entry and requested pressure level the script computes,
at every forecast step, the area-weighted bias, RMSE and spatial-variance
ratio (pred/tgt — the "blurring" measure) both globally and over the 60–90°N
polar cap, the zonal-mean bias by latitude at the last step, and the zonal
wavenumber spectrum of ``z`` at 10 hPa along 60°N at selected lead days.

Multi-stream layout: the forecast store of the strato_s2s arms holds the
column in several streams (``ERA5pl`` 50–1000 hPa, ``S2Spredictors`` 1–70
hPa).  Levels are resolved in hPa through ``discover_column``; a level that
is absent (e.g. 10 hPa in the ``memory`` arm) is skipped with a warning.

Usage::

    ssw-analyze forecast-climatology \\
        --config config/evaluate/ssw_feb2018.yml \\
        --data-dir results --output-dir plots/forecast_climatology \\
        [--variables u t] [--levels 10 50 100 500] [--spectrum-days 1 7 14 28]

Outputs per event: ``forecast_climatology_<event>.nc`` plus
``forecast_climatology_<event>_<var>_lead.png``,
``..._<var>_zonal_bias.png`` and ``..._z10_spectrum.png``.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import xarray as xr
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from weathergen.stratosphere.config import (  # noqa: E402
    group_validations_by_event,
    load_validations_config,
)
from weathergen.stratosphere.coupling import (  # noqa: E402
    lead_dependent_bias,
    zonal_mean_by_latitude,
    zonal_wavenumber_spectrum,
)
from weathergen.stratosphere.io import (  # noqa: E402
    LevelRef,
    discover_column,
    find_polar_cap_indices,
    get_coords,
    get_forecast_steps,
    get_o_grid_weights,
    load_level,
    load_levels,
    open_validation,
    zarr_store_path,
)
from weathergen.stratosphere.levels import format_pressure  # noqa: E402

_logger = logging.getLogger(__name__)

_DEFAULT_VARIABLES = ["u", "t"]
_DEFAULT_LEVELS = [10.0, 50.0, 100.0, 500.0]
_DEFAULT_SPECTRUM_DAYS = [1, 7, 14, 28]
_KMAX = 10
_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
_LAT_BINS = np.arange(-90.0, 91.0, 5.0)


def _safe(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s)


def _nearest_key(dct: dict[float, Any], p: float, tol: float = 0.25) -> float | None:
    """Key of *dct* within *tol* (relative) of pressure *p*, else ``None``."""
    cands = [q for q in dct if abs(q - p) / p <= tol]
    return min(cands, key=lambda q: abs(q - p)) if cands else None


def _get_level(dct: dict[float, Any], p: float) -> Any | None:
    q = _nearest_key(dct, p)
    return None if q is None else dct[q]


def _step_hours(zio, ref: LevelRef, steps: list[int], sample: int) -> float:
    """Forecast step length in hours (default 6 h if it cannot be inferred)."""
    if len(steps) < 2:
        return 6.0
    _, _, t1 = load_level(zio, ref, steps[0], sample)
    _, _, t2 = load_level(zio, ref, steps[1], sample)
    dh = (t2[0] - t1[0]) / np.timedelta64(1, "h")
    return float(dh) if np.isfinite(dh) and dh > 0 else 6.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_entry(
    zio,
    label: str,
    sample: int,
    variables: list[str],
    levels: list[float],
    spectrum_days: list[int],
    min_lat: float,
) -> dict[str, Any] | None:
    """
    Per-step metrics for one validation entry.

    Returns a dict with ``steps``, ``lead_hours``, ``metrics[var][p]`` →
    ``{"bias","rmse","var_ratio","bias_cap","rmse_cap","var_ratio_cap"}``
    (each ``(n_steps,)``), ``zonal_bias[var][p]`` ``(n_lat,)``, ``lat`` and
    ``spectrum[day]`` → ``{"pred","tgt"}`` ``(kmax+1,)``; ``None`` if no level
    was found.
    """
    steps = get_forecast_steps(zio, skip_source_step=True)
    refs: list[LevelRef] = []
    for var in variables:
        col = discover_column(zio, var, sample)
        if not col:
            _logger.warning("%s: variable %r not found in streams %s", label, var, zio.streams)
            continue
        found = col.select(levels)
        _logger.info("%s: %s levels %s hPa", label, var, [f"{r.pressure:g}" for r in found])
        refs.extend(found)
    if not refs:
        return None

    coords = get_coords(zio, refs[0].stream, sample)
    w_glob = get_o_grid_weights(coords)
    cap = find_polar_cap_indices(coords, min_lat, "north")
    step_hours = _step_hours(zio, refs[0], steps, sample)

    z_col = discover_column(zio, "z", sample)
    z10: LevelRef | None = None
    try:
        z10 = z_col.nearest(10.0)
    except KeyError:
        _logger.warning("%s: z 10 hPa unavailable — skipping wavenumber spectrum", label)
    spec_steps = {d: int(round(d * 24 / step_hours)) for d in spectrum_days}

    n_steps = len(steps)
    keys = ("bias", "rmse", "var_ratio", "bias_cap", "rmse_cap", "var_ratio_cap")
    metrics: dict[str, dict[float, dict[str, NDArray]]] = {}
    for ref in refs:
        metrics.setdefault(ref.variable, {})[ref.pressure] = {
            k: np.full(n_steps, np.nan) for k in keys
        }
    zonal_bias: dict[str, dict[float, NDArray]] = {}
    spectrum: dict[int, dict[str, NDArray]] = {}
    lat_c = None

    for i, step in enumerate(steps):
        pred, tgt, _ = load_levels(zio, refs, step, sample)
        pred = pred.mean(axis=-1)  # ensemble mean -> (n_pts, n_lev)
        tgt = tgt.mean(axis=-1)
        for j, ref in enumerate(refs):
            m = metrics[ref.variable][ref.pressure]
            b, r, v = lead_dependent_bias(pred[None, :, j], tgt[None, :, j], w_glob)
            m["bias"][i], m["rmse"][i], m["var_ratio"][i] = b[0], r[0], v[0]
            b, r, v = lead_dependent_bias(pred[None, cap, j], tgt[None, cap, j], w_glob[cap])
            m["bias_cap"][i], m["rmse_cap"][i], m["var_ratio_cap"][i] = b[0], r[0], v[0]
            if step == steps[-1]:
                lat_c, zb = zonal_mean_by_latitude(pred[:, j] - tgt[:, j], coords, _LAT_BINS)
                zonal_bias.setdefault(ref.variable, {})[ref.pressure] = zb
        if z10 is not None and step in spec_steps.values():
            zp, zt, _ = load_level(zio, z10, step, sample)
            day = next(d for d, s in spec_steps.items() if s == step)
            spectrum[day] = {
                "pred": zonal_wavenumber_spectrum(zp.mean(-1), coords, kmax=_KMAX),
                "tgt": zonal_wavenumber_spectrum(zt.mean(-1), coords, kmax=_KMAX),
            }
    missing_days = [d for d, s in spec_steps.items() if s not in steps]
    if z10 is not None and missing_days:
        _logger.warning("%s: no forecast step for spectrum days %s", label, missing_days)

    return {
        "label": label,
        "steps": np.asarray(steps),
        "lead_hours": np.asarray(steps) * step_hours,
        "metrics": metrics,
        "zonal_bias": zonal_bias,
        "lat": lat_c,
        "spectrum": spectrum,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lead_metrics(data: list[dict], var: str, levels: list[float], out: Path) -> None:
    """Bias / RMSE / var-ratio vs lead time; one column per level, lines per entry."""
    fig, axes = plt.subplots(
        3, len(levels), figsize=(3.6 * len(levels), 8.5), squeeze=False, sharex=True
    )
    rows = (("bias", "Bias"), ("rmse", "RMSE"), ("var_ratio", "Var ratio pred/tgt"))
    for c, p in enumerate(levels):
        for r, (key, name) in enumerate(rows):
            ax = axes[r, c]
            for k, d in enumerate(data):
                m = _get_level(d["metrics"].get(var, {}), p)
                if m is None:
                    continue
                col = _COLORS[k % len(_COLORS)]
                days = d["lead_hours"] / 24.0
                ax.plot(days, m[key], color=col, label=f"{d['label']} global")
                ax.plot(days, m[key + "_cap"], color=col, ls="--", label=f"{d['label']} ≥60°N")
            if key == "bias":
                ax.axhline(0, color="k", lw=0.6)
            if key == "var_ratio":
                ax.axhline(1, color="k", lw=0.6)
                ax.set_xlabel("Lead time (days)")
            ax.set_ylabel(name)
            if r == 0:
                ax.set_title(f"{var} {format_pressure(p)}")
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    _logger.info("Saved %s", out)


def plot_zonal_bias(data: list[dict], var: str, levels: list[float], out: Path) -> None:
    """Zonal-mean bias at the last forecast step by latitude, one panel per level."""
    fig, axes = plt.subplots(1, len(levels), figsize=(3.6 * len(levels), 3.6), squeeze=False)
    for c, p in enumerate(levels):
        ax = axes[0, c]
        for k, d in enumerate(data):
            zb = _get_level(d["zonal_bias"].get(var, {}), p)
            if zb is None:
                continue
            ax.plot(d["lat"], zb, "o-", ms=2, color=_COLORS[k % len(_COLORS)], label=d["label"])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{var} {format_pressure(p)} — last step")
        ax.set_xlabel("Latitude (°)")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Zonal-mean bias")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    _logger.info("Saved %s", out)


def plot_spectra(data: list[dict], days: list[int], out: Path) -> None:
    """Zonal wavenumber power of z 10 hPa at 60°N; one panel per lead day."""
    days = [d for d in days if any(d in e["spectrum"] for e in data)]
    if not days:
        return
    fig, axes = plt.subplots(1, len(days), figsize=(3.6 * len(days), 3.6), squeeze=False)
    k = np.arange(_KMAX + 1)
    for c, day in enumerate(days):
        ax = axes[0, c]
        positive = False
        for i, d in enumerate(data):
            sp = d["spectrum"].get(day)
            if sp is None:
                continue
            col = _COLORS[i % len(_COLORS)]
            ax.plot(k[1:], sp["pred"][1:], "o-", color=col, label=f"{d['label']} pred")
            ax.plot(k[1:], sp["tgt"][1:], "s--", color=col, alpha=0.6, label="target")
            positive |= bool(np.nanmax(np.r_[sp["pred"][1:], sp["tgt"][1:]]) > 0)
        if positive:
            ax.set_yscale("log")
        ax.set_title(f"z 10 hPa @60°N, day {day}")
        ax.set_xlabel("Zonal wavenumber k")
        ax.grid(True, alpha=0.3, which="both")
    axes[0, 0].set_ylabel("Power (squared amplitude)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    _logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# NetCDF
# ---------------------------------------------------------------------------


def build_dataset(
    data: list[dict], variables: list[str], levels: list[float], days: list[int]
) -> xr.Dataset:
    """Pack per-entry metrics into one Dataset (NaN where a level is missing)."""
    labels = [d["label"] for d in data]
    n_steps = max(len(d["steps"]) for d in data)
    ref = max(data, key=lambda d: len(d["steps"]))
    lat = next((d["lat"] for d in data if d["lat"] is not None), np.array([]))
    shape = (len(data), len(variables), len(levels), n_steps)
    keys = ("bias", "rmse", "var_ratio", "bias_cap", "rmse_cap", "var_ratio_cap")
    arrs = {k: np.full(shape, np.nan) for k in keys}
    zonal = np.full(shape[:3] + (len(lat),), np.nan)
    spec = np.full((len(data), len(days), 2, _KMAX + 1), np.nan)

    for i, d in enumerate(data):
        for j, var in enumerate(variables):
            for m, p in enumerate(levels):
                q = _nearest_key(d["metrics"].get(var, {}), p)
                if q is None:
                    continue
                n = len(d["steps"])
                for k in keys:
                    arrs[k][i, j, m, :n] = d["metrics"][var][q][k]
                zonal[i, j, m] = d["zonal_bias"][var][q]
        for n_d, day in enumerate(days):
            if day in d["spectrum"]:
                spec[i, n_d, 0] = d["spectrum"][day]["pred"]
                spec[i, n_d, 1] = d["spectrum"][day]["tgt"]

    ds = xr.Dataset(
        {k: (("entry", "variable", "level", "step"), v) for k, v in arrs.items()}
        | {
            "zonal_bias_last": (("entry", "variable", "level", "lat"), zonal),
            "z10_spectrum": (("entry", "day", "kind", "k"), spec),
        },
        coords={
            "entry": labels,
            "variable": variables,
            "level": ("level", np.asarray(levels, float), {"units": "hPa"}),
            "step": ref["steps"],
            "lead_hours": ("step", ref["lead_hours"]),
            "lat": lat,
            "day": days,
            "kind": ["pred", "tgt"],
            "k": np.arange(_KMAX + 1),
        },
    )
    ds["var_ratio"].attrs["description"] = "weighted spatial variance pred / tgt (blurring)"
    ds["z10_spectrum"].attrs["description"] = "squared amplitude per zonal wavenumber at 60°N"
    return ds


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
        description="Lead-dependent bias, RMSE, blurring and zonal spectra of forecasts."
    )
    parser.add_argument(
        "--validations-config",
        "--config",
        dest="validations_config",
        type=Path,
        required=True,
        help="YAML validations config (see config/evaluate/).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/forecast_climatology"))
    parser.add_argument(
        "--variables", nargs="+", default=_DEFAULT_VARIABLES, help="Variables (default: u t)."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=_DEFAULT_LEVELS,
        metavar="hPa",
        help="Pressure levels in hPa (default: 10 50 100 500); missing ones are skipped.",
    )
    parser.add_argument(
        "--spectrum-days",
        nargs="+",
        type=int,
        default=_DEFAULT_SPECTRUM_DAYS,
        help="Lead days for the z 10 hPa wavenumber spectrum (default: 1 7 14 28).",
    )
    parser.add_argument("--min-latitude", type=float, default=60.0, help="Polar-cap edge (°N).")
    args = parser.parse_args(argv)

    cfg = load_validations_config(args.validations_config)
    events = group_validations_by_event(cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    any_data = False

    for event, event_cfg in events.items():
        data: list[dict] = []
        for label, spec in event_cfg.items():
            path = zarr_store_path(args.data_dir, spec["id"], spec.get("rank", 0))
            if not path.exists():
                _logger.warning("Store not found, skipping %s: %s", label, path)
                continue
            with open_validation(path) as zio:
                d = extract_entry(
                    zio,
                    label,
                    spec["sample"],
                    args.variables,
                    args.levels,
                    args.spectrum_days,
                    args.min_latitude,
                )
            if d is None:
                _logger.warning("%s: no requested level found — skipped", label)
                continue
            data.append(d)
        if not data:
            _logger.warning("[%s] no data — skipping event", event)
            continue
        any_data = True
        tag = _safe(event)
        for var in args.variables:
            lv = [
                p
                for p in args.levels
                if any(_nearest_key(d["metrics"].get(var, {}), p) is not None for d in data)
            ]
            if not lv:
                _logger.warning("[%s] %s: no level available in any entry", event, var)
                continue
            plot_lead_metrics(
                data, var, lv, args.output_dir / f"forecast_climatology_{tag}_{var}_lead.png"
            )
            plot_zonal_bias(
                data,
                var,
                lv,
                args.output_dir / f"forecast_climatology_{tag}_{var}_zonal_bias.png",
            )
        plot_spectra(
            data,
            args.spectrum_days,
            args.output_dir / f"forecast_climatology_{tag}_z10_spectrum.png",
        )
        ds = build_dataset(data, args.variables, args.levels, args.spectrum_days)
        nc = args.output_dir / f"forecast_climatology_{tag}.nc"
        ds.to_netcdf(nc)
        _logger.info("Saved %s", nc)

    if not any_data:
        _logger.warning("No data extracted — check data-dir and validations config.")
        return
    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
