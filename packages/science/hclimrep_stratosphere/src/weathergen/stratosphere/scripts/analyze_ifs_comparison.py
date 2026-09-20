#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
IFS S2S Onset Comparison.

Adds the ECMWF IFS S2S ensemble hindcast as a third SSW-onset-timing
baseline (alongside the WeatherGenerator prediction and the ERA5 target),
limited to onset prediction at 10 hPa (u10/t10) as per the combined hindcast
file already downloaded via ``scripts/download_forecast_ifs.py``.

For each ``lead_days`` value present in the validations config, the IFS
initialization closest to ``ssw_date - lead_days`` is selected (nearest
match, skipped with a warning if farther than ``--tolerance-days``), its
full ensemble is run through :func:`detect_ssw_reversal`, and the resulting
onset-date distribution is reduced to a timing-error metric (mean/median/std
in days, consistent with the ``timing_error`` field in
``lead_time_metrics_<channel>.json`` produced by ``analyze_ssw_lead_times.py``).

Usage::

    ssw-analyze ifs-comparison \\
        --ifs-file /path/to/s2s_ecmwf_10hpa_combined.nc \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --output-dir plots/ifs_comparison

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_ifs_comparison \\
        --ifs-file /path/to/s2s_ecmwf_10hpa_combined.nc \\
        --validations-config config/evaluate/ssw_feb2018.yml
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
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.diagnostics import detect_ssw_reversal
from weathergen.stratosphere.io_ifs import (
    extract_ensemble_zonal_mean,
    nearest_init_date,
    open_ifs_hindcast,
)
from weathergen.stratosphere.scripts.analyze_ssw_lead_times import (
    group_by_experiment_and_lead,
)

_logger = logging.getLogger(__name__)

SSW_DATE = datetime(2018, 2, 12)

_VARIABLE_TO_CHANNEL_LABEL = {"u": "u10", "t": "t10"}


# ---------------------------------------------------------------------------
# Onset metrics
# ---------------------------------------------------------------------------


def compute_ifs_onset_metrics(
    ds: Any,
    init_date: datetime,
    ssw_date: datetime,
    variable: str = "u",
    level_hpa: float = 10.0,
    latitude: float = 60.0,
) -> dict[str, Any]:
    """
    Run onset detection on every ensemble member of one IFS initialization.

    Returns a dict with ``lead_days`` (filled in by caller), ``n_members``,
    ``n_detected``, ``detected_fraction``, and ``timing_error_mean/median/std``
    (days, positive = late relative to *ssw_date*; ``None`` if nothing
    detected).
    """
    valid_times, series = extract_ensemble_zonal_mean(
        ds, init_date, variable=variable, level_hpa=level_hpa, target_lat=latitude
    )
    n_members = series.shape[1]

    timing_errors: list[float] = []
    for m in range(n_members):
        result = detect_ssw_reversal(series[:, m], valid_times)
        if result is not None:
            delta = (result["reversal_date"] - ssw_date).total_seconds() / 86400.0
            timing_errors.append(delta)

    n_detected = len(timing_errors)
    metrics: dict[str, Any] = {
        "init_date": init_date.isoformat(),
        "n_members": n_members,
        "n_detected": n_detected,
        "detected_fraction": n_detected / n_members if n_members else 0.0,
        "timing_error_mean": float(np.mean(timing_errors)) if timing_errors else None,
        "timing_error_median": float(np.median(timing_errors))
        if timing_errors
        else None,
        "timing_error_std": float(np.std(timing_errors)) if timing_errors else None,
    }
    return metrics, valid_times, series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ifs_timing_error(
    ifs_metrics: dict[int, dict[str, Any]],
    output_dir: Path,
    variable: str,
    wg_metrics: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> None:
    """
    Plot IFS onset-timing error vs lead time, with mean±std error bars.

    If *wg_metrics* (parsed ``lead_time_metrics_<channel>.json``) is given,
    each WG experiment group's ``timing_error`` is overlaid as a scatter
    series for direct comparison.
    """
    leads = sorted(ifs_metrics)
    if not leads:
        _logger.warning("No IFS onset metrics to plot.")
        return

    means = [ifs_metrics[lv]["timing_error_mean"] for lv in leads]
    stds = [ifs_metrics[lv]["timing_error_std"] or 0.0 for lv in leads]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    valid = [(lv, m, s) for lv, m, s in zip(leads, means, stds) if m is not None]
    if valid:
        v_leads, v_means, v_stds = zip(*valid)
        ax.errorbar(
            v_leads,
            v_means,
            yerr=v_stds,
            marker="o",
            color="k",
            linewidth=2,
            capsize=4,
            label=f"IFS S2S ensemble ({_VARIABLE_TO_CHANNEL_LABEL.get(variable, variable)})",
            zorder=10,
        )

    if wg_metrics:
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
        for i, (exp_label, lead_map) in enumerate(wg_metrics.items()):
            xs = sorted(int(lv) for lv in lead_map)
            ys = [lead_map[str(lv)].get("timing_error") for lv in xs]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.scatter(
                px, py, color=colors[i % len(colors)], label=exp_label, s=40, alpha=0.85
            )

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Lead time (days before onset)", fontsize=12)
    ax.set_ylabel("Onset timing error (days, +late/-early)", fontsize=12)
    ax.set_title("SSW onset timing error vs lead time", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "onset_timing_error_with_ifs.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


def plot_ifs_spaghetti(
    valid_times: list[datetime],
    series: np.ndarray,
    output_dir: Path,
    lead_days: int,
    ssw_date: datetime,
    variable: str,
) -> None:
    """Spaghetti plot of all ensemble members for one IFS initialization."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for m in range(series.shape[1]):
        ax.plot(valid_times, series[:, m], color="steelblue", alpha=0.25, linewidth=0.8)
    ax.plot(valid_times, series.mean(axis=1), color="k", linewidth=2, label="ensemble mean")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    if valid_times[0] <= ssw_date <= valid_times[-1]:
        ax.axvline(ssw_date, color="red", linestyle="-.", linewidth=1.5, label="SSW onset")
    ax.set_xlabel("Date", fontsize=11)
    ylabel = "u (m/s)" if variable == "u" else "T (K)"
    ax.set_ylabel(f"{ylabel} @ 60°N, 10 hPa", fontsize=11)
    ax.set_title(f"IFS S2S ensemble — T-{lead_days}d init", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out = output_dir / f"ifs_spaghetti_t{lead_days:02d}d.png"
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
        description="IFS S2S ensemble onset-timing baseline for SSW lead-time analysis."
    )
    parser.add_argument(
        "--ifs-file",
        type=Path,
        required=True,
        help="Combined IFS S2S hindcast NetCDF file (number/time/step/lat/lon).",
    )
    parser.add_argument(
        "--validations-config",
        type=Path,
        required=True,
        help="YAML config with 'lead_days' fields (see config/evaluate/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/ifs_comparison"),
        help="Directory for output plots and JSON.  [default: plots/ifs_comparison]",
    )
    parser.add_argument(
        "--variable",
        choices=["u", "t"],
        default="u",
        help="Variable to analyse (u10 or t10).  [default: u]",
    )
    parser.add_argument(
        "--level-hpa", type=float, default=10.0, help="Pressure level (hPa). [default: 10]"
    )
    parser.add_argument(
        "--latitude", type=float, default=60.0, help="Latitude (°N). [default: 60]"
    )
    parser.add_argument(
        "--tolerance-days",
        type=float,
        default=1.5,
        help="Max distance (days) between requested and available IFS init date. "
        "[default: 1.5]",
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    parser.add_argument(
        "--wg-metrics-json",
        type=Path,
        default=None,
        help="Optional lead_time_metrics_<channel>.json from analyze_ssw_lead_times.py "
        "to overlay as a comparison baseline.",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("IFS S2S ONSET COMPARISON")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_validations_config(args.validations_config)
    groups = group_by_experiment_and_lead(cfg)
    leads = sorted({lead for lead_map in groups.values() for lead in lead_map})
    if not leads:
        _logger.error("No 'lead_days' entries found in validations config.")
        return

    ds = open_ifs_hindcast(args.ifs_file)
    _logger.info(
        "Opened %s: %d members, %d init dates, %d steps",
        args.ifs_file,
        ds.sizes.get("number", 0),
        ds.sizes.get("time", 0),
        ds.sizes.get("step", 0),
    )

    ifs_metrics: dict[int, dict[str, Any]] = {}
    for lead in leads:
        target_init = args.ssw_date - timedelta(days=lead)
        init = nearest_init_date(ds, target_init, args.tolerance_days)
        if init is None:
            _logger.warning("T-%dd: no IFS init within tolerance — skipping", lead)
            continue

        metrics, valid_times, series = compute_ifs_onset_metrics(
            ds,
            init,
            args.ssw_date,
            variable=args.variable,
            level_hpa=args.level_hpa,
            latitude=args.latitude,
        )
        ifs_metrics[lead] = metrics
        _logger.info(
            "T-%dd (init %s): %d/%d members detected onset, timing_error=%s",
            lead,
            init.date(),
            metrics["n_detected"],
            metrics["n_members"],
            f"{metrics['timing_error_mean']:+.1f}d" if metrics["timing_error_mean"]
            is not None else "N/A",
        )
        plot_ifs_spaghetti(
            valid_times, series, args.output_dir, lead, args.ssw_date, args.variable
        )

    wg_metrics = None
    if args.wg_metrics_json is not None and args.wg_metrics_json.exists():
        with open(args.wg_metrics_json) as f:
            wg_metrics = json.load(f)

    plot_ifs_timing_error(ifs_metrics, args.output_dir, args.variable, wg_metrics)

    metrics_file = args.output_dir / "ifs_onset_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(ifs_metrics, f, indent=2)
    _logger.info("Saved metrics: %s", metrics_file)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
