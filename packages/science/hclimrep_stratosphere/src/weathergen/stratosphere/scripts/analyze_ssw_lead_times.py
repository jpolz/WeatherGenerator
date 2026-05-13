#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
SSW Lead-Time Skill Analysis.

Analyses how SSW prediction skill degrades with increasing lead time.
Expects a validations config that groups runs by experiment and lead time.

Four plots are produced:
  1. u60N time series per experiment — panels by lead time, ensemble spread shaded
  2. SSW onset timing error vs lead time — scatter per experiment
  3. u60N RMSE at the SSW date vs lead time — bar chart per experiment
  4. Ensemble spread (std) at the SSW date vs lead time — bar chart

Usage::

    ssw-analyze ssw-lead-times \\
        --validations-config eval_config/ssw_lead_times.yml \\
        --data-dir /path/to/validation/data \\
        --output-dir plots/lead_time \\
        --channel u_30
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
from weathergen.stratosphere.levels import channel_pressure

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"
SSW_DATE = datetime(2018, 2, 12)

_FALLBACK_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def _channel_label(channel: str) -> str:
    """Return a human-readable pressure label for a channel name."""
    p = channel_pressure(channel)
    if p is None:
        return channel
    parts = channel.split("_")
    level = int(parts[1]) if len(parts) == 2 else None
    if level is not None and level < 150:
        return f"{p:.0f} hPa (L{level})"
    return f"{p:.0f} hPa"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_run(
    zarr_path: Path,
    label: str,
    channel: str,
    sample: int = 0,
    target_latitude: float = 60.0,
) -> dict[str, Any] | None:
    """
    Extract zonal mean u-wind for a single channel from one validation zarr.

    Args:
        zarr_path:       Path to the ``.zarr`` store.
        label:           Human-readable label for logging.
        channel:         Channel name (e.g. ``'u_30'``).
        sample:          Ensemble member index (default 0).
        target_latitude: Latitude for zonal mean (default 60°N).

    Returns:
        Dict with keys ``label``, ``channel``, ``sample``, ``predictions``,
        ``targets``, ``datetimes``; or ``None`` on failure.
    """
    _logger.info("Loading %s (sample %d) …", label, sample)

    with open_validation(zarr_path) as zio:
        stream = get_stream(zio)
        steps = get_forecast_steps(zio)
        channels = get_channels(zio, stream, sample)
        coords = get_coords(zio, stream, sample)

        if channel not in channels:
            _logger.warning(
                "Channel '%s' not found in %s. Available: %s",
                channel,
                label,
                [c for c in channels if c.startswith(channel.split("_")[0] + "_")],
            )
            return None

        ch_idx = channels.index(channel)
        lat_idx = find_latitude_indices(coords, target_latitude)

        preds: list[float] = []
        targets: list[float] = []
        times_list = []

        for step in steps:
            pred, tgt, times = load_step(zio, stream, step, sample)
            preds.append(float(np.mean(pred[lat_idx, ch_idx, 0])))
            tgt_slice = (
                tgt[lat_idx, ch_idx, 0] if tgt.ndim == 3 else tgt[lat_idx, ch_idx]
            )
            targets.append(float(np.mean(tgt_slice)))
            times_list.append(times[0])

    datetimes = convert_times_to_datetime(np.array(times_list))
    _logger.info("  %d steps, %s → %s", len(steps), datetimes[0], datetimes[-1])

    return {
        "label": label,
        "channel": channel,
        "sample": sample,
        "predictions": np.array(preds),
        "targets": np.array(targets),
        "datetimes": datetimes,
    }


def load_experiment_group(
    group_cfg: dict[str, dict[str, Any]],
    data_dir: Path,
    channel: str,
) -> list[dict[str, Any]]:
    """
    Load all runs in one experiment group.

    Returns list of run dicts enriched with ``lead_days``, ``color``,
    ``validation_id``.
    """
    runs = []
    for label, spec in group_cfg.items():
        val_id = spec["id"]
        sample = spec.get("sample", 0)
        lead_days = spec.get("lead_days")
        color = spec.get("color")

        zarr_path = data_dir / val_id / _ZARR_FNAME
        if not zarr_path.exists():
            _logger.warning("Zarr not found, skipping %s: %s", label, zarr_path)
            continue

        run = extract_run(zarr_path, label, channel, sample=sample)
        if run is None:
            continue

        run["lead_days"] = lead_days
        run["color"] = color
        run["validation_id"] = val_id
        runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def ensemble_mean_std(
    runs: list[dict[str, Any]], key: str = "predictions"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) over the samples dimension."""
    arrays = np.stack([r[key] for r in runs], axis=0)  # (n_samples, time)
    return arrays.mean(axis=0), arrays.std(axis=0)


def rmse_at_ssw(runs: list[dict[str, Any]]) -> float:
    """RMSE of ensemble mean vs target at the time step closest to SSW_DATE."""
    datetimes = runs[0]["datetimes"]
    target = runs[0]["targets"]
    diffs = [abs((dt - SSW_DATE).total_seconds()) for dt in datetimes]
    idx = int(np.argmin(diffs))
    mean_pred, _ = ensemble_mean_std(runs)
    return float(np.sqrt((mean_pred[idx] - target[idx]) ** 2))


def ssw_timing_error(run: dict[str, Any]) -> float | None:
    """
    Return prediction SSW onset timing error in days (positive = late).
    ``None`` if SSW not detected in either prediction or target.
    """
    pred_ssw = detect_ssw_reversal(run["predictions"], run["datetimes"])
    tgt_ssw = detect_ssw_reversal(run["targets"], run["datetimes"])
    if pred_ssw is None or tgt_ssw is None:
        return None
    delta = (
        pred_ssw["reversal_date"] - tgt_ssw["reversal_date"]
    ).total_seconds() / 86400
    return float(delta)


def ensemble_spread_at_ssw(runs: list[dict[str, Any]]) -> float:
    """Ensemble std of prediction at the time step closest to SSW_DATE."""
    datetimes = runs[0]["datetimes"]
    diffs = [abs((dt - SSW_DATE).total_seconds()) for dt in datetimes]
    idx = int(np.argmin(diffs))
    _, std = ensemble_mean_std(runs)
    return float(std[idx])


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def group_by_experiment_and_lead(
    cfg: dict[str, dict[str, Any]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Split a flat validations config into::

        {experiment_label: {lead_days: {label: spec}}}

    Entries without a ``lead_days`` field are skipped.
    Entries are grouped by their ``group`` field (default: ``"default"``).
    """
    groups: dict[str, dict[int, dict]] = {}
    for label, spec in cfg.items():
        if not isinstance(spec, dict) or spec.get("lead_days") is None:
            continue
        exp = spec.get("group") or "default"
        lead = int(spec["lead_days"])
        groups.setdefault(exp, {}).setdefault(lead, {})[label] = spec
    return groups


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_timeseries_by_lead_time(
    lead_groups: dict[int, list[dict[str, Any]]],
    exp_label: str,
    channel: str,
    output_dir: Path,
) -> None:
    """Plot 1: one panel per lead time showing ensemble mean ± spread vs ERA5."""
    level_label = _channel_label(channel)
    lead_times = sorted(lead_groups.keys(), reverse=True)
    if not lead_times:
        return

    fig, axes = plt.subplots(
        len(lead_times), 1, figsize=(14, 4 * len(lead_times)), squeeze=False
    )

    for ax, lead in zip(axes.flatten(), lead_times):
        runs = lead_groups[lead]
        mean_pred, std_pred = ensemble_mean_std(runs)
        datetimes = runs[0]["datetimes"]
        target = runs[0]["targets"]
        color = runs[0].get("color") or "#377eb8"

        ax.fill_between(
            datetimes,
            mean_pred - std_pred,
            mean_pred + std_pred,
            color=color,
            alpha=0.25,
            label=f"Ensemble spread (n={len(runs)})",
        )
        ax.plot(
            datetimes, mean_pred, color=color, lw=2.5, label="Prediction (ens. mean)"
        )
        ax.plot(datetimes, target, color="black", lw=2, ls="--", label="ERA5 target")
        ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.7)

        if min(datetimes) <= SSW_DATE <= max(datetimes):
            ax.axvline(
                SSW_DATE,
                color="red",
                ls="-.",
                lw=1.5,
                alpha=0.8,
                label="Observed SSW (Feb 12, 2018)",
            )

        ax.set_title(
            f"T–{lead}d  (init: {datetimes[0].strftime('%Y-%m-%d')})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("U-wind (m/s)", fontsize=10)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    axes.flatten()[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle(
        f"{exp_label} — 60°N Zonal Mean U-wind at {level_label} by Lead Time",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out = output_dir / f"lead_time_series_{exp_label.replace(' ', '_')}_{channel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved %s", out)


def plot_timing_error(
    all_exp_metrics: dict[str, dict[int, dict[str, Any]]],
    channel: str,
    output_dir: Path,
) -> None:
    """Plot 2: SSW onset timing error (days) vs lead time, one series per experiment."""
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for i, (exp_label, metrics) in enumerate(all_exp_metrics.items()):
        color = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        xs = [
            lead
            for lead, m in sorted(metrics.items())
            if m.get("timing_error") is not None
        ]
        ys = [metrics[lead]["timing_error"] for lead in xs]
        if not xs:
            continue
        has_data = True
        ax.plot(xs, ys, "o-", color=color, lw=2, markersize=8, label=exp_label)

    if not has_data:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Lead time (days before SSW)", fontsize=12)
    ax.set_ylabel("Timing error (days, positive = late)", fontsize=12)
    ax.set_title(
        f"SSW Onset Timing Error vs Lead Time  [{channel}]",
        fontsize=13,
        fontweight="bold",
    )
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_dir / f"ssw_timing_error_{channel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved %s", out)


def plot_rmse_and_spread(
    all_exp_metrics: dict[str, dict[int, dict[str, Any]]],
    channel: str,
    output_dir: Path,
) -> None:
    """Plot 3+4: RMSE and ensemble spread at SSW date vs lead time, grouped bar charts."""
    level_label = _channel_label(channel)
    exp_labels = list(all_exp_metrics.keys())
    all_leads = sorted(
        {lead for m in all_exp_metrics.values() for lead in m}, reverse=True
    )
    if not all_leads:
        return

    x = np.arange(len(all_leads))
    width = 0.8 / max(len(exp_labels), 1)

    fig, (ax_rmse, ax_spread) = plt.subplots(1, 2, figsize=(14, 5))

    for i, exp_label in enumerate(exp_labels):
        color = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        offset = (i - len(exp_labels) / 2 + 0.5) * width

        rmse_vals = [
            all_exp_metrics[exp_label].get(lead, {}).get("rmse_at_ssw")
            for lead in all_leads
        ]
        spread_vals = [
            all_exp_metrics[exp_label].get(lead, {}).get("spread_at_ssw")
            for lead in all_leads
        ]

        for ax, vals, title, ylabel in [
            (ax_rmse, rmse_vals, f"RMSE at SSW date — {level_label}", "RMSE (m/s)"),
            (
                ax_spread,
                spread_vals,
                f"Ensemble spread at SSW date — {level_label}",
                "Std dev (m/s)",
            ),
        ]:
            clean = [v if v is not None else 0.0 for v in vals]
            bars = ax.bar(
                x + offset, clean, width, color=color, alpha=0.8, label=exp_label
            )
            for bar, v in zip(bars, vals):
                if v is None:
                    bar.set_alpha(0.15)

    for ax, title, ylabel in [
        (ax_rmse, f"RMSE at SSW date — {level_label}", "RMSE (m/s)"),
        (ax_spread, f"Ensemble spread at SSW date — {level_label}", "Std dev (m/s)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([f"T–{ld}d" for ld in all_leads], rotation=30, ha="right")
        ax.set_xlabel("Lead time", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    ax_rmse.legend(fontsize=9)
    plt.tight_layout()

    out = output_dir / f"lead_time_skill_{channel}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--validations-config",
        type=Path,
        required=True,
        help="YAML config with 'lead_days' and 'group' fields per entry.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/lead_time"))
    parser.add_argument(
        "--channel", default="u_30", help="Channel to analyse (default: u_30 ≈ 11 hPa)"
    )
    parser.add_argument("--latitude", type=float, default=60.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_validations_config(args.validations_config)
    if not cfg:
        _logger.error("No entries in validations config.")
        return

    groups = group_by_experiment_and_lead(cfg)
    if not groups:
        _logger.error(
            "No entries with 'lead_days' found. "
            "Add 'lead_days: <int>' and 'group: <name>' to each entry."
        )
        return

    _logger.info("Found %d experiment group(s):", len(groups))
    for exp, leads in groups.items():
        total = sum(len(v) for v in leads.values())
        _logger.info(
            "  %s: leads=%s (%d entries)",
            exp,
            sorted(leads.keys(), reverse=True),
            total,
        )

    all_exp_metrics: dict[str, dict[int, dict[str, Any]]] = {}

    for exp_label, lead_map in groups.items():
        _logger.info("─" * 50)
        _logger.info("Experiment group: %s", exp_label)

        lead_runs: dict[int, list[dict[str, Any]]] = {}
        for lead_days, label_specs in lead_map.items():
            _logger.info("  Lead time: T–%dd (%d entries)", lead_days, len(label_specs))
            runs = load_experiment_group(label_specs, args.data_dir, args.channel)
            if runs:
                lead_runs[lead_days] = runs

        if not lead_runs:
            _logger.warning("No data loaded for %s, skipping.", exp_label)
            continue

        plot_timeseries_by_lead_time(
            lead_runs, exp_label, args.channel, args.output_dir
        )

        exp_metrics: dict[int, dict[str, Any]] = {}
        for lead, runs in lead_runs.items():
            m: dict[str, Any] = {"lead_days": lead, "n_samples": len(runs)}
            m["rmse_at_ssw"] = rmse_at_ssw(runs)
            m["spread_at_ssw"] = ensemble_spread_at_ssw(runs) if len(runs) > 1 else None

            mean_pred, _ = ensemble_mean_std(runs)
            proxy = {
                "predictions": mean_pred,
                "targets": runs[0]["targets"],
                "datetimes": runs[0]["datetimes"],
            }
            m["timing_error"] = ssw_timing_error(proxy)

            spread_str = (
                f"{m['spread_at_ssw']:.2f}" if m["spread_at_ssw"] is not None else "N/A"
            )
            timing_str = (
                f"{m['timing_error']:+.1f}d"
                if m["timing_error"] is not None
                else "not detected"
            )
            _logger.info(
                "  T–%dd: RMSE=%.2f m/s  spread=%s m/s  timing=%s",
                lead,
                m["rmse_at_ssw"],
                spread_str,
                timing_str,
            )
            exp_metrics[lead] = m

        all_exp_metrics[exp_label] = exp_metrics

    plot_timing_error(all_exp_metrics, args.channel, args.output_dir)
    plot_rmse_and_spread(all_exp_metrics, args.channel, args.output_dir)

    # Save metrics JSON
    def _serialise(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    metrics_file = args.output_dir / f"lead_time_metrics_{args.channel}.json"
    with open(metrics_file, "w") as f:
        json.dump(all_exp_metrics, f, indent=2, default=_serialise)
    _logger.info("Saved metrics: %s", metrics_file)


if __name__ == "__main__":
    main()
