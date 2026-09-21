#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
manage_runs.py — CLI for the SSW run-ID tracking manifest.

Actions:
  list           Print manifest entries, optionally filtered by event/model/lead/status.
  check          Cross-reference manifest entries against results/<run_id> on disk,
                 flag missing directories (replaces silent-skip in transfer_results.sh).
  export-config  Auto-generate a config/evaluate/ssw_<event>.yml validations config
                 from manifest entries for a given event (fills id/group/lead_days/color).
  upsert         Insert/update a single manifest entry (used by submit_ssw_validation.sh).
  exists         Exit 0 if a non-empty entry exists for (event, model, lead), else 1.
                 (used by submit_ssw_validation.sh to implement --force-aware dedup)

Usage:
  python3 manage_runs.py list [--event E] [--model M] [--lead L] [--status S]
  python3 manage_runs.py check [--event E] [--results-dir DIR]
  python3 manage_runs.py export-config --event E [--output PATH]
  python3 manage_runs.py upsert --event E --model M --lead L --run-id ID \\
      --model-id ID --model-label LABEL --init-date DATE --fsteps N [--status S]
  python3 manage_runs.py exists --event E --model M --lead L
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)
# parents[1] == packages/science/hclimrep_stratosphere (package root, has src/)

from weathergen.stratosphere.runs_manifest import (  # noqa: E402
    get_entry,
    iter_entries,
    load_manifest,
    save_manifest,
    upsert_entry,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WG_ROOT = _SCRIPT_DIR.parents[3]
_DEFAULT_MANIFEST = _WG_ROOT / "config" / "evaluate" / "runs_manifest.yml"
_DEFAULT_RESULTS_DIR = Path("/e/scratch/weatherai/shared_work/results")

# Blue -> red palette (matches existing RdYlBu-style hand-curated configs)
_PALETTE = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]


def _palette_color(lead_days: int, lead_max: int) -> str:
    if lead_max <= 0:
        return _PALETTE[0]
    frac = min(max(lead_days / lead_max, 0.0), 1.0)
    idx = int(round(frac * (len(_PALETTE) - 1)))
    return _PALETTE[idx]


def cmd_list(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    n = 0
    for event, model, lead, entry in iter_entries(manifest):
        if args.event and event != args.event:
            continue
        if args.model and model != args.model:
            continue
        if args.lead and lead != args.lead:
            continue
        if args.status and entry.get("status") != args.status:
            continue
        print(
            f"{event:12s} {model:12s} {lead:6s} "
            f"run_id={entry.get('run_id', ''):12s} "
            f"model_id={entry.get('model_id', ''):12s} "
            f"status={entry.get('status', ''):10s} "
            f"init_date={entry.get('init_date', '')} "
            f"fsteps={entry.get('fsteps', '')}"
        )
        n += 1
    print(f"\n{n} entries.")


def cmd_check(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    missing = []
    for event, model, lead, entry in iter_entries(manifest):
        if args.event and event != args.event:
            continue
        run_id = entry.get("run_id")
        if not run_id:
            missing.append((event, model, lead, "no run_id in manifest"))
            continue
        result_path = args.results_dir / run_id
        if not result_path.is_dir():
            missing.append((event, model, lead, f"missing dir: {result_path}"))

    if not missing:
        print("All manifest entries have a matching results directory.")
        return

    print(f"{len(missing)} missing/inconsistent entries:")
    for event, model, lead, reason in missing:
        print(f"  {event:12s} {model:12s} {lead:6s}  {reason}")
    sys.exit(1)


def cmd_export_config(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    event_entries = manifest.get(args.event, {})
    if not event_entries:
        print(f"No manifest entries found for event '{args.event}'.", file=sys.stderr)
        sys.exit(1)

    leads_all = []
    for model, leads in event_entries.items():
        for lead, entry in leads.items():
            days = int(lead.rstrip("d").lstrip("t"))
            leads_all.append(days)
    lead_max = max(leads_all) if leads_all else 0

    validations: dict[str, dict] = {}
    for model, leads in event_entries.items():
        for lead, entry in sorted(leads.items()):
            run_id = entry.get("run_id")
            if not run_id:
                continue
            days = int(lead.rstrip("d").lstrip("t"))
            label = f"{model}_{lead}"
            validations[label] = {
                "id": run_id,
                "sample": 0,
                "group": model,
                "lead_days": days,
                "color": _palette_color(days, lead_max),
            }

    output_path = args.output or (
        _WG_ROOT / "config" / "evaluate" / f"ssw_{args.event}.yml"
    )
    doc = {"validations": validations}
    with open(output_path, "w") as f:
        f.write(f"# Auto-generated from runs_manifest.yml for event '{args.event}'.\n")
        f.write("# Regenerate with: manage_runs.py export-config --event "
                f"{args.event}\n\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(validations)} entries to {output_path}")


def cmd_upsert(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    submitted_at = args.submitted_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    upsert_entry(
        manifest,
        args.event,
        args.model,
        args.lead,
        run_id=args.run_id,
        model_id=args.model_id,
        model_label=args.model_label,
        init_date=args.init_date,
        fsteps=args.fsteps,
        submitted_at=submitted_at,
        status=args.status,
    )
    save_manifest(manifest, args.manifest)


def cmd_exists(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    entry = get_entry(manifest, args.event, args.model, args.lead)
    sys.exit(0 if entry and entry.get("run_id") else 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", type=Path, default=_DEFAULT_MANIFEST, help="Path to runs_manifest.yml"
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="List manifest entries.")
    p_list.add_argument("--event")
    p_list.add_argument("--model")
    p_list.add_argument("--lead")
    p_list.add_argument("--status")
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser("check", help="Verify results dirs exist for manifest entries.")
    p_check.add_argument("--event")
    p_check.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    p_check.set_defaults(func=cmd_check)

    p_export = sub.add_parser("export-config", help="Generate ssw_<event>.yml from manifest.")
    p_export.add_argument("--event", required=True)
    p_export.add_argument("--output", type=Path, default=None)
    p_export.set_defaults(func=cmd_export_config)

    p_upsert = sub.add_parser("upsert", help="Insert/update one manifest entry.")
    p_upsert.add_argument("--event", required=True)
    p_upsert.add_argument("--model", required=True)
    p_upsert.add_argument("--lead", required=True)
    p_upsert.add_argument("--run-id", required=True)
    p_upsert.add_argument("--model-id", required=True)
    p_upsert.add_argument("--model-label", required=True)
    p_upsert.add_argument("--init-date", required=True)
    p_upsert.add_argument("--fsteps", type=int, required=True)
    p_upsert.add_argument("--status", default="submitted")
    p_upsert.add_argument("--submitted-at", default=None)
    p_upsert.set_defaults(func=cmd_upsert)

    p_exists = sub.add_parser("exists", help="Exit 0 if entry exists, else 1.")
    p_exists.add_argument("--event", required=True)
    p_exists.add_argument("--model", required=True)
    p_exists.add_argument("--lead", required=True)
    p_exists.set_defaults(func=cmd_exists)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
