#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Run-ID tracking manifest for SSW validation submissions.

Replaces ad-hoc parsing of ``validation_submissions.log`` with a structured
YAML manifest keyed by ``event / model_key / lead_key``, so submission
scripts can skip already-submitted combinations and downstream tooling
(status checks, result transfers, config generation) can consume a single
source of truth instead of re-parsing the raw log.

Manifest schema::

    <event>:
      <model_key>:
        <lead_key>:
          run_id: <generated inference run id>
          model_id: <base run id used for --from-run-id>
          model_label: <model_name>
          init_date: <YYYYMMDDhhmm>
          fsteps: <int>
          submitted_at: <"YYYY-MM-DD HH:MM:SS">
          status: submitted | done | failed | missing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_STATUS = "submitted"
VALID_STATUSES = {"submitted", "done", "failed", "missing"}


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the manifest YAML, returning an empty dict if it doesn't exist yet."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the manifest back to disk, sorted for stable diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=True, default_flow_style=False)


def get_entry(
    manifest: dict[str, Any], event: str, model: str, lead: str
) -> dict[str, Any] | None:
    """Return the manifest entry for (event, model, lead), or None if absent."""
    return manifest.get(event, {}).get(model, {}).get(lead)


def entry_exists(manifest: dict[str, Any], event: str, model: str, lead: str) -> bool:
    """True if a non-empty entry with a run_id already exists for this key."""
    entry = get_entry(manifest, event, model, lead)
    return bool(entry and entry.get("run_id"))


def upsert_entry(
    manifest: dict[str, Any],
    event: str,
    model: str,
    lead: str,
    *,
    run_id: str,
    model_id: str,
    model_label: str,
    init_date: str,
    fsteps: int,
    submitted_at: str,
    status: str = DEFAULT_STATUS,
) -> dict[str, Any]:
    """Insert or overwrite the (event, model, lead) entry in-place and return it."""
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    entry = {
        "run_id": run_id,
        "model_id": model_id,
        "model_label": model_label,
        "init_date": init_date,
        "fsteps": int(fsteps),
        "submitted_at": submitted_at,
        "status": status,
    }
    manifest.setdefault(event, {}).setdefault(model, {})[lead] = entry
    return entry


def set_status(
    manifest: dict[str, Any], event: str, model: str, lead: str, status: str
) -> bool:
    """Update just the status field of an existing entry. Returns False if absent."""
    entry = get_entry(manifest, event, model, lead)
    if entry is None:
        return False
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    entry["status"] = status
    return True


def iter_entries(manifest: dict[str, Any]):
    """Yield (event, model, lead, entry) tuples for every entry in the manifest."""
    for event, models in manifest.items():
        for model, leads in models.items():
            for lead, entry in leads.items():
                yield event, model, lead, entry
