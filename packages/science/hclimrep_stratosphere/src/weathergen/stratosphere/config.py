# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Config-file loading utilities for stratospheric analysis scripts.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_validations_config(config_path: Path) -> dict[str, dict[str, Any]]:
    """
    Load a validations YAML config file.

    Expected format::

        validations:
          my_label:
            id: <validation_id>     # zarr directory name inside data_dir
            sample: 0               # local sample index within the rank's store (default 0)
            rank: 0                 # SLURM rank whose store holds this sample (default 0)
            color: "#e41a1c"        # optional plot colour
            group: experiment_A     # optional experiment group label
            lead_days: 10           # optional lead time in days
            event: feb2018          # optional real-world SSW event this sample verifies
                                     # against; used to split a config's samples into
                                     # separate per-event plots (falls back to ``group``,
                                     # then a single "default" bucket, when omitted)
            event_date: 2018-02-12  # optional onset date for that event (YYYY-MM-DD);
                                     # used as the SSW reference line/origin for all
                                     # samples sharing the same ``event``

    ``rank`` matters because a single inference run submitted with the
    ``date_ranges`` ("init date list") feature spreads its requested init
    dates across multiple SLURM ranks (each rank writes its own zarr store),
    so ``sample`` alone (local to one rank's store) is not enough to locate
    a forecast — see ``weathergen.stratosphere.io.zarr_store_path``.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict mapping label → spec dict with keys:
        ``id``, ``sample``, ``rank``, ``color``, ``group``, ``lead_days``,
        ``event_type``, ``event``, ``event_date``.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    entries = raw.get("validations") or {}
    result: dict[str, dict[str, Any]] = {}
    for label, spec in entries.items():
        if isinstance(spec, int):
            # Shorthand: label: sample_index  (id inferred from label)
            result[label] = {
                "id": label,
                "sample": spec,
                "rank": 0,
                "color": None,
                "group": None,
                "lead_days": None,
                "event_type": None,
                "event": None,
                "event_date": None,
            }
        else:
            result[label] = {
                "id": str(spec["id"]),
                "sample": int(spec.get("sample", 0)),
                "rank": int(spec.get("rank", 0)),
                "color": spec.get("color"),
                "group": spec.get("group"),
                "lead_days": spec.get("lead_days"),
                "event_type": spec.get("event_type"),
                "event": spec.get("event"),
                "event_date": spec.get("event_date"),
            }
    return result


def group_validations_by_event(
    cfg: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Partition a validations config into per-event sub-configs.

    A single inference run submitted with ``date_ranges`` now bundles many
    init-date samples in one config, but those samples can verify against
    several distinct real-world SSW events (e.g. Feb 2016, Feb 2018,
    Dec 2018/Jan 2019).  Overlaying all of them on one plot is unreadable,
    so analysis scripts should group entries by event and produce one plot
    (or figure set) per event instead.

    Each entry's ``event`` field determines its bucket.  Entries without an
    explicit ``event`` fall back to ``group``, then to a single
    ``"default"`` bucket, so existing single-event configs keep producing
    exactly one combined plot as before.

    Args:
        cfg: Dict as returned by :func:`load_validations_config`.

    Returns:
        Dict mapping event name → sub-dict of ``{label: spec}``, in
        insertion order of first appearance.
    """
    events: dict[str, dict[str, dict[str, Any]]] = {}
    for label, spec in cfg.items():
        event = spec.get("event") or spec.get("group") or "default"
        events.setdefault(str(event), {})[label] = spec
    return events


def event_reference_date(
    event_cfg: dict[str, dict[str, Any]], default: datetime
) -> datetime:
    """
    Resolve the SSW onset reference date for one event bucket.

    Uses the first ``event_date`` found among the bucket's entries
    (accepting either a ``datetime``/``date`` from YAML's auto-parsing or a
    ``"YYYY-MM-DD"`` string), falling back to *default* — typically the
    script's ``--ssw-date`` CLI argument — when no entry specifies one.

    Args:
        event_cfg: One event's sub-dict, as produced by
            :func:`group_validations_by_event`.
        default: Reference date to use when no entry sets ``event_date``.

    Returns:
        The resolved reference ``datetime``.
    """
    for spec in event_cfg.values():
        raw = spec.get("event_date")
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return raw
        if hasattr(raw, "year") and hasattr(raw, "month") and hasattr(raw, "day"):
            # yaml.safe_load auto-parses unquoted YYYY-MM-DD as datetime.date
            return datetime(raw.year, raw.month, raw.day)
        return datetime.strptime(str(raw), "%Y-%m-%d")
    return default
