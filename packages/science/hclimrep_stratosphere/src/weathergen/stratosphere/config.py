# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Config-file loading utilities for stratospheric analysis scripts.
"""

from __future__ import annotations

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
            sample: 0               # ensemble member index (default 0)
            color: "#e41a1c"        # optional plot colour
            group: experiment_A     # optional experiment group label
            lead_days: 10           # optional lead time in days

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dict mapping label → spec dict with keys:
        ``id``, ``sample``, ``color``, ``group``, ``lead_days``, ``event_type``.
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
                "color": None,
                "group": None,
                "lead_days": None,
                "event_type": None,
            }
        else:
            result[label] = {
                "id": str(spec["id"]),
                "sample": int(spec.get("sample", 0)),
                "color": spec.get("color"),
                "group": spec.get("group"),
                "lead_days": spec.get("lead_days"),
                "event_type": spec.get("event_type"),
            }
    return result
