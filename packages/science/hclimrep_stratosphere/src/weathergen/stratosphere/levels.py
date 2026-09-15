# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Model level ↔ pressure conversion utilities.

Loads ECMWF L137 half-level pressures from the bundled ``level_listings.csv``
(shipped with this package). The CSV has columns::

    n,a [Pa],b,ph [hPa],pf [hPa]

where *n* is the model level number and *ph [hPa]* is the half-level pressure.

Key levels referenced in the analysis code:
    L29  ≈ 10 hPa   (upper stratosphere – SSW critical level)
    L30  ≈ 11 hPa
    L41  ≈ 30 hPa
    L51  ≈ 62 hPa
    L55  ≈ 75 hPa
    L137 ≈ 1013 hPa (surface)
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

_CSV_PATH = Path(__file__).parent / "data" / "level_listings.csv"


@functools.lru_cache(maxsize=1)
def load_level_pressures() -> dict[int, float]:
    """
    Load all model level → half-level pressure (hPa) mappings.

    Cached after first call.

    Returns:
        Dict mapping model level number to pressure in hPa.
    """
    pressures: dict[int, float] = {}
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pressures[int(row["n"])] = float(row["ph [hPa]"])
    return pressures


@functools.lru_cache(maxsize=1)
def load_full_level_pressures() -> dict[int, float]:
    """
    Load all model level → full-level pressure (hPa) mappings.

    Full-level pressures (``pf [hPa]`` column) are the pressure at the centre
    of each model layer — i.e. where u, v, T fields are defined.  Use these
    for TEM and other computations that need the pressure at the field level.

    Level 0 and any row with a non-numeric ``pf [hPa]`` value are skipped.

    Cached after first call.

    Returns:
        Dict mapping model level number to full-level pressure in hPa.
    """
    pressures: dict[int, float] = {}
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pressures[int(row["n"])] = float(row["pf [hPa]"])
            except (ValueError, KeyError):
                pass  # skip level 0 (value is "-") and any malformed rows
    return pressures


def get_full_level_pressure(level: int) -> float:
    """
    Return full-level pressure in hPa for a given model level.

    Raises:
        ValueError: If the level is not in the CSV or has no valid pf value.
    """
    pressures = load_full_level_pressures()
    if level not in pressures:
        raise ValueError(
            f"Model level {level} not found in level_listings.csv (pf column). "
            f"Valid range: {min(pressures)}–{max(pressures)}."
        )
    return pressures[level]


def build_full_level_map(variable: str, channels: list[str]) -> dict[str, float]:
    """
    Like :func:`build_level_map` but uses full-level pressure (``pf [hPa]``).

    Use this when you need the pressure at which fields are defined (TEM, EP flux,
    vertical derivatives), rather than the half-level interface pressure.
    """
    pressures = load_full_level_pressures()
    result: dict[str, float] = {}
    prefix = f"{variable}_"
    for ch in channels:
        if not ch.startswith(prefix):
            continue
        parts = ch.split("_", 1)
        if len(parts) != 2:
            continue
        try:
            level = int(parts[1])
        except ValueError:
            continue
        if level >= 150:
            # ERA5pl channel: integer suffix is already pressure in hPa
            result[ch] = float(level)
        elif level in pressures:
            result[ch] = pressures[level]
    return result


def get_level_pressure(level: int) -> float:
    """
    Return half-level pressure in hPa for a given model level.

    Raises:
        ValueError: If the level is not in the CSV.
    """
    pressures = load_level_pressures()
    if level not in pressures:
        raise ValueError(
            f"Model level {level} not found in level_listings.csv. "
            f"Valid range: {min(pressures)}–{max(pressures)}."
        )
    return pressures[level]


def channel_pressure(channel_name: str) -> float | None:
    """
    Infer pressure in hPa from a channel name like ``'u_30'`` or ``'t_55'``.

    For ERA5pl channels (e.g. ``'u_50'``) the integer suffix *is* the pressure.
    For ERA5ml channels (e.g. ``'u_30'``) the integer suffix is a model level
    and is looked up in the CSV.  Returns ``None`` when undetermined.
    """
    if "_" not in channel_name:
        return None
    parts = channel_name.split("_")
    if len(parts) != 2:
        return None
    try:
        level = int(parts[1])
    except ValueError:
        return None
    # Heuristic: levels ≥ 150 are not valid ECMWF L137 numbers → treat as hPa
    if level >= 150:
        return float(level)
    try:
        return get_level_pressure(level)
    except ValueError:
        return float(level)


def build_level_map(variable: str, channels: list[str]) -> dict[str, float]:
    """
    Build a dict mapping channel name → pressure (hPa) for all channels
    of the given variable (e.g. ``'u'``, ``'t'``, ``'v'``).

    Channels whose pressure cannot be determined are omitted.
    """
    result: dict[str, float] = {}
    prefix = f"{variable}_"
    for ch in channels:
        if ch.startswith(prefix):
            p = channel_pressure(ch)
            if p is not None:
                result[ch] = p
    return result
