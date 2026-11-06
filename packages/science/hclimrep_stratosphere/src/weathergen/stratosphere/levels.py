# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Channel name ↔ pressure resolution for the stratospheric analysis.

WeatherGenerator channel names have the form ``<variable>_<int>`` where the
integer is **either** an ECMWF L137 model-level index (ERA5ml-type streams,
``u_29`` ≈ 10 hPa) **or** a pressure in hPa (ERA5pl / S2Spredictors-type
streams, ``u_10`` = 10 hPa).  The two cannot be told apart from the name
alone (``u_50`` is 56 hPa on model levels but 50 hPa on pressure levels), so
the level type must come from the *stream*:

    level_type = stream_level_type("S2Spredictors")      # 'pressure'
    p = channel_pressure("u_10", level_type)             # 10.0

Stream → level-type mapping lives in :data:`STREAM_LEVEL_TYPES`; unknown
streams fall back to :func:`guess_level_type` which inspects the set of level
integers present in the stream (pressure-level streams contain values like
1000/925/850 that are not valid L137 indices).

The L137 table is loaded from the bundled ``level_listings.csv`` with columns
``n,a [Pa],b,ph [hPa],pf [hPa],...``: *ph* is the half-level (interface)
pressure and *pf* the full-level pressure at which u, v, T are defined.
Historical note: the analysis code used *ph* for channel labelling
(L29 → 10.4 hPa); :func:`channel_pressure` now returns the full-level value
(L29 → 9.8 hPa) because that is where the field lives.  Standard key levels:

    L19 ≈ 2.5 hPa   L29 ≈ 10 hPa   L40 ≈ 28 hPa   L48 ≈ 50 hPa
    L55 ≈ 75 hPa    L61 ≈ 104 hPa  L137 ≈ 1013 hPa (surface)
"""

from __future__ import annotations

import contextlib
import csv
import functools
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

LevelType = Literal["model", "pressure"]

_CSV_PATH = Path(__file__).parent / "data" / "level_listings.csv"

# Stream name → level type.  Extend here when a new stream is added to the
# training configs (config/strato_s2s/streams/*).  Override at runtime with
# the env var ``WG_STRATO_LEVEL_TYPES="MyStream=pressure,Other=model"``.
STREAM_LEVEL_TYPES: dict[str, LevelType] = {
    "ERA5ml": "model",
    "ERA5ML": "model",
    "TEM": "model",
    "ERA5pl": "pressure",
    "ERA5PL": "pressure",
    "ERA5": "pressure",
    "S2Spredictors": "pressure",
    "S2SPredictors": "pressure",
}

# Surface / single-level channels that carry no vertical level.
SURFACE_CHANNELS: frozenset[str] = frozenset(
    {
        "10u",
        "10v",
        "2t",
        "2d",
        "msl",
        "sp",
        "skt",
        "sst",
        "sd",
        "ci",
        "ttr",
        "tcw",
        "tp",
        "cp",
        "ssrd",
        "strd",
        "stl1",
        "stl2",
        "stl3",
        "swvl1",
        "swvl2",
        "swvl3",
        "z",
        "lsm",
        "slt",
        "ro",
        "sf",
        "tcc",
    }
)

# Stratospheric pressure levels (hPa) that matter for the analysis, in the order
# the S2S / strato_s2s configs provide them.  Used as plotting defaults.
STRATO_PRESSURE_LEVELS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0, 50.0, 70.0, 100.0)


def _env_overrides() -> dict[str, LevelType]:
    raw = os.environ.get("WG_STRATO_LEVEL_TYPES", "")
    out: dict[str, LevelType] = {}
    for item in raw.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            v = v.strip().lower()
            if v in ("model", "pressure"):
                out[k.strip()] = v  # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# L137 table
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_level_pressures() -> dict[int, float]:
    """Model level → half-level (interface) pressure in hPa (``ph`` column)."""
    pressures: dict[int, float] = {}
    with open(_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            pressures[int(row["n"])] = float(row["ph [hPa]"])
    return pressures


@functools.lru_cache(maxsize=1)
def load_full_level_pressures() -> dict[int, float]:
    """Model level → full-level pressure in hPa (``pf`` column; where u, v, T live)."""
    pressures: dict[int, float] = {}
    with open(_CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            with contextlib.suppress(ValueError, KeyError):  # level 0 has '-'
                pressures[int(row["n"])] = float(row["pf [hPa]"])
    return pressures


def get_level_pressure(level: int) -> float:
    """Half-level pressure (hPa) of model level *level*."""
    pressures = load_level_pressures()
    if level not in pressures:
        raise ValueError(
            f"Model level {level} not in level_listings.csv "
            f"(valid {min(pressures)}–{max(pressures)})."
        )
    return pressures[level]


def get_full_level_pressure(level: int) -> float:
    """Full-level pressure (hPa) of model level *level*."""
    pressures = load_full_level_pressures()
    if level not in pressures:
        raise ValueError(
            f"Model level {level} not in level_listings.csv pf column "
            f"(valid {min(pressures)}–{max(pressures)})."
        )
    return pressures[level]


def nearest_model_level(pressure_hpa: float) -> int:
    """Model level whose full-level pressure is closest to *pressure_hpa*."""
    pressures = load_full_level_pressures()
    return min(pressures, key=lambda lv: abs(pressures[lv] - pressure_hpa))


# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------


def parse_channel(channel: str) -> tuple[str, int | None]:
    """
    Split ``'u_29'`` → ``('u', 29)``; surface channels → ``('10u', None)``.

    Multi-underscore names such as ``ep_flux_div_61`` or ``F_p_48`` are
    split at the *last* underscore when the trailing token is an integer.
    """
    if "_" not in channel:
        return channel, None
    var, _, suffix = channel.rpartition("_")
    try:
        return var, int(suffix)
    except ValueError:
        return channel, None


def stream_level_type(stream: str, channels: Iterable[str] | None = None) -> LevelType:
    """
    Level type of *stream*: explicit table / env override first, otherwise
    :func:`guess_level_type` on *channels*.  Defaults to ``'pressure'`` when
    nothing else is known (the current training setup is pressure-level only).
    """
    overrides = _env_overrides()
    if stream in overrides:
        return overrides[stream]
    if stream in STREAM_LEVEL_TYPES:
        return STREAM_LEVEL_TYPES[stream]
    if channels is not None:
        return guess_level_type(channels)
    return "pressure"


def guess_level_type(channels: Iterable[str]) -> LevelType:
    """
    Heuristic for streams not in :data:`STREAM_LEVEL_TYPES`.

    Any level integer > 137 is impossible on L137 → pressure levels.  A set
    containing only values ≤ 137 whose sorted sequence looks like
    consecutive model indices (step 1–3) → model levels.  Otherwise pressure.
    """
    levels = sorted({lv for _, lv in map(parse_channel, channels) if lv is not None})
    if not levels:
        return "pressure"
    if max(levels) > 137:
        return "pressure"
    # Standard pressure levels are all in this set; model-level selections
    # typically include indices that are not.
    std_pl = {
        1,
        2,
        3,
        5,
        7,
        10,
        20,
        30,
        50,
        70,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        775,
        800,
        825,
        850,
        875,
        900,
        925,
        950,
        975,
        1000,
    }
    if all(lv in std_pl for lv in levels):
        return "pressure"
    return "model"


def channel_pressure(channel: str, level_type: LevelType | None = None) -> float | None:
    """
    Pressure (hPa) at which *channel* is defined, or ``None`` for surface
    channels / unparsable names.

    Args:
        channel: e.g. ``'u_29'``, ``'t_10'``, ``'ep_flux_div_61'``.
        level_type: ``'model'`` → L137 full-level lookup; ``'pressure'`` → the
            suffix is the pressure.  ``None`` (legacy) applies the old
            ``>= 150 → hPa`` heuristic and should be avoided in new code.
    """
    _, lv = parse_channel(channel)
    if lv is None:
        return None
    if level_type == "pressure":
        return float(lv)
    if level_type == "model":
        try:
            return get_full_level_pressure(lv)
        except ValueError:
            return None
    # legacy heuristic
    if lv >= 150:
        return float(lv)
    try:
        return get_full_level_pressure(lv)
    except ValueError:
        return float(lv)


def build_level_map(
    variable: str, channels: Iterable[str], level_type: LevelType | None = None
) -> dict[str, float]:
    """``{channel: pressure_hPa}`` for all channels of *variable* (``'u'``, ``'t'``...)."""
    result: dict[str, float] = {}
    for ch in channels:
        var, lv = parse_channel(ch)
        if var != variable or lv is None:
            continue
        p = channel_pressure(ch, level_type)
        if p is not None:
            result[ch] = p
    return result


def build_full_level_map(
    variable: str, channels: Iterable[str], level_type: LevelType | None = None
) -> dict[str, float]:
    """Alias of :func:`build_level_map` (full-level pressures are now the default)."""
    return build_level_map(variable, channels, level_type)


def nearest_channel(
    variable: str,
    pressure_hpa: float,
    channels: Iterable[str],
    level_type: LevelType | None = None,
    tolerance_frac: float = 0.25,
) -> str | None:
    """
    Channel of *variable* closest (in log-pressure) to *pressure_hpa*, or
    ``None`` if the closest one is off by more than *tolerance_frac* in
    relative terms (``|p - p0| / p0``).
    """
    import math

    lm = build_level_map(variable, channels, level_type)
    if not lm:
        return None
    ch = min(lm, key=lambda c: abs(math.log(lm[c]) - math.log(pressure_hpa)))
    if abs(lm[ch] - pressure_hpa) / pressure_hpa > tolerance_frac:
        return None
    return ch


def format_pressure(p_hpa: float) -> str:
    """``'10 hPa'`` / ``'2.5 hPa'`` for labels."""
    return f"{p_hpa:g} hPa"
