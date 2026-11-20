# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
I/O utilities for stratospheric analysis.

Thin adapter over ``weathergen.common.io.zarrio_reader`` that exposes the
data access patterns used by the analysis scripts:

    with open_validation(path) as zio:
        # Level-aware access (preferred).  A forecast store can hold several
        # streams (e.g. ERA5pl 50–1000 hPa + S2Spredictors 1–70 hPa); the
        # column index merges them into one pressure axis per variable.
        col = discover_column(zio, "u")             # {pressure_hPa: LevelRef}
        ref = col.nearest(10.0)                     # LevelRef(stream, channel, p)
        pred, tgt, times = load_level(zio, ref, step)   # (n_pts, n_ens) each
        coords = get_coords(zio, ref.stream)

        # Low-level per-stream access (unchanged).
        channels = get_channels(zio, "ERA5pl")
        pred, tgt, times = load_step(zio, "ERA5pl", step)
            # pred / tgt: (n_pts, n_channels, n_ens)  numpy arrays
            # times:      (n_pts,)                    numpy datetime64[ns]

Coordinate helpers (find_latitude_indices etc.) work on the ``coords``
array and are independent of the zarr backend.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from weathergen.common.io import ZarrIO, zarrio_reader
from weathergen.stratosphere.levels import (
    LevelType,
    build_level_map,
    stream_level_type,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-export so callers only need to import from this module
# ---------------------------------------------------------------------------
open_validation = zarrio_reader  # context-manager: open_validation(path) -> ZarrIO


# ---------------------------------------------------------------------------
# Zarr store path resolution
# ---------------------------------------------------------------------------


def zarr_store_name(rank: int = 0, mini_epoch: int = 0) -> str:
    """
    Return the validation zarr store filename written by the trainer for a
    given SLURM rank and mini_epoch (see ``config.get_path_results``).

    Each rank writes its own store, and with the ``date_ranges`` ("init date
    list") inference feature, a single run can spread its requested init
    dates across multiple ranks — so ``rank`` must be tracked alongside
    ``sample`` (which is local to that rank's store) to locate a specific
    forecast on disk.
    """
    return f"validation_chkpt{mini_epoch:05d}_rank{rank:04d}.zip"


def zarr_store_path(data_dir: Path, run_id: str, rank: int = 0, mini_epoch: int = 0) -> Path:
    """Resolve the on-disk zarr store path for a given run, rank, and mini_epoch."""
    return Path(data_dir) / run_id / zarr_store_name(rank, mini_epoch)


# ---------------------------------------------------------------------------
# ZarrIO accessors
# ---------------------------------------------------------------------------


def get_stream(zio: ZarrIO) -> str:
    """
    Return the primary *atmospheric* stream name.

    .. deprecated::
        Forecast stores of the strato_s2s arms spread the atmospheric column
        over two streams (``ERA5pl`` + ``S2Spredictors``); use
        :func:`discover_column` / :func:`load_level` instead.  Kept for the
        single-stream ERA5ml runs.
    """
    warnings.warn(
        "get_stream() picks one stream; use discover_column() for multi-stream stores",
        DeprecationWarning,
        stacklevel=2,
    )
    streams = zio.streams
    for preferred in ("ERA5ml", "ERA5pl", "ERA5", "S2Spredictors"):
        if preferred in streams:
            return preferred
    return streams[0]


def get_streams(zio: ZarrIO) -> list[str]:
    """All stream names in the store (order as written by the trainer)."""
    return list(zio.streams)


def get_forecast_steps(zio: ZarrIO, skip_source_step: bool = True) -> list[int]:
    """
    Return sorted forecast steps.

    Args:
        skip_source_step: If True (default) exclude step 0 which is the
            source/analysis field, not a forecast.
    """
    # zio.forecast_steps returns zarr group keys as strings; convert to int first
    steps = sorted(int(f) for f in zio.forecast_steps)
    if skip_source_step:
        steps = [s for s in steps if s > 0]
    return steps


def get_channels(zio: ZarrIO, stream: str, sample: int = 0) -> list[str]:
    """
    Return list of channel names for *stream* (empty list if the stream wrote
    no targets for this sample, which happens for sparse/masked streams).
    """
    steps = get_forecast_steps(zio)
    item = zio.get_data(sample, stream, steps[0])
    try:
        return list(item.prediction.channels)
    except (AttributeError, KeyError):
        return []


def get_coords(zio: ZarrIO, stream: str, sample: int = 0) -> NDArray[np.float32]:
    """
    Return spatial coordinates as ``(n_pts, 2)`` array with columns [lat, lon].
    """
    steps = get_forecast_steps(zio)
    item = zio.get_data(sample, stream, steps[0])
    return np.asarray(item.prediction.coords)


def load_step(
    zio: ZarrIO,
    stream: str,
    step: int,
    sample: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray]:
    """
    Load prediction and target data for a single forecast step.

    Returns:
        pred:  (n_pts, n_channels, n_ens)  float32 numpy array
        tgt:   (n_pts, n_channels, n_ens)  float32 numpy array
        times: (n_pts,)                    numpy datetime64[ns] array
    """
    item = zio.get_data(sample, stream, step)
    # see load_level: target lacks the ensemble axis in real stores
    pred = np.atleast_3d(np.asarray(item.prediction.data))
    tgt = np.atleast_3d(np.asarray(item.target.data))
    times = np.asarray(item.prediction.times)
    return pred, tgt, times


def load_source(
    zio: ZarrIO,
    stream: str,
    sample: int = 0,
) -> tuple[NDArray[np.float32], NDArray]:
    """
    Load source (analysis / initial condition) field from step 0.

    Returns:
        src:   (n_pts, n_channels, n_ens)  float32 numpy array
        times: (n_pts,)                    numpy datetime64[ns] array
    """
    item = zio.get_data(sample, stream, 0)
    # see load_level: source lacks the ensemble axis in real stores
    src = np.atleast_3d(np.asarray(item.source.data))
    times = np.asarray(item.source.times)
    return src, times


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def find_latitude_indices(
    coords: NDArray[np.float32],
    target_lat: float,
    tolerance: float = 2.5,
) -> NDArray[np.intp]:
    """Return indices of grid points within *tolerance* degrees of *target_lat*."""
    lats = coords[:, 0]
    indices = np.where(np.abs(lats - target_lat) <= tolerance)[0]
    if len(indices) == 0:
        raise ValueError(f"No grid points found near {target_lat}°N (tolerance={tolerance}°)")
    return indices


def find_polar_cap_indices(
    coords: NDArray[np.float32],
    min_lat: float = 60.0,
    hemisphere: str = "north",
) -> NDArray[np.intp]:
    """Return indices of grid points poleward of *min_lat*."""
    lats = coords[:, 0]
    if hemisphere == "north":
        mask = lats >= min_lat
    elif hemisphere == "south":
        mask = lats <= -min_lat
    else:
        raise ValueError(f"Invalid hemisphere: {hemisphere!r}. Use 'north' or 'south'.")
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"No grid points found poleward of {min_lat}° ({hemisphere})")
    return indices


def find_latitude_band_indices(
    coords: NDArray[np.float32],
    lat_min: float,
    lat_max: float,
) -> NDArray[np.intp]:
    """Return indices of grid points within the latitude band [lat_min, lat_max]."""
    lats = coords[:, 0]
    indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    if len(indices) == 0:
        raise ValueError(f"No grid points found in band {lat_min}°–{lat_max}°")
    return indices


def get_area_weights(
    coords: NDArray[np.float32],
    indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """
    Compute area weights for a subset of grid points.

    For O<n> reduced Gaussian grids (e.g. O96), uses the correct formula accounting
    for varying point density per latitude band. For other grids, falls back to
    cosine-latitude weighting.

    Returns normalized weights summing to 1.
    """
    weights = get_o_grid_weights(coords, indices)
    return weights / weights.sum()


def get_o_grid_weights(
    coords: NDArray[np.float32],
    indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """
    Compute (unnormalized) per-point area weights, accounting for the O<n>
    reduced Gaussian grid's varying point density per latitude band if
    *coords* matches a known O<n> grid size; otherwise falls back to plain
    cosine-latitude weighting.

    Args:
        coords:  ``(n_pts, 2)`` [lat, lon] array for the *full* grid (used to
                 detect the O<n> grid size).
        indices: Optional subset of point indices to return weights for
                 (default: all points).

    Returns:
        Unnormalized weights, same length as *indices* (or *coords* if
        *indices* is None).
    """
    # Map total grid point count to n for O<n> reduced Gaussian grids (formula: 4*n*(n+9))
    o_grid_n = {4 * n * (n + 9): n for n in [96, 160, 256, 320, 640, 1280]}

    total_points = coords.shape[0]
    sub_coords = coords if indices is None else coords[indices]
    lats = sub_coords[:, 0]

    if total_points in o_grid_n:
        n = o_grid_n[total_points]
        # O<n> reduced Gaussian: each latitude band has 20+4*(i-1) points
        # i ranges from 1 (pole) to n (equator)
        n_lat_max = 20 + 4 * (n - 1)
        n_lat = n_lat_max - np.abs(lats) * (n_lat_max - 20) / 90.0
        return np.clip(np.cos(np.deg2rad(lats.astype(np.float64))), 0.0, None) / n_lat
    # float32 cos(90°) is -4e-8; clip so sqrt-weights downstream stay finite
    return np.clip(np.cos(np.deg2rad(lats.astype(np.float64))), 0.0, None)


def convert_times_to_datetime(times: NDArray) -> list:
    """Convert numpy datetime64 array to Python datetime objects."""
    return [t.astype("datetime64[s]").astype(object) for t in times]


# ---------------------------------------------------------------------------
# Level-aware, multi-stream column access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelRef:
    """One vertical level of one variable, located in a specific stream."""

    variable: str
    stream: str
    channel: str
    pressure: float  # hPa (full-level pressure for model-level streams)
    level_type: LevelType

    @property
    def label(self) -> str:
        return f"{self.variable} {self.pressure:g} hPa"


@dataclass
class ColumnIndex:
    """
    Pressure-sorted index of all levels of *variable* across all streams.

    ``levels`` maps pressure (hPa) → :class:`LevelRef`, ordered top → bottom.
    """

    variable: str
    levels: dict[float, LevelRef] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.levels)

    def __bool__(self) -> bool:
        return bool(self.levels)

    @property
    def pressures(self) -> list[float]:
        return list(self.levels)

    @property
    def streams(self) -> list[str]:
        return sorted({r.stream for r in self.levels.values()})

    def nearest(self, pressure_hpa: float, tolerance_frac: float = 0.25) -> LevelRef:
        """
        Level closest to *pressure_hpa* in log-pressure.

        Raises:
            KeyError: if no level is within ``tolerance_frac`` (relative).
        """
        if not self.levels:
            raise KeyError(f"no levels for variable {self.variable!r}")
        p = min(self.levels, key=lambda q: abs(math.log(q) - math.log(pressure_hpa)))
        if abs(p - pressure_hpa) / pressure_hpa > tolerance_frac:
            raise KeyError(
                f"{self.variable}: nearest available level {p:g} hPa is too far from "
                f"{pressure_hpa:g} hPa (have {[f'{q:g}' for q in self.levels]})"
            )
        return self.levels[p]

    def select(self, pressures_hpa: list[float] | None, strict: bool = False) -> list[LevelRef]:
        """
        Resolve a list of requested pressures; ``None`` → all levels.
        Missing levels are skipped with a warning unless *strict*.
        """
        if pressures_hpa is None:
            return list(self.levels.values())
        out: list[LevelRef] = []
        for p in pressures_hpa:
            try:
                out.append(self.nearest(p))
            except KeyError as e:
                if strict:
                    raise
                _logger.warning("%s", e)
        return out

    def between(self, p_top: float, p_bottom: float) -> list[LevelRef]:
        """Levels with ``p_top <= p <= p_bottom`` (hPa)."""
        return [r for p, r in self.levels.items() if p_top <= p <= p_bottom]


def discover_column(
    zio: ZarrIO,
    variable: str,
    sample: int = 0,
    streams: list[str] | None = None,
    level_types: dict[str, LevelType] | None = None,
) -> ColumnIndex:
    """
    Build the :class:`ColumnIndex` of *variable* over all (or the given) streams.

    The level type of each stream comes from ``level_types`` (if given), then
    :data:`weathergen.stratosphere.levels.STREAM_LEVEL_TYPES` / the
    ``WG_STRATO_LEVEL_TYPES`` env var, then a heuristic on the channel names.
    When two streams provide the same pressure (within 2 %), the first stream
    in store order wins and the duplicate is logged.
    """
    idx = ColumnIndex(variable)
    for stream in streams or get_streams(zio):
        channels = get_channels(zio, stream, sample)
        if not channels:
            continue
        lt = (level_types or {}).get(stream) or stream_level_type(stream, channels)
        for ch, p in build_level_map(variable, channels, lt).items():
            dup = [q for q in idx.levels if abs(q - p) / p < 0.02]
            if dup:
                _logger.debug(
                    "%s %g hPa from %s shadows %s/%s",
                    variable,
                    p,
                    idx.levels[dup[0]].stream,
                    stream,
                    ch,
                )
                continue
            idx.levels[p] = LevelRef(variable, stream, ch, p, lt)
    idx.levels = dict(sorted(idx.levels.items()))
    return idx


def discover_columns(
    zio: ZarrIO, variables: list[str], sample: int = 0, **kw
) -> dict[str, ColumnIndex]:
    """:func:`discover_column` for several variables."""
    return {v: discover_column(zio, v, sample, **kw) for v in variables}


def common_pressures(*columns: ColumnIndex, tol_frac: float = 0.02) -> list[float]:
    """Pressures present in every column (e.g. for u, v, t together)."""
    if not columns:
        return []
    out = []
    for p in columns[0].pressures:
        if all(any(abs(q - p) / p < tol_frac for q in c.pressures) for c in columns[1:]):
            out.append(p)
    return out


def nearest_common_pressure(
    columns: list[ColumnIndex], pressure_hpa: float, tol_frac: float = 0.25
) -> float | None:
    """
    Pressure available in *all* columns closest (log-p) to *pressure_hpa*, or
    ``None`` if none is within *tol_frac*.  E.g. for v'T' at 100 hPa with
    fallback to 70/50 hPa when 100 hPa is missing in one arm.
    """
    common = common_pressures(*columns)
    if not common:
        return None
    p = min(common, key=lambda q: abs(math.log(q) - math.log(pressure_hpa)))
    return p if abs(p - pressure_hpa) / pressure_hpa <= tol_frac else None


def surface_channel_ref(zio: ZarrIO, channel: str, sample: int = 0) -> LevelRef | None:
    """Locate a surface channel (``'2t'``, ``'msl'``, ``'10u'`` ...) in any stream."""
    for stream in get_streams(zio):
        if channel in get_channels(zio, stream, sample):
            return LevelRef(channel, stream, channel, float("nan"), "pressure")
    return None


def load_level(
    zio: ZarrIO,
    ref: LevelRef,
    step: int,
    sample: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray]:
    """
    Prediction and target of one level at one step.

    Returns:
        pred, tgt: ``(n_pts, n_ens)``; times: ``(n_pts,)`` datetime64[ns].
    """
    item = zio.get_data(sample, ref.stream, step)
    channels = list(item.prediction.channels)
    ci = channels.index(ref.channel)
    # Real stores write target/source without the ensemble axis (2D:
    # (n_pts, n_channels)), unlike prediction (3D: (n_pts, n_channels, n_ens));
    # np.atleast_3d appends the missing axis (n_ens=1) so both index the same
    # way below (see _TEM_STREAM handling in analyze_qg_tem.py for the same
    # defensive pattern on load_step).
    pred = np.atleast_3d(np.asarray(item.prediction.data))[:, ci, :]
    tgt = np.atleast_3d(np.asarray(item.target.data))[:, ci, :]
    times = np.asarray(item.prediction.times)
    return pred, tgt, times


def load_levels(
    zio: ZarrIO,
    refs: list[LevelRef],
    step: int,
    sample: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray]:
    """
    Several levels (possibly from different streams) at one step, stacked on a
    new axis 1: ``(n_pts, n_levels, n_ens)``.  All streams must share the grid;
    a :class:`ValueError` is raised otherwise.
    """
    cache: dict[str, tuple[NDArray, NDArray, list[str], NDArray]] = {}
    preds, tgts, times = [], [], None
    for ref in refs:
        if ref.stream not in cache:
            item = zio.get_data(sample, ref.stream, step)
            cache[ref.stream] = (
                # see load_level: target lacks the ensemble axis in real stores
                np.atleast_3d(np.asarray(item.prediction.data)),
                np.atleast_3d(np.asarray(item.target.data)),
                list(item.prediction.channels),
                np.asarray(item.prediction.times),
            )
        p, t, chans, tm = cache[ref.stream]
        ci = chans.index(ref.channel)
        if preds and p.shape[0] != preds[0].shape[0]:
            raise ValueError(
                f"stream {ref.stream} has {p.shape[0]} points, expected {preds[0].shape[0]}"
            )
        preds.append(p[:, ci, :])
        tgts.append(t[:, ci, :])
        times = tm if times is None else times
    return np.stack(preds, axis=1), np.stack(tgts, axis=1), times


def load_source_level(
    zio: ZarrIO, ref: LevelRef, sample: int = 0
) -> tuple[NDArray[np.float32], NDArray]:
    """Source (analysis) field of one level at step 0: ``(n_pts, n_ens)``, times."""
    item = zio.get_data(sample, ref.stream, 0)
    channels = list(item.source.channels)
    ci = channels.index(ref.channel)
    # see load_level: source also lacks the ensemble axis in real stores
    return (
        np.atleast_3d(np.asarray(item.source.data))[:, ci, :],
        np.asarray(item.source.times),
    )


def resolve_channel_arg(
    zio: ZarrIO,
    variable: str,
    channels: list[str] | None,
    levels_hpa: list[float] | None,
    default_levels_hpa: list[float],
    sample: int = 0,
    fallback_all: bool = True,
) -> list[LevelRef]:
    """
    Resolve the usual CLI pair ``--channels u_29 ...`` (legacy, channel names)
    / ``--levels 10 50`` (hPa) into :class:`LevelRef` objects.

    Legacy channel names are looked up in every stream; levels are matched via
    :meth:`ColumnIndex.nearest`.  Falls back to *default_levels_hpa*, and
    finally (if *fallback_all*) to every level of the variable when none of
    the defaults exist; with ``fallback_all=False`` an empty list is returned
    instead, which single-level analyses should treat as "skip this run".
    """
    col = discover_column(zio, variable, sample)
    if channels:
        refs = []
        for ch in channels:
            hits = [r for r in col.levels.values() if r.channel == ch]
            if not hits:
                raise KeyError(f"channel {ch!r} not found in any stream ({col.streams})")
            refs.append(hits[0])
        return refs
    refs = col.select(levels_hpa or default_levels_hpa)
    if not refs and fallback_all:
        _logger.warning(
            "%s: none of %s hPa available, using all %d levels",
            variable,
            levels_hpa or default_levels_hpa,
            len(col),
        )
        refs = list(col.levels.values())
    return refs
