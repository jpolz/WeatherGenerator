# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
I/O utilities for stratospheric analysis.

Thin adapter over ``weathergen.common.io.zarrio_reader`` that exposes the
data access patterns used by the analysis scripts:

    with open_validation(path) as zio:
        stream_name = get_stream(zio)          # 'ERA5ml' or 'ERA5pl'
        channels    = get_channels(zio, stream_name)
        coords      = get_coords(zio, stream_name)   # (n_pts, 2) [lat, lon]
        for step in get_forecast_steps(zio):
            pred, tgt, times = load_step(zio, stream_name, step)
            # pred / tgt: (n_pts, n_channels, n_ens)  numpy arrays
            # times:      (n_pts,)                    numpy datetime64[ns]

Coordinate helpers (find_latitude_indices etc.) work on the ``coords``
array and are independent of the zarr backend.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from weathergen.common.io import ZarrIO, zarrio_reader

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public re-export so callers only need to import from this module
# ---------------------------------------------------------------------------
open_validation = zarrio_reader  # context-manager: open_validation(path) -> ZarrIO


# ---------------------------------------------------------------------------
# ZarrIO accessors
# ---------------------------------------------------------------------------


def get_stream(zio: ZarrIO) -> str:
    """Return the primary stream name ('ERA5ml' preferred over 'ERA5pl')."""
    streams = zio.streams
    for preferred in ("ERA5ml", "ERA5pl"):
        if preferred in streams:
            return preferred
    return streams[0]


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
    """Return list of channel names for *stream*."""
    steps = get_forecast_steps(zio)
    item = zio.get_data(sample, stream, steps[0])
    return list(item.prediction.channels)


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
    pred = np.asarray(item.prediction.data)
    tgt = np.asarray(item.target.data)
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
    src = np.asarray(item.source.data)
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
        raise ValueError(
            f"No grid points found near {target_lat}°N (tolerance={tolerance}°)"
        )
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
    Compute cosine-latitude area weights for a subset of grid points.

    Returns normalized weights summing to 1.
    """
    lats = coords[indices, 0]
    weights = np.cos(np.deg2rad(lats))
    return weights / weights.sum()


def convert_times_to_datetime(times: NDArray) -> list:
    """Convert numpy datetime64 array to Python datetime objects."""
    return [t.astype("datetime64[s]").astype(object) for t in times]
