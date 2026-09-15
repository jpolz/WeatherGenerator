# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Stratospheric diagnostics: SSW detection, polar vortex, and related metrics.

All functions operate on plain numpy arrays; they do not open zarr files.
Use :mod:`weathergen.stratosphere.io` for data loading.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# SSW detection
# ---------------------------------------------------------------------------


def detect_ssw_reversal(
    u_wind: NDArray[np.float32],
    datetimes: list[datetime],
    threshold: float = 0.0,
) -> dict[str, Any] | None:
    """
    Detect an SSW wind-reversal event.

    An SSW is characterised by a reversal of the zonal mean zonal wind at
    60°N / 10 hPa from westerly to easterly (WMO definition: November–March).

    Args:
        u_wind:    1-D time series of zonal mean u-wind at 60°N (m/s).
        datetimes: Matching list of datetime objects.
        threshold: Wind speed below which reversal is declared (default 0 m/s).

    Returns:
        Dict with reversal info, or ``None`` if no reversal detected::

            {
                "detected":       True,
                "reversal_date":  datetime,
                "reversal_index": int,
                "reversal_u":     float,
                "min_u":          float,
                "min_date":       datetime,
            }
    """
    reversal_indices = np.where(u_wind < threshold)[0]
    if len(reversal_indices) == 0:
        return None

    idx = int(reversal_indices[0])
    min_idx = int(np.argmin(u_wind))

    return {
        "detected": True,
        "reversal_date": datetimes[idx],
        "reversal_index": idx,
        "reversal_u": float(u_wind[idx]),
        "min_u": float(u_wind[min_idx]),
        "min_date": datetimes[min_idx],
    }


def detect_warming_event(
    temperature: NDArray[np.float32],
    datetimes: list[datetime],
    warming_threshold: float = 10.0,
    window_days: int = 7,
) -> dict[str, Any] | None:
    """
    Detect a sudden polar warming event from a temperature time series.

    Args:
        temperature:       Polar cap mean temperature (K).
        datetimes:         Matching list of datetime objects.
        warming_threshold: Minimum temperature increase (K) over *window_days*.
        window_days:       Window length for warming rate calculation.

    Returns:
        Dict with warming event info, or ``None`` if not detected.
    """
    if len(temperature) < 2:
        return None

    dt_hours = (datetimes[1] - datetimes[0]).total_seconds() / 3600
    window_steps = max(1, int(window_days * 24 / dt_hours))

    if len(temperature) < window_steps:
        return None

    max_warming, warming_idx = 0.0, 0
    for i in range(len(temperature) - window_steps):
        w = temperature[i + window_steps] - temperature[i]
        if w > max_warming:
            max_warming, warming_idx = w, i

    if max_warming < warming_threshold:
        return None

    return {
        "detected": True,
        "warming_start": datetimes[warming_idx],
        "warming_end": datetimes[warming_idx + window_steps],
        "warming_magnitude": float(max_warming),
        "max_temperature": float(np.max(temperature)),
        "max_temp_date": datetimes[int(np.argmax(temperature))],
    }


# ---------------------------------------------------------------------------
# Spatial aggregation
# ---------------------------------------------------------------------------


def polar_cap_mean(
    data: NDArray[np.float32],
    coords: NDArray[np.float32],
    min_lat: float = 60.0,
    hemisphere: str = "north",
) -> NDArray[np.float64]:
    """
    Compute area-weighted polar cap mean.

    Args:
        data:       ``(n_pts,)`` or ``(n_time, n_pts)`` field array.
        coords:     ``(n_pts, 2)`` [lat, lon] array.
        min_lat:    Minimum absolute latitude.
        hemisphere: ``'north'`` or ``'south'``.

    Returns:
        Scalar or ``(n_time,)`` array of polar cap means.
    """
    from weathergen.stratosphere.io import find_polar_cap_indices, get_area_weights

    indices = find_polar_cap_indices(coords, min_lat, hemisphere)
    weights = get_area_weights(coords, indices)

    if data.ndim == 1:
        return float(np.average(data[indices], weights=weights))
    return np.average(data[:, indices], weights=weights, axis=1)


def zonal_mean(
    data: NDArray[np.float32],
    coords: NDArray[np.float32],
    target_lat: float,
    tolerance: float = 2.5,
) -> NDArray[np.float64]:
    """
    Compute zonal mean at *target_lat*.

    Args:
        data:       ``(n_pts,)`` or ``(n_time, n_pts)`` field array.
        coords:     ``(n_pts, 2)`` [lat, lon] array.
        target_lat: Target latitude in degrees.
        tolerance:  Latitude tolerance in degrees.

    Returns:
        Scalar or ``(n_time,)`` array.
    """
    from weathergen.stratosphere.io import find_latitude_indices

    indices = find_latitude_indices(coords, target_lat, tolerance)

    if data.ndim == 1:
        return float(np.mean(data[indices]))
    return np.mean(data[:, indices], axis=1)


def zonal_mean_profile(
    data: NDArray[np.float32],
    coords: NDArray[np.float32],
    lat_bins: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute zonal mean for every latitude bin.

    Args:
        data:     ``(n_pts,)`` or ``(n_time, n_pts)`` field array.
        coords:   ``(n_pts, 2)`` [lat, lon] array.
        lat_bins: Bin edges (default: 5° bins from −90 to 90).

    Returns:
        ``(zonal_means, lat_centers)``  — shapes ``(n_lats,)`` or
        ``(n_time, n_lats)`` and ``(n_lats,)``.
    """
    if lat_bins is None:
        lat_bins = np.arange(-90.0, 95.0, 5.0)

    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    lats = coords[:, 0]
    n_lats = len(lat_centers)

    if data.ndim == 2:
        n_times = data.shape[0]
        out = np.full((n_times, n_lats), np.nan)
    else:
        out = np.full(n_lats, np.nan)

    for i in range(n_lats):
        mask = (lats >= lat_bins[i]) & (lats < lat_bins[i + 1])
        if np.any(mask):
            if data.ndim == 2:
                out[:, i] = np.mean(data[:, mask], axis=1)
            else:
                out[i] = np.mean(data[mask])

    return out, lat_centers


# ---------------------------------------------------------------------------
# Vortex diagnostics
# ---------------------------------------------------------------------------


def vortex_strength(
    u_wind: NDArray[np.float32],
    coords: NDArray[np.float32],
    lat_band: tuple[float, float] = (55.0, 65.0),
) -> NDArray[np.float64]:
    """
    Area-weighted zonal wind averaged over *lat_band* as a vortex strength index.
    """
    from weathergen.stratosphere.io import find_latitude_band_indices, get_area_weights

    indices = find_latitude_band_indices(coords, lat_band[0], lat_band[1])
    weights = get_area_weights(coords, indices)

    if u_wind.ndim == 1:
        return float(np.average(u_wind[indices], weights=weights))
    return np.average(u_wind[:, indices], weights=weights, axis=1)


def nao_index(
    pressure: NDArray[np.float32],
    coords: NDArray[np.float32],
    azores_lat: float = 37.7,
    iceland_lat: float = 65.1,
) -> NDArray[np.float64]:
    """
    Simplified NAO index: normalised pressure difference Azores − Iceland.

    Args:
        pressure:    ``(n_time, n_pts)`` sea-level pressure field.
        coords:      ``(n_pts, 2)`` [lat, lon] array.
        azores_lat:  Latitude for southern node (Azores, ~37.7°N).
        iceland_lat: Latitude for northern node (Iceland, ~65.1°N).

    Returns:
        ``(n_time,)`` NAO index.
    """
    from weathergen.stratosphere.io import find_latitude_indices

    az_idx = find_latitude_indices(coords, azores_lat)
    ic_idx = find_latitude_indices(coords, iceland_lat)

    az = np.mean(pressure[:, az_idx], axis=1)
    ic = np.mean(pressure[:, ic_idx], axis=1)

    diff = az - ic
    return (diff - diff.mean()) / diff.std()
