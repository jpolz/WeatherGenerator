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


# ---------------------------------------------------------------------------
# SSW precursor diagnostics
# ---------------------------------------------------------------------------


def heat_flux_wave_activity(
    v: NDArray[np.float32],
    t: NDArray[np.float32],
    coords: NDArray[np.float32],
    lat_band: tuple[float, float] = (45.0, 75.0),
    model_level: int | None = None,
) -> NDArray[np.float64] | float:
    """
    Area-weighted eddy heat flux v'T' averaged over *lat_band*.

    Poleward eddy heat flux is a standard proxy for upward-propagating wave
    activity into the stratosphere and is a known SSW precursor (enhanced
    heat flux ~1-2 weeks ahead of onset). Computed as the zonal-mean eddy
    covariance of *v* and *t* on a single model level.

    Args:
        v, t:       Meridional wind (m/s) and temperature (K) on one model
                    level. Shape ``(n_pts,)`` or ``(n_time, n_pts)``.
        coords:     ``(n_pts, 2)`` [lat, lon] array.
        lat_band:   Latitude band to average over (degrees N).
        model_level: Model-level index of *v*/*t*, kept only for labeling —
                    not used to select data (caller must pass the correct
                    channel slice). The channel nearest 100 hPa in the
                    ``era5_strato_final`` ERA5ml stream is level 61
                    (pf≈103.7 hPa); see stratosphere_stream_channels notes.

    Returns:
        Scalar or ``(n_time,)`` array of eddy heat flux (K m/s).
    """
    from weathergen.stratosphere.io import find_latitude_band_indices, get_area_weights

    indices = find_latitude_band_indices(coords, lat_band[0], lat_band[1])
    weights = get_area_weights(coords, indices)

    def _one(v_field: NDArray, t_field: NDArray) -> float:
        v_sub = v_field[indices]
        t_sub = t_field[indices]
        v_bar = np.average(v_sub, weights=weights)
        t_bar = np.average(t_sub, weights=weights)
        return float(np.average((v_sub - v_bar) * (t_sub - t_bar), weights=weights))

    if v.ndim == 1:
        return _one(v, t)
    return np.array([_one(v[i], t[i]) for i in range(v.shape[0])])


def wave_amplitude(
    z: NDArray[np.float32],
    coords: NDArray[np.float32],
    wavenumber: int = 1,
    lat: float = 60.0,
    tolerance: float = 2.5,
) -> NDArray[np.float64] | float:
    """
    Zonal wavenumber amplitude of *z* at *lat*, via harmonic least-squares fit.

    Fits ``z(lon) = a0 + a*cos(k*lon) + b*sin(k*lon)`` at the grid points
    within *tolerance* of *lat* (works on the unstructured/irregular-longitude
    WeatherGenerator grid, unlike a plain FFT). Wave-1/wave-2 amplitude at
    60°N is a standard planetary-wave precursor for SSW events.

    Args:
        z:          Geopotential (height) field, e.g. ``z_500``. Shape
                    ``(n_pts,)`` or ``(n_time, n_pts)``.
        coords:     ``(n_pts, 2)`` [lat, lon] array.
        wavenumber: Zonal wavenumber ``k`` (1 or 2 for planetary waves).
        lat:        Target latitude (degrees N).
        tolerance:  Latitude tolerance (degrees).

    Returns:
        Scalar or ``(n_time,)`` array of wave amplitude (same units as *z*).
    """
    from weathergen.stratosphere.io import find_latitude_indices

    indices = find_latitude_indices(coords, lat, tolerance)
    lons = np.deg2rad(coords[indices, 1].astype(np.float64))
    design = np.column_stack(
        [np.ones_like(lons), np.cos(wavenumber * lons), np.sin(wavenumber * lons)]
    )

    def _one(z_field: NDArray) -> float:
        z_sub = z_field[indices].astype(np.float64)
        coeffs, *_ = np.linalg.lstsq(design, z_sub, rcond=None)
        _, a, b = coeffs
        return float(np.hypot(a, b))

    if z.ndim == 1:
        return _one(z)
    return np.array([_one(z[i]) for i in range(z.shape[0])])


def blocking_index(
    z: NDArray[np.float32],
    coords: NDArray[np.float32],
    lat_south: float = 40.0,
    lat_central: float = 60.0,
    lat_north: float = 80.0,
    lon_width: float = 5.0,
    tolerance: float = 2.5,
    ghgs_threshold: float = 0.0,
    ghgn_threshold: float = -10.0,
) -> dict[str, NDArray]:
    """
    Tibaldi-Molteni-style meridional gradient reversal blocking index.

    For longitude sectors of width *lon_width*, computes the southern
    (GHGS) and northern (GHGN) 500 hPa geopotential gradients between
    *lat_south*/*lat_central*/*lat_north*, and flags a sector as blocked
    when ``GHGS > ghgs_threshold`` (reversed/weak subtropical gradient) and
    ``GHGN < ghgn_threshold`` (strong poleward gradient reversal). This is a
    tropospheric blocking precursor associated with subsequent stratospheric
    wave-driven warming.

    Args:
        z:          Geopotential (height) field, e.g. ``z_500``. Shape
                    ``(n_pts,)`` or ``(n_time, n_pts)``.
        coords:     ``(n_pts, 2)`` [lat, lon] array.
        lat_south, lat_central, lat_north: The three reference latitudes.
        lon_width:  Longitude sector width (degrees).
        tolerance:  Latitude tolerance for the sample nearest each reference
                    latitude in a sector (degrees).
        ghgs_threshold, ghgn_threshold: Gradient thresholds (m per degree
            latitude) for the classic TM-index blocking criterion.

    Returns:
        Dict with ``lon_centers`` ``(n_lon,)`` and, per time step:
        ``blocked`` boolean array (``(n_lon,)`` or ``(n_time, n_lon)``) and
        ``index`` — the fraction of blocked longitudes (scalar or
        ``(n_time,)``).
    """
    lats = coords[:, 0]
    lons = coords[:, 1] % 360.0
    lon_edges = np.arange(0.0, 360.0 + lon_width, lon_width)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    def _sector_indices(lat_target: float) -> list[NDArray[np.intp]]:
        lat_mask = np.abs(lats - lat_target) <= tolerance
        return [
            np.where(lat_mask & (lons >= lo) & (lons < hi))[0]
            for lo, hi in zip(lon_edges[:-1], lon_edges[1:])
        ]

    idx_s = _sector_indices(lat_south)
    idx_c = _sector_indices(lat_central)
    idx_n = _sector_indices(lat_north)

    def _one(z_field: NDArray) -> NDArray[np.bool_]:
        n = len(lon_centers)
        blocked = np.zeros(n, dtype=bool)
        for i in range(n):
            if len(idx_s[i]) == 0 or len(idx_c[i]) == 0 or len(idx_n[i]) == 0:
                continue
            z_s = float(np.mean(z_field[idx_s[i]]))
            z_c = float(np.mean(z_field[idx_c[i]]))
            z_n = float(np.mean(z_field[idx_n[i]]))
            ghgs = (z_c - z_s) / (lat_central - lat_south)
            ghgn = (z_n - z_c) / (lat_north - lat_central)
            blocked[i] = ghgs > ghgs_threshold and ghgn < ghgn_threshold
        return blocked

    if z.ndim == 1:
        blocked = _one(z)
        return {
            "lon_centers": lon_centers,
            "blocked": blocked,
            "index": float(np.mean(blocked)),
        }

    n_t = z.shape[0]
    blocked_all = np.zeros((n_t, len(lon_centers)), dtype=bool)
    for i in range(n_t):
        blocked_all[i] = _one(z[i])
    return {
        "lon_centers": lon_centers,
        "blocked": blocked_all,
        "index": blocked_all.mean(axis=1),
    }


# ---------------------------------------------------------------------------
# NAM / Northern Annular Mode index
# ---------------------------------------------------------------------------


def nam_index_eof_reference(
    z_anom_ref: NDArray[np.float64],
    domain_coords: NDArray[np.float32],
    high_lat_threshold: float = 70.0,
    full_coords: NDArray[np.float32] | None = None,
    domain_indices: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute a reference EOF pattern from ERA5 (or another reference dataset).

    This is computed once and then reused to project all predictions and targets
    onto the same spatial pattern, avoiding sign ambiguity from independent SVD
    calls at each pressure level.

    Args:
        z_anom_ref:    Reference geopotential anomalies (e.g. ERA5 time-mean
                       or climatology removed), shape ``(n_time, n_domain_pts)``.
        domain_coords: ``(n_domain_pts, 2)`` [lat, lon] for the same points.
        high_lat_threshold: Latitude for sign-fixing convention.
        full_coords:   Optional ``(n_pts, 2)`` full-grid coords, needed to
                       correctly detect an O<n> reduced Gaussian grid (the
                       *domain_coords* subset alone can't be matched against
                       the O<n> point-count formula). If omitted, falls back
                       to plain cosine-latitude weighting.
        domain_indices: Indices into *full_coords* selecting *domain_coords*
                       (required together with *full_coords*).

    Returns:
        (eof1_pattern, singular_value): The leading EOF spatial pattern 
        (length n_domain_pts, normalized) and its singular value.
    """
    lat = domain_coords[:, 0]
    if full_coords is not None and domain_indices is not None:
        from weathergen.stratosphere.io import get_o_grid_weights

        w = np.sqrt(get_o_grid_weights(full_coords, domain_indices))
    else:
        w = np.sqrt(np.cos(np.deg2rad(lat)))

    weighted = z_anom_ref * w[None, :]
    weighted = weighted - weighted.mean(axis=0, keepdims=True)

    u, s, _vt = np.linalg.svd(weighted.T, full_matrices=False)
    eof1 = u[:, 0]  # (n_domain_pts,)
    sv = s[0]

    # Fix sign so positive index = low heights at high lats
    high_idx = np.where(lat >= high_lat_threshold)[0]
    if len(high_idx) == 0:
        high_idx = np.argsort(lat)[-max(1, len(lat) // 10) :]
    if np.dot(eof1[high_idx], z_anom_ref[:, high_idx].mean(axis=0)) > 0:
        eof1 = -eof1

    return eof1, sv


def nam_index_eof_project(
    z_anom: NDArray[np.float64],
    eof1_pattern: NDArray[np.float64],
    domain_coords: NDArray[np.float32],
    full_coords: NDArray[np.float32] | None = None,
    domain_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """
    Project geopotential anomalies onto a pre-computed reference EOF pattern.

    Args:
        z_anom:        Geopotential anomalies to project, shape ``(n_time, n_domain_pts)``.
        eof1_pattern:  Reference EOF spatial pattern from ``nam_index_eof_reference``.
        domain_coords: ``(n_domain_pts, 2)`` [lat, lon] for the same points.
        full_coords:   Optional full-grid coords for O<n> grid detection (see
                       ``nam_index_eof_reference``); must match what was used
                       to build *eof1_pattern*.
        domain_indices: Indices into *full_coords* selecting *domain_coords*.

    Returns:
        ``(n_time,)`` PC time series, unit-std normalized.
    """
    lat = domain_coords[:, 0]
    if full_coords is not None and domain_indices is not None:
        from weathergen.stratosphere.io import get_o_grid_weights

        w = np.sqrt(get_o_grid_weights(full_coords, domain_indices))
    else:
        w = np.sqrt(np.cos(np.deg2rad(lat)))

    weighted = z_anom * w[None, :]
    weighted = weighted - weighted.mean(axis=0, keepdims=True)

    # Project onto reference pattern: dot product with EOF1
    pc1 = np.dot(weighted, eof1_pattern)
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-10)
    return pc1


def nam_index_eof(
    z_anom: NDArray[np.float32],
    domain_coords: NDArray[np.float32],
    high_lat_threshold: float = 70.0,
    full_coords: NDArray[np.float32] | None = None,
    domain_indices: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """
    NAM/AO index: leading EOF (PC1) of area-weighted geopotential anomalies.

    Standard approach: EOF1 of geopotential-height anomalies poleward of
    ~20°N, PC1 normalized to unit std, sign fixed so that positive values
    correspond to anomalously LOW geopotential height at high latitudes
    (i.e. a stronger polar vortex / positive AO-like phase).

    Uses O<n> reduced Gaussian grid weighting if *full_coords*/*domain_indices*
    are given and match a known O<n> grid, otherwise falls back to
    cosine-latitude weighting.

    **Note:** For consistency across multiple datasets/pressure levels, prefer
    using ``nam_index_eof_reference`` once on ERA5 and then 
    ``nam_index_eof_project`` for all simulations.

    Args:
        z_anom:        Geopotential (height) anomalies (climatology or
                       time-mean already removed by the caller), already
                       restricted to the EOF domain (e.g. poleward of 20°N).
                       Shape ``(n_time, n_domain_pts)``.
        domain_coords: ``(n_domain_pts, 2)`` [lat, lon] for the same points.
        high_lat_threshold: Latitude used to determine the sign convention
                       (mean anomaly poleward of this latitude).
        full_coords:   Optional full-grid coords for O<n> grid detection
                       (the *domain_coords* subset alone can't be matched
                       against the O<n> point-count formula).
        domain_indices: Indices into *full_coords* selecting *domain_coords*.

    Returns:
        ``(n_time,)`` NAM index, unit-std normalized with sign fixed.
    """
    lat = domain_coords[:, 0]
    if full_coords is not None and domain_indices is not None:
        from weathergen.stratosphere.io import get_o_grid_weights

        w = np.sqrt(get_o_grid_weights(full_coords, domain_indices))
    else:
        w = np.sqrt(np.cos(np.deg2rad(lat)))

    weighted = z_anom * w[None, :]
    weighted = weighted - weighted.mean(axis=0, keepdims=True)

    u, s, _vt = np.linalg.svd(weighted, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    pc1 = (pc1 - pc1.mean()) / pc1.std()

    high_idx = np.where(lat >= high_lat_threshold)[0]
    if len(high_idx) == 0:
        # fallback: use the highest-latitude decile if none reach the threshold
        high_idx = np.argsort(lat)[-max(1, len(lat) // 10) :]
    high_lat_mean = z_anom[:, high_idx].mean(axis=1)
    corr = np.corrcoef(pc1, high_lat_mean)[0, 1]
    if corr > 0:
        pc1 = -pc1

    return pc1
