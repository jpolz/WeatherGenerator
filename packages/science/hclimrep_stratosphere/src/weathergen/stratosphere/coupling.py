# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Stratosphere–troposphere coupling and forecast-climatology diagnostics.

Pure numpy helpers operating on unstructured ``(n_pts,)`` fields with a
``(n_pts, 2)`` ``[lat, lon]`` coordinate array (any grid; area weights come
from :func:`weathergen.stratosphere.io.get_area_weights`).  They are used by
``scripts/analyze_forecast_climatology.py`` (lead-dependent bias, blurring,
zonal spectra) and ``scripts/analyze_coupling.py`` (polar-cap height
"dripping paint" plots, downward-propagation lag correlation, thermal-wind
consistency).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from weathergen.stratosphere.io import find_polar_cap_indices, get_area_weights

R_DRY = 287.0  # J kg^-1 K^-1
EARTH_RADIUS = 6.371e6  # m
OMEGA = 7.292e-5  # s^-1


def zonal_mean_by_latitude(
    field: NDArray,
    coords: NDArray,
    lat_bins: NDArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Bin-average *field* into latitude bands.

    Args:
        field:    ``(n_pts,)`` values.
        coords:   ``(n_pts, 2)`` ``[lat, lon]`` in degrees.
        lat_bins: Bin edges in degrees (default ``np.arange(-90, 91, 2.5)``).
                  The upper edge is inclusive so the pole is not dropped.

    Returns:
        ``(lat_centers, zonal_mean)`` — both ``(n_bins,)``; bins without any
        grid point are ``NaN``.
    """
    if lat_bins is None:
        lat_bins = np.arange(-90.0, 91.0, 2.5)
    lat_bins = np.asarray(lat_bins, dtype=float)
    field = np.asarray(field, dtype=float)
    lats = np.asarray(coords[:, 0], dtype=float)
    n_bins = len(lat_bins) - 1
    centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])

    inside = (lats >= lat_bins[0]) & (lats <= lat_bins[-1]) & np.isfinite(field)
    idx = np.clip(np.digitize(lats[inside], lat_bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=field[inside], minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        zm = np.where(counts > 0, sums / counts, np.nan)
    return centers, zm


def polar_cap_anomaly(
    pred: NDArray,
    tgt: NDArray,
    coords: NDArray,
    weights: NDArray | None = None,
    min_lat: float = 60.0,
) -> float:
    """
    Forecast error of the area-weighted polar-cap mean: ``mean(pred) - mean(tgt)``
    over ``lat >= min_lat`` (northern hemisphere).

    Args:
        pred, tgt: ``(n_pts,)`` fields on the full grid.
        coords:    ``(n_pts, 2)`` ``[lat, lon]``.
        weights:   Optional ``(n_pts,)`` full-grid area weights; default
                   :func:`weathergen.stratosphere.io.get_area_weights`.
        min_lat:   Equatorward edge of the cap in °N.

    Returns:
        Scalar polar-cap mean error (units of the field).
    """
    idx = find_polar_cap_indices(coords, min_lat, "north")
    w = get_area_weights(coords, idx) if weights is None else np.asarray(weights, float)[idx]
    w = w / w.sum()
    diff = np.asarray(pred, float)[idx] - np.asarray(tgt, float)[idx]
    return float(np.sum(w * diff))


def _wmean(x: NDArray, w: NDArray, axis: int = -1) -> NDArray:
    return np.sum(w * x, axis=axis) / np.sum(w)


def lead_dependent_bias(
    pred: NDArray,
    tgt: NDArray,
    weights: NDArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Area-weighted bias, RMSE and spatial-variance ratio per forecast step.

    ``var_ratio = var_w(pred) / var_w(tgt)`` (weighted spatial variance at
    each step) is the "blurring" measure: values < 1 mean the forecast has
    lost spatial variability relative to the target.

    Args:
        pred, tgt: ``(n_steps, n_pts)``.
        weights:   ``(n_pts,)`` area weights (normalised internally); uniform
                   if ``None``.

    Returns:
        ``(bias, rmse, var_ratio)`` — each ``(n_steps,)``.
    """
    pred = np.asarray(pred, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    n_pts = pred.shape[-1]
    w = np.ones(n_pts) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()

    diff = pred - tgt
    bias = _wmean(diff, w)
    rmse = np.sqrt(_wmean(diff**2, w))
    var_p = _wmean((pred - _wmean(pred, w)[..., None]) ** 2, w)
    var_t = _wmean((tgt - _wmean(tgt, w)[..., None]) ** 2, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        var_ratio = np.where(var_t > 0, var_p / var_t, np.nan)
    return bias, rmse, var_ratio


def zonal_wavenumber_spectrum(
    field: NDArray,
    coords: NDArray,
    lat_center: float = 60.0,
    lat_halfwidth: float = 2.5,
    n_lon: int = 128,
    kmax: int = 10,
) -> NDArray[np.float64]:
    """
    Zonal wavenumber power spectrum of *field* along a latitude ring.

    Grid points with ``|lat - lat_center| <= lat_halfwidth`` are sorted by
    longitude and linearly interpolated (periodic in 360°) onto *n_lon*
    regular longitudes before an FFT.

    Returns:
        ``power`` of shape ``(kmax + 1,)`` normalised to squared amplitude:
        ``power[k] = A_k**2`` for a wave ``A_k cos(k λ + φ)`` (``k >= 1``) and
        ``power[0]`` = squared zonal mean.  All-``NaN`` if the ring holds fewer
        than two points.
    """
    field = np.asarray(field, dtype=float)
    lats = coords[:, 0]
    lons = np.mod(np.asarray(coords[:, 1], dtype=float), 360.0)
    sel = np.where((np.abs(lats - lat_center) <= lat_halfwidth) & np.isfinite(field))[0]
    if len(sel) < 2:
        return np.full(kmax + 1, np.nan)

    order = np.argsort(lons[sel])
    ring_lon = lons[sel][order]
    ring_val = field[sel][order]
    lon_reg = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    reg = np.interp(lon_reg, ring_lon, ring_val, period=360.0)

    coef = np.fft.rfft(reg) / n_lon
    power = np.abs(coef) ** 2
    power[1:] *= 4.0
    out = np.full(kmax + 1, np.nan)
    n = min(kmax + 1, len(power))
    out[:n] = power[:n]
    return out


def thermal_wind_residual(
    u_top: NDArray,
    u_bot: NDArray,
    t_mid: NDArray,
    coords: NDArray,
    p_top: float,
    p_bot: float,
    lat_bins: NDArray | None = None,
    min_abs_lat: float = 20.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Zonal-mean thermal-wind consistency check between two pressure levels.

    In pressure coordinates the geostrophic thermal wind is
    ``∂ū/∂ln p = (R / (f a)) ∂T̄/∂φ`` (Holton & Hakim eq. 3.30; with
    ``f = 2Ω sin φ``, ``R = 287``, ``a = 6.371e6``).  The left-hand side is
    estimated by the finite difference ``(ū_top - ū_bot) / (ln p_top - ln p_bot)``
    and the right-hand side from the meridional gradient of the layer-mean
    zonal-mean temperature ``t_mid``.  Note the sign: with a cold pole
    (``∂T̄/∂φ < 0`` in the NH) westerlies strengthen with height.

    Args:
        u_top, u_bot: ``(n_pts,)`` zonal wind at *p_top* and *p_bot*.
        t_mid:        ``(n_pts,)`` temperature representative of the layer.
        coords:       ``(n_pts, 2)`` ``[lat, lon]``.
        p_top, p_bot: Pressures in hPa (``p_top < p_bot``).
        lat_bins:     Edges for :func:`zonal_mean_by_latitude`.
        min_abs_lat:  Residual is ``NaN`` equatorward of this latitude where
                      ``f → 0``.

    Returns:
        ``(lat_centers, residual)`` with ``residual = LHS - RHS`` in
        m s⁻¹ per unit ln p.
    """
    lat_c, u_t = zonal_mean_by_latitude(u_top, coords, lat_bins)
    _, u_b = zonal_mean_by_latitude(u_bot, coords, lat_bins)
    _, t_m = zonal_mean_by_latitude(t_mid, coords, lat_bins)

    dlnp = np.log(float(p_top)) - np.log(float(p_bot))
    lhs = (u_t - u_b) / dlnp

    # gradient only over populated bins so empty bins do not poison neighbours
    residual = np.full_like(lat_c, np.nan)
    valid = np.isfinite(t_m) & np.isfinite(lhs)
    if valid.sum() >= 2:
        phi = np.deg2rad(lat_c[valid])
        dt_dphi = np.gradient(t_m[valid], phi)
        f = 2.0 * OMEGA * np.sin(phi)
        with np.errstate(invalid="ignore", divide="ignore"):
            rhs = R_DRY / (f * EARTH_RADIUS) * dt_dphi
        residual[valid] = lhs[valid] - rhs
    residual[np.abs(lat_c) <= min_abs_lat] = np.nan
    return lat_c, residual


def lagged_correlation(
    x: NDArray, y: NDArray, max_lag: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Pearson correlation of ``x[t]`` with ``y[t + lag]`` for ``lag`` in
    ``[-max_lag, max_lag]``.

    A maximum at positive lag means *y* lags *x* — e.g. ``x`` = 10 hPa
    polar-cap height, ``y`` = 1000 hPa: the surface follows the stratosphere
    (downward propagation).  Pairs containing ``NaN`` are dropped; ``r`` is
    ``NaN`` when fewer than three pairs remain or a series is constant.

    Returns:
        ``(lags, r)`` — both ``(2 * max_lag + 1,)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    r = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            xs, ys = x[: n - lag], y[lag:]
        else:
            xs, ys = x[-lag:], y[: n + lag]
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() < 3:
            continue
        xs, ys = xs[ok], ys[ok]
        sx, sy = xs.std(), ys.std()
        if sx == 0 or sy == 0:
            continue
        r[i] = np.mean((xs - xs.mean()) * (ys - ys.mean())) / (sx * sy)
    return lags, r
