#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Quasi-Geostrophic Transformed Eulerian Mean (QG-TEM) Analysis.

Computes the QG-TEM diagnostics from u, v, T fields on a pressure axis:

  * Potential temperature  θ = T · (p₀/p)^κ
  * EP flux components     F_φ = −u′v′̄                          [m² s⁻²]
                           F_p =  f · v′θ′̄ / (∂θ̄/∂p)           [m Pa s⁻²]
  * EP flux divergence     ∇·F = (1/a cosφ)·∂(cosφ F_φ)/∂φ + ∂F_p/∂p  [m s⁻²]
  * Residual meridional    v* = v̄ − ∂/∂p(v′θ′̄ / ∂θ̄/∂p)
    velocity

Because WeatherGenerator uses an unstructured grid, zonal means are computed
by grouping grid points into latitude bands of configurable width and averaging
within each band.

Multi-stream layout
-------------------
A strato_s2s forecast store spreads the atmospheric column over several
streams: ``ERA5pl`` holds 50–1000 hPa, ``S2Spredictors`` holds 1–70 hPa, and
the ``strat_tem`` arm adds a ``TEM`` stream with the model's *own* EP-flux
prediction (``ep_flux_div_<L>``, ``F_p_<L>``, ``vtheta_prime_<L>`` on L137
model levels, zonal means broadcast to every grid point).  The pressure axis
used for the vertical derivatives is the merged, sorted set of pressures that
u, v **and** t share across all streams (``common_pressures``), 1–1000 hPa.
When a ``TEM`` stream exists, its predicted ∇·F is compared with the
diagnosed one at the nearest pressures.

Outputs (per experiment, saved to ``--output-dir/<label>/``):

  * ``ep_flux_divergence.png``  — latitude–pressure Hovmöller of ∇·F
  * ``ep_flux_vectors.png``     — quiver plot of EP flux + divergence shading
                                  (time-mean over entire rollout)
  * ``v_residual.png``          — Hovmöller of residual circulation v*
  * ``polar_cap_efd.png``       — polar-cap (60–90°N) mean ∇·F time series
  * ``qg_tem_diagnostics.npz``  — numpy archive of all computed fields

Per event (saved to ``--output-dir`` or ``--output-dir/<event>/``):

  * ``polar_cap_efd_timeseries.png``       — all runs, polar-cap ∇·F
  * ``tem_stream_vs_diagnosed_<event>.png`` — TEM-stream ∇·F (pred/target)
                                              vs diagnosed ∇·F; only when a
                                              ``TEM`` stream is present

Usage::

    ssw-analyze qg-tem \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/qg_tem \\
        --p-range 1 100

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_qg_tem \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from weathergen.stratosphere.config import (
    event_reference_date,
    group_validations_by_event,
    load_validations_config,
)
from weathergen.stratosphere.io import (
    LevelRef,
    common_pressures,
    convert_times_to_datetime,
    discover_columns,
    get_channels,
    get_coords,
    get_forecast_steps,
    get_streams,
    load_levels,
    load_step,
    open_validation,
    zarr_store_path,
)
from weathergen.stratosphere.levels import channel_pressure, parse_channel

_logger = logging.getLogger(__name__)

# Physical constants
_EARTH_RADIUS = 6.371e6  # m
_OMEGA = 7.292e-5  # rad s⁻¹
_P0_PA = 1.0e5  # Pa  (reference pressure for θ)
_KAPPA = 2.0 / 7.0  # R/cp for dry air

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

# Pressure axis used for the diagnostics (hPa): everything u, v, t share.
_AXIS_P_TOP = 1.0
_AXIS_P_BOTTOM = 1000.0

# Standard pressure ticks for log-pressure y-axis
_P_TICKS = [1, 3, 10, 30, 100, 300, 1000]
_P_TICK_LABELS = [str(p) for p in _P_TICKS]

# Channels of the optional TEM stream (model-level suffix).
_TEM_STREAM = "TEM"
_TEM_VARIABLES = ("ep_flux_div", "F_p", "vtheta_prime")
# Units of the TEM stream channels (from the TEM zarr attrs).  ``ep_flux_div``
# matches the diagnosed ∇·F, so both can share one axis.
TEM_STREAM_UNITS: dict[str, str] = {
    "ep_flux_div": "m s⁻¹ day⁻¹",
    "F_p": "K m s⁻¹ Pa⁻¹",
    "vtheta_prime": "K m s⁻¹",
}
_TEM_TITLES = {"F_p": "F_p (vertical EP flux)", "vtheta_prime": "v′θ′ (eddy heat flux)"}


# ---------------------------------------------------------------------------
# Zonal-mean helper for unstructured grids
# ---------------------------------------------------------------------------


def _build_lat_bins(lat_width: float = 2.5) -> tuple[NDArray, NDArray]:
    """Return (edges, centres) for latitude bins spanning −90 to 90."""
    edges = np.arange(-90.0, 90.0 + lat_width, lat_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return edges, centres


def _build_lat_groups(
    lats: NDArray,
    edges: NDArray,
) -> list[NDArray]:
    """
    For each latitude bin return the indices of grid points that fall in it.
    """
    groups: list[NDArray] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = np.where((lats >= lo) & (lats < hi))[0]
        groups.append(idx)
    return groups


def _zonal_mean_field(
    data: NDArray,
    groups: list[NDArray],
) -> NDArray:
    """
    Compute the zonal mean of a (n_pts, n_levels) field.

    Returns (n_lats, n_levels).  Bins with no points return NaN.
    """
    n_lats = len(groups)
    n_levels = data.shape[1] if data.ndim == 2 else 1
    zm = np.full((n_lats, n_levels), np.nan, dtype=np.float64)
    for i, idx in enumerate(groups):
        if len(idx) > 0:
            zm[i] = data[idx].mean(axis=0) if data.ndim == 2 else data[idx].mean()
    return zm


# ---------------------------------------------------------------------------
# Core QG-TEM computation
# ---------------------------------------------------------------------------


def compute_qg_tem(
    u: NDArray,
    v: NDArray,
    t: NDArray,
    lats: NDArray,
    pressures_pa: NDArray,
    lat_width: float = 2.5,
) -> dict[str, NDArray]:
    """
    Compute QG-TEM diagnostics from instantaneous 2-D (n_pts, n_levels) fields.

    Parameters
    ----------
    u, v, t:
        Zonal wind (m/s), meridional wind (m/s), temperature (K) on the
        unstructured grid.  Shape ``(n_pts, n_levels)``.
    lats:
        Latitude of each grid point in degrees.  Shape ``(n_pts,)``.
    pressures_pa:
        Pressure of each level in **Pa**.  Shape ``(n_levels,)``.
        Must be sorted ascending (increasing pressure = decreasing altitude).
    lat_width:
        Latitude bin width in degrees for the zonal mean.

    Returns
    -------
    dict with keys (all shapes ``(n_lats, n_levels)`` unless noted):
        ``lat_centres``: (n_lats,)
        ``u_bar``, ``theta_bar``, ``dtheta_dp``
        ``F_phi``: meridional EP flux component  [m² s⁻²]
        ``F_p``:   vertical EP flux component    [m Pa s⁻²]
        ``div_F``: EP flux divergence            [m s⁻¹ day⁻¹]
                   (negative = westward/easterly forcing)
        ``v_star``: residual meridional velocity [m s⁻¹]
    """
    edges, lat_centres = _build_lat_bins(lat_width)
    groups = _build_lat_groups(lats, edges)
    phi = np.deg2rad(lat_centres)  # (n_lats,)
    f = 2.0 * _OMEGA * np.sin(phi)  # Coriolis (n_lats,)
    cos_phi = np.cos(phi)  # (n_lats,)

    n_lats = len(lat_centres)
    n_lev = len(pressures_pa)

    # ---- potential temperature ------------------------------------------------
    theta = t * (_P0_PA / pressures_pa[None, :]) ** _KAPPA  # (n_pts, n_lev)

    # ---- zonal means ---------------------------------------------------------
    u_bar = _zonal_mean_field(u, groups)  # (n_lats, n_lev)
    v_bar = _zonal_mean_field(v, groups)
    theta_bar = _zonal_mean_field(theta, groups)

    # ---- eddy covariances u′v′ and v′θ′ --------------------------------------
    uv_bar = np.full((n_lats, n_lev), np.nan, dtype=np.float64)
    vtheta_bar = np.full((n_lats, n_lev), np.nan, dtype=np.float64)

    for i, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        u_prime = u[idx] - u_bar[i]  # (n_grp, n_lev)
        v_prime = v[idx] - v_bar[i]
        theta_prime = theta[idx] - theta_bar[i]
        uv_bar[i] = (u_prime * v_prime).mean(axis=0)
        vtheta_bar[i] = (v_prime * theta_prime).mean(axis=0)

    # ---- ∂θ̄/∂p — centred finite differences in pressure space ---------------
    dtheta_dp = np.full_like(theta_bar, np.nan)
    dp = pressures_pa[1:] - pressures_pa[:-1]  # (n_lev-1,) positive (ascending p)
    dtheta_dp[:, 1:-1] = (theta_bar[:, 2:] - theta_bar[:, :-2]) / (
        pressures_pa[2:] - pressures_pa[:-2]
    )[None, :]
    dtheta_dp[:, 0] = (theta_bar[:, 1] - theta_bar[:, 0]) / dp[0]
    dtheta_dp[:, -1] = (theta_bar[:, -1] - theta_bar[:, -2]) / dp[-1]
    # In a stable atmosphere ∂θ/∂p < 0; guard against near-zero values
    dtheta_dp_safe = np.where(np.abs(dtheta_dp) > 1e-6, dtheta_dp, np.nan)

    # ---- EP flux components --------------------------------------------------
    # Correct pressure-coordinate QG-TEM definitions (no a cosφ factor here;
    # that factor belongs only in the spherical divergence operator below).
    #
    # F_φ = −u′v′   [m² s⁻²]
    f_phi = -uv_bar

    # F_p = f · v′θ′ / (∂θ̄/∂p)   [m Pa s⁻²]
    f_p = f[:, None] * vtheta_bar / dtheta_dp_safe

    # ---- EP flux divergence ∇·F = (1/a cosφ) ∂(F_φ cosφ)/∂φ + ∂F_p/∂p --------
    # ∂(F_φ cosφ)/∂φ via centred differences in φ
    fphi_cosphi = f_phi * cos_phi[:, None]
    dphi = np.deg2rad(lat_width)
    d_fphi = np.full_like(fphi_cosphi, np.nan)
    d_fphi[1:-1] = (fphi_cosphi[2:] - fphi_cosphi[:-2]) / (2.0 * dphi)
    d_fphi[0] = (fphi_cosphi[1] - fphi_cosphi[0]) / dphi
    d_fphi[-1] = (fphi_cosphi[-1] - fphi_cosphi[-2]) / dphi

    # ∂F_p/∂p via centred differences in pressure
    dfp_dp = np.full_like(f_p, np.nan)
    dfp_dp[:, 1:-1] = (f_p[:, 2:] - f_p[:, :-2]) / (pressures_pa[2:] - pressures_pa[:-2])[None, :]
    dfp_dp[:, 0] = (f_p[:, 1] - f_p[:, 0]) / dp[0]
    dfp_dp[:, -1] = (f_p[:, -1] - f_p[:, -2]) / dp[-1]

    # ∇·F = (1/(a cosφ)) ∂(cosφ F_φ)/∂φ + ∂F_p/∂p
    # Units: (1/m)·(m²/s²) + (m Pa/s²)/Pa = m/s²
    # Multiply by 86 400 s/day → m s⁻¹ day⁻¹
    with np.errstate(invalid="ignore"):
        div_f = (1.0 / (_EARTH_RADIUS * cos_phi[:, None])) * d_fphi + dfp_dp
    div_f_accel = div_f * 86400.0  # m s⁻¹ day⁻¹

    # ---- residual meridional velocity ----------------------------------------
    # v* = v̄ − ∂/∂p (v′θ′ / ∂θ̄/∂p)
    heat_flux_ratio = vtheta_bar / dtheta_dp_safe  # (n_lats, n_lev)
    d_hfr_dp = np.full_like(heat_flux_ratio, np.nan)
    d_hfr_dp[:, 1:-1] = (heat_flux_ratio[:, 2:] - heat_flux_ratio[:, :-2]) / (
        pressures_pa[2:] - pressures_pa[:-2]
    )[None, :]
    d_hfr_dp[:, 0] = (heat_flux_ratio[:, 1] - heat_flux_ratio[:, 0]) / dp[0]
    d_hfr_dp[:, -1] = (heat_flux_ratio[:, -1] - heat_flux_ratio[:, -2]) / dp[-1]

    v_star = v_bar - d_hfr_dp

    return {
        "lat_centres": lat_centres,
        "u_bar": u_bar,
        "theta_bar": theta_bar,
        "dtheta_dp": dtheta_dp,
        "F_phi": f_phi,
        "F_p": f_p,
        "div_F": div_f_accel,
        "v_star": v_star,
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def _tem_stream_levels(channels: list[str]) -> dict[str, dict[float, str]]:
    """
    ``{variable: {pressure_hPa: channel}}`` for the TEM stream, whose channel
    suffix is an L137 model level (``ep_flux_div_61`` → 104 hPa).
    """
    out: dict[str, dict[float, str]] = {}
    for ch in channels:
        var, lv = parse_channel(ch)
        if var not in _TEM_VARIABLES or lv is None:
            continue
        p = channel_pressure(ch, "model")
        if p is None:
            _logger.warning("TEM channel %s: unknown model level %s — skipped", ch, lv)
            continue
        out.setdefault(var, {})[p] = ch
    return {v: dict(sorted(m.items())) for v, m in out.items()}


def _extract_tem_stream(
    zio: Any,
    steps: list[int],
    lat_width: float,
    sample: int,
    label: str,
) -> dict[str, Any] | None:
    """
    Zonal-mean the predicted / target channels of the ``TEM`` stream.

    Returns ``None`` when the store has no TEM stream, otherwise a dict
    ``{"pressures_hpa": {var: (n_lev,)}, "pred": {var: (n_t, n_lats, n_lev)},
    "tgt": {...}}``.  The stream carries zonal means broadcast to every grid
    point, so the latitude-bin mean just recovers the value of the ring.
    """
    if _TEM_STREAM not in get_streams(zio):
        return None
    channels = get_channels(zio, _TEM_STREAM, sample)
    levels = _tem_stream_levels(channels)
    if not levels:
        _logger.warning("%s: TEM stream has no recognised channels (%s)", label, channels)
        return None

    # Bin on the TEM stream's own grid (same bins as the diagnostics).
    tem_coords = get_coords(zio, _TEM_STREAM, sample)
    edges, _ = _build_lat_bins(lat_width)
    tem_groups = _build_lat_groups(tem_coords[:, 0], edges)
    n_lats = len(tem_groups)

    pred_acc = {v: np.full((len(steps), n_lats, len(m)), np.nan) for v, m in levels.items()}
    tgt_acc = {v: np.full((len(steps), n_lats, len(m)), np.nan) for v, m in levels.items()}
    idx = {v: [channels.index(ch) for ch in m.values()] for v, m in levels.items()}

    for i, step in enumerate(steps):
        pred, tgt, _ = load_step(zio, _TEM_STREAM, step, sample)
        pred3 = np.atleast_3d(pred).astype(np.float64)
        tgt3 = np.atleast_3d(tgt).astype(np.float64)
        for v in levels:
            pred_acc[v][i] = _zonal_mean_field(pred3[:, idx[v], :].mean(axis=-1), tem_groups)
            tgt_acc[v][i] = _zonal_mean_field(tgt3[:, idx[v], :].mean(axis=-1), tem_groups)

    _logger.info(
        "%s: TEM stream with %s",
        label,
        {v: [f"{p:.3g}" for p in m] for v, m in levels.items()},
    )
    return {
        "pressures_hpa": {v: np.array(list(m), dtype=np.float64) for v, m in levels.items()},
        "pred": pred_acc,
        "tgt": tgt_acc,
    }


def extract_qg_tem(
    zarr_path: Path,
    label: str,
    sample: int = 0,
    lat_width: float = 2.5,
    lead_days: int = 0,
    color: str = "C0",
    ssw_date: datetime = SSW_DATE,
    p_top: float = _AXIS_P_TOP,
    p_bottom: float = _AXIS_P_BOTTOM,
) -> dict[str, Any] | None:
    """
    Extract and compute QG-TEM diagnostics for every forecast step.

    The pressure axis is the merged set of levels that u, v and t share
    across *all* streams of the store (``ERA5pl`` + ``S2Spredictors`` ...),
    restricted to ``[p_top, p_bottom]`` hPa and sorted top → bottom.

    Returns a dict with keys:
        ``label``, ``lead_days``, ``color``, ``datetimes``, ``days_rel``,
        ``pressures_hpa``, ``lat_centres``, ``streams``,
        ``pred``: dict of (n_time, n_lats, n_levels) arrays,
        ``tgt``:  dict of (n_time, n_lats, n_levels) arrays,
        ``tem``:  output of :func:`_extract_tem_stream` or ``None``.
    ``pred`` and ``tgt`` each contain: ``div_F``, ``v_star``, ``F_phi``,
    ``F_p``, ``u_bar``.  ``days_rel`` is a float array of days relative to
    *ssw_date* (negative = before onset, 0 = onset, positive = after).
    """
    _logger.info("%s: extracting QG-TEM diagnostics …", label)

    with open_validation(zarr_path) as zio:
        cols = discover_columns(zio, ["u", "v", "t"], sample)
        pressures_hpa_list = [
            p for p in common_pressures(cols["u"], cols["v"], cols["t"]) if p_top <= p <= p_bottom
        ]
        if len(pressures_hpa_list) < 3:
            _logger.warning(
                "%s: fewer than 3 common u/v/t levels in %g–%g hPa — cannot compute "
                "vertical derivatives; skipping.  u: %s  v: %s  t: %s",
                label,
                p_top,
                p_bottom,
                cols["u"].pressures,
                cols["v"].pressures,
                cols["t"].pressures,
            )
            return None

        refs: dict[str, list[LevelRef]] = {
            var: [cols[var].nearest(p) for p in pressures_hpa_list] for var in ("u", "v", "t")
        }
        # Pressures straight from the (u) LevelRefs, sorted top → bottom.
        pressures_hpa = np.array([r.pressure for r in refs["u"]], dtype=np.float64)
        order = np.argsort(pressures_hpa)
        pressures_hpa = pressures_hpa[order]
        refs = {var: [rs[i] for i in order] for var, rs in refs.items()}
        pressures_pa = pressures_hpa * 100.0
        streams = sorted({r.stream for rs in refs.values() for r in rs})

        coords = get_coords(zio, refs["u"][0].stream, sample)
        lats = coords[:, 0]

        steps = get_forecast_steps(zio, skip_source_step=True)
        _, lat_centres = _build_lat_bins(lat_width)

        n_t = len(steps)
        n_lats = len(lat_centres)
        n_lev = len(pressures_hpa)

        # Accumulate per-step diagnostics
        keys = ("div_F", "v_star", "F_phi", "F_p", "u_bar")
        pred_acc: dict[str, NDArray] = {k: np.full((n_t, n_lats, n_lev), np.nan) for k in keys}
        tgt_acc: dict[str, NDArray] = {k: np.full((n_t, n_lats, n_lev), np.nan) for k in keys}
        raw_times: list = []

        for i, step in enumerate(steps):
            fields: dict[str, tuple[NDArray, NDArray]] = {}
            for var in ("u", "v", "t"):
                pred, tgt, times = load_levels(zio, refs[var], step, sample)
                # (n_pts, n_lev, n_ens) → ensemble member 0, as before
                fields[var] = (
                    np.atleast_3d(pred)[:, :, 0].astype(np.float64),
                    np.atleast_3d(tgt)[:, :, 0].astype(np.float64),
                )

            for which, acc in ((0, pred_acc), (1, tgt_acc)):
                diags = compute_qg_tem(
                    fields["u"][which],
                    fields["v"][which],
                    fields["t"][which],
                    lats,
                    pressures_pa,
                    lat_width,
                )
                for k in keys:
                    acc[k][i] = diags[k]

            raw_times.append(times[0])

        tem = _extract_tem_stream(zio, steps, lat_width, sample, label)

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info(
        "%s: %d steps, %d lat bands, %d levels (%g–%g hPa) from streams %s",
        label,
        n_t,
        n_lats,
        n_lev,
        pressures_hpa[0],
        pressures_hpa[-1],
        streams,
    )

    days_rel = np.array(
        [(dt - ssw_date).total_seconds() / 86400.0 for dt in datetimes],
        dtype=np.float64,
    )

    return {
        "label": label,
        "lead_days": lead_days,
        "color": color,
        "datetimes": datetimes,
        "days_rel": days_rel,
        "pressures_hpa": pressures_hpa,
        "lat_centres": lat_centres,
        "streams": streams,
        "pred": pred_acc,
        "tgt": tgt_acc,
        "tem": tem,
    }


def _smooth(field: NDArray, sigma: float = 1.5) -> NDArray:
    """Apply 2-D Gaussian smoothing, ignoring NaNs by normalised convolution."""
    out = np.empty_like(field)
    valid = np.isfinite(field).astype(float)
    filled = np.where(np.isfinite(field), field, 0.0)
    blurred = gaussian_filter(filled, sigma=sigma)
    weight = gaussian_filter(valid, sigma=sigma)
    with np.errstate(invalid="ignore"):
        out = np.where(weight > 0.1, blurred / weight, np.nan)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _pressure_axis(
    ax: Any,
    p_range: tuple[float, float] = (0.01, 1000.0),
) -> None:
    ax.set_yscale("log")
    # NOTE: invert_yaxis() is NOT called here because with sharey=True a second
    # call would un-invert.  Each plot function sets ylim explicitly instead.
    visible_ticks = [p for p in _P_TICKS if p_range[0] <= p <= p_range[1]]
    ax.set_yticks(visible_ticks)
    ax.set_yticklabels([str(p) for p in visible_ticks])
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")


def _mark_onset(ax: Any, days_rel: NDArray) -> None:
    """Draw a vertical line at day 0 (SSW onset) if it falls within the data range."""
    if days_rel[0] <= 0.0 <= days_rel[-1]:
        ax.axvline(
            0.0,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="SSW onset (day 0)",
        )


def _day_ticks(ax: Any, days_rel: NDArray, every: int = 5) -> None:
    """Set x-ticks at round multiples of *every* days within the data range."""
    lo = int(np.ceil(days_rel[0] / every)) * every
    hi = int(np.floor(days_rel[-1] / every)) * every
    ticks = np.arange(lo, hi + 1, every)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=9)
    ax.set_xlabel("Days relative to SSW onset", fontsize=11)


def _clip_p_range(pressures: NDArray, p_range: tuple[float, float]) -> tuple[float, float] | None:
    """Restrict *p_range* to the data's pressure span; ``None`` if disjoint."""
    lo = max(p_range[0], float(pressures.min()))
    hi = min(p_range[1], float(pressures.max()))
    if lo >= hi:
        _logger.warning(
            "requested pressure range %s hPa has no overlap with data (%g–%g hPa)",
            p_range,
            pressures.min(),
            pressures.max(),
        )
        return None
    return lo, hi


# ---------------------------------------------------------------------------
# Plot: EP flux divergence Hovmöller
# ---------------------------------------------------------------------------


def plot_ep_flux_divergence_hovmoller(
    data: dict[str, Any],
    output_dir: Path,
    lat_range: tuple[float, float] = (20.0, 90.0),
    clim: float = 5.0,
    p_range: tuple[float, float] = (1.0, 100.0),
) -> None:
    """
    Hovmöller (time × pressure) of EP flux divergence at a fixed latitude band.

    A 2-column figure is produced: prediction (left) and ERA5 target (right).
    """
    label = data["label"]
    days_rel = data["days_rel"]
    pressures = data["pressures_hpa"]
    lat_centres = data["lat_centres"]

    # Average over latitude band
    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    if not lat_mask.any():
        _logger.warning("No latitude bins in range %s — skipping Hovmöller", lat_range)
        return
    p_range = _clip_p_range(pressures, p_range) or p_range

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)
    cmap = "RdBu_r"

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target"), strict=True):
        # Average ∇·F over the latitude band: (n_time, n_lev)
        field = np.nanmean(data[key]["div_F"][:, lat_mask, :], axis=1)
        # Smooth in time × pressure space to suppress grid-scale noise
        field = _smooth(field, sigma=1.5)
        im = ax.contourf(
            days_rel,
            pressures,
            field.T,
            levels=np.linspace(-clim, clim, 21),
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        ax.contour(
            days_rel,
            pressures,
            field.T,
            levels=[0.0],
            colors="k",
            linewidths=0.8,
        )
        _pressure_axis(ax, p_range=p_range)
        _mark_onset(ax, days_rel)
        _day_ticks(ax, days_rel)
        ax.set_title(
            f"{label}  {title}\n∇·F ({lat_range[0]:.0f}–{lat_range[1]:.0f}°N mean)",
            fontsize=12,
            fontweight="bold",
        )

    axes[0].set_ylim(p_range[1], p_range[0])  # high p at bottom, low p at top
    cbar = fig.colorbar(
        im, ax=axes.tolist(), orientation="horizontal", pad=0.18, shrink=0.7, aspect=40
    )
    cbar.set_label("∇·F  (m s⁻¹ day⁻¹)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = output_dir / f"{label}_ep_flux_divergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: time-mean EP flux vectors + divergence
# ---------------------------------------------------------------------------


def plot_ep_flux_vectors(
    data: dict[str, Any],
    output_dir: Path,
    clim: float = 3.0,
    lat_range: tuple[float, float] = (0.0, 90.0),
    p_range: tuple[float, float] = (1.0, 100.0),
) -> None:
    """
    Time-mean EP flux vectors (F_φ, F_p) overlaid on ∇·F shading.

    The two EP-flux components live in very different physical spaces
    (F_φ in m²/s², F_p in m·Pa/s²), so they are independently normalised to
    their 95th-percentile absolute value before being passed to quiver.  This
    gives arrows whose *direction* is meaningful while the *length* indicates
    relative local magnitude.  A reference arrow of length 1 is drawn in the
    corner.

    A 1×2 panel: prediction (left) and target (right).
    """
    label = data["label"]
    lat_centres = data["lat_centres"]
    pressures = data["pressures_hpa"]
    p_range = _clip_p_range(pressures, p_range) or p_range

    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    lats_plot = lat_centres[lat_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)
    cmap = "RdBu_r"

    # Quiver subsampling: ~15 arrows in each dimension
    lat_skip = max(1, lat_mask.sum() // 15)
    lev_skip = max(1, len(pressures) // 10)
    lat_q = lats_plot[::lat_skip]
    p_q = pressures[::lev_skip]

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target"), strict=True):
        # Time-mean then smooth
        div_f_mean = _smooth(np.nanmean(data[key]["div_F"][:, lat_mask, :], axis=0))
        f_phi_mean = _smooth(np.nanmean(data[key]["F_phi"][:, lat_mask, :], axis=0))
        f_p_mean = _smooth(np.nanmean(data[key]["F_p"][:, lat_mask, :], axis=0))
        u_bar_mean = _smooth(np.nanmean(data[key]["u_bar"][:, lat_mask, :], axis=0))

        im = ax.contourf(
            lats_plot,
            pressures,
            div_f_mean.T,
            levels=np.linspace(-clim, clim, 21),
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        ax.contour(lats_plot, pressures, div_f_mean.T, levels=[0.0], colors="k", linewidths=0.8)

        # Overlay zonal mean wind contours (thin grey dashed)
        u_levels = np.arange(-80, 90, 10)
        cs = ax.contour(
            lats_plot,
            pressures,
            u_bar_mean.T,
            levels=u_levels,
            colors="grey",
            linewidths=0.6,
            linestyles="--",
            alpha=0.6,
        )
        label_levels = [lv for lv in u_levels if lv % 20 == 0 and lv in cs.levels]
        if label_levels:
            ax.clabel(cs, levels=label_levels, inline=True, fontsize=7, fmt="%d")

        # EP flux arrows: normalise each component by its 95th-percentile
        # absolute value so arrows show direction+relative magnitude, not
        # absolute magnitude (which differs by orders of magnitude between
        # F_φ and F_p).
        f_q = f_phi_mean[::lat_skip, :][:, ::lev_skip]
        g_q = f_p_mean[::lat_skip, :][:, ::lev_skip]
        ref_f = np.nanpercentile(np.abs(f_phi_mean), 95) or 1.0
        ref_g = np.nanpercentile(np.abs(f_p_mean), 95) or 1.0
        if not np.isfinite(ref_f):
            ref_f = 1.0
        if not np.isfinite(ref_g):
            ref_g = 1.0
        u_arrow = f_q / ref_f
        # F_p > 0 means downward (increasing p); flip sign so arrow points upward
        v_arrow = -(g_q / ref_g)
        ax.quiver(
            lat_q,
            p_q,
            u_arrow.T,
            v_arrow.T,
            scale=15,
            width=0.004,
            color="k",
            alpha=0.8,
            headwidth=4,
            headlength=5,
        )

        _pressure_axis(ax, p_range=p_range)
        ax.set_xlim(lat_range)
        ax.set_xlabel("Latitude (°N)", fontsize=11)
        ax.set_title(
            f"{label}  {title}\nTime-mean EP flux  (arrows: direction, grey = ū contours)",
            fontsize=12,
            fontweight="bold",
        )

    axes[0].set_ylim(p_range[1], p_range[0])  # high p at bottom, low p at top
    cbar = fig.colorbar(
        im, ax=axes.tolist(), orientation="horizontal", pad=0.08, shrink=0.7, aspect=40
    )
    cbar.set_label("∇·F  (m s⁻¹ day⁻¹)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = output_dir / f"{label}_ep_flux_vectors.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: residual meridional velocity Hovmöller
# ---------------------------------------------------------------------------


def plot_v_residual_hovmoller(
    data: dict[str, Any],
    output_dir: Path,
    lat_range: tuple[float, float] = (50.0, 80.0),
    clim: float = 0.3,
    p_range: tuple[float, float] = (1.0, 100.0),
) -> None:
    """
    Hovmöller (time × pressure) of residual meridional velocity v*.
    """
    label = data["label"]
    pressures = data["pressures_hpa"]
    lat_centres = data["lat_centres"]

    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    if not lat_mask.any():
        _logger.warning("No latitude bins in range %s for v* Hovmöller", lat_range)
        return
    p_range = _clip_p_range(pressures, p_range) or p_range

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    days_rel = data["days_rel"]

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target"), strict=True):
        field = np.nanmean(data[key]["v_star"][:, lat_mask, :], axis=1)
        field = _smooth(field, sigma=1.5)
        im = ax.contourf(
            days_rel,
            pressures,
            field.T,
            levels=np.linspace(-clim, clim, 21),
            cmap="RdBu_r",
            norm=norm,
            extend="both",
        )
        ax.contour(
            days_rel,
            pressures,
            field.T,
            levels=[0.0],
            colors="k",
            linewidths=0.8,
        )
        _pressure_axis(ax, p_range=p_range)
        _mark_onset(ax, days_rel)
        _day_ticks(ax, days_rel)
        ax.set_title(
            f"{label}  {title}\nv* residual ({lat_range[0]:.0f}–{lat_range[1]:.0f}°N mean)",
            fontsize=12,
            fontweight="bold",
        )

    axes[0].set_ylim(p_range[1], p_range[0])  # high p at bottom, low p at top
    cbar = fig.colorbar(
        im, ax=axes.tolist(), orientation="horizontal", pad=0.18, shrink=0.7, aspect=40
    )
    cbar.set_label("v*  (m s⁻¹)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = output_dir / f"{label}_v_residual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: polar-cap integrated EP flux divergence time series
# ---------------------------------------------------------------------------


def plot_polar_cap_efd_timeseries(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
    polar_lat_min: float = 60.0,
    pressure_range_hpa: tuple[float, float] = (1.0, 100.0),
) -> None:
    """
    Time series of polar-cap, pressure-layer-mean ∇·F for all experiments.

    Averages ∇·F over latitudes ≥ *polar_lat_min* and over the specified
    pressure range, yielding one scalar per time step per run.
    """
    if not data_list:
        return

    fig, ax = plt.subplots(figsize=(13, 5))

    # Draw ERA5 target once using the run with the longest precursor window
    # (smallest days_rel[0], i.e. earliest initialization).
    era5_drawn = False
    data_sorted = sorted(data_list, key=lambda d: d["days_rel"][0])

    for data in data_sorted:
        days_rel = data["days_rel"]
        lat_centres = data["lat_centres"]
        pressures = data["pressures_hpa"]
        color = data["color"]
        lead_days = data["lead_days"]

        lat_mask = lat_centres >= polar_lat_min
        p_mask = (pressures >= pressure_range_hpa[0]) & (pressures <= pressure_range_hpa[1])

        if not lat_mask.any() or not p_mask.any():
            _logger.warning(
                "%s: no levels in %s hPa (have %s) — skipped in polar-cap series",
                data["label"],
                pressure_range_hpa,
                pressures,
            )
            continue

        # Prediction line — colored by lead time
        pred_series = np.nanmean(
            data["pred"]["div_F"][:, lat_mask, :][:, :, p_mask],
            axis=(1, 2),
        )
        ax.plot(
            days_rel,
            pred_series,
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.85,
            label=f"t{lead_days:02d}d pred",
        )

        # ERA5 target — draw only once (all runs share the same ERA5 reality;
        # use the longest run to cover the widest window).
        if not era5_drawn:
            tgt_series = np.nanmean(
                data["tgt"]["div_F"][:, lat_mask, :][:, :, p_mask],
                axis=(1, 2),
            )
            ax.plot(
                days_rel,
                tgt_series,
                color="k",
                linestyle="-",
                linewidth=2.5,
                alpha=0.9,
                label="ERA5",
                zorder=10,
            )
            era5_drawn = True

    ax.axvline(
        0.0,
        color="red",
        linestyle="-.",
        linewidth=1.8,
        alpha=0.7,
        label="SSW onset (day 0)",
    )
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Days relative to SSW onset", fontsize=12)
    ax.set_ylabel("∇·F  (m s⁻¹ day⁻¹)", fontsize=12)
    ax.set_title(
        f"Polar cap (≥{polar_lat_min:.0f}°N, "
        f"{pressure_range_hpa[0]:.0f}–{pressure_range_hpa[1]:.0f} hPa) EP flux divergence",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / "polar_cap_efd_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: TEM stream (model-predicted ∇·F) vs diagnosed ∇·F
# ---------------------------------------------------------------------------


def _nearest_index(pressures: NDArray, p: float) -> int:
    """Index of the level closest to *p* in log-pressure."""
    return int(np.argmin(np.abs(np.log(pressures) - math.log(p))))


def _polar_cap_series(field: NDArray, lat_mask: NDArray, k: int) -> NDArray:
    """Polar-cap mean of ``field[:, lats, k]`` → ``(n_time,)``."""
    return np.nanmean(field[:, lat_mask, k], axis=1)


def plot_tem_stream_vs_diagnosed(
    data_list: list[dict[str, Any]],
    output_dir: Path,
    event_name: str,
    polar_lat_min: float = 60.0,
) -> None:
    """
    Polar-cap mean EP-flux diagnostics of the ``TEM`` stream versus the
    diagnosed ones.

    Left column, one row per TEM ``ep_flux_div`` level: TEM-stream prediction
    (per run, solid) and target (black) overlaid with the diagnosed ∇·F at
    the nearest pressure (dotted) — all in m s⁻¹ day⁻¹ (:data:`TEM_STREAM_UNITS`).
    Right column: ``F_p`` and ``vtheta_prime`` of the TEM stream (pred vs
    target) in their own units; these are not compared with the diagnosed
    F_p because the stream's F_p convention (K m s⁻¹ Pa⁻¹) differs from the
    m Pa s⁻² used here.  Runs without a TEM stream are ignored; nothing is
    drawn when no run has one.
    """
    runs = [d for d in data_list if d.get("tem") and "ep_flux_div" in d["tem"]["pred"]]
    if not runs:
        _logger.debug("[%s] no TEM stream in any run — no TEM comparison figure", event_name)
        return

    runs_sorted = sorted(runs, key=lambda d: d["days_rel"][0])
    tem0 = runs_sorted[0]["tem"]
    efd_pressures = tem0["pressures_hpa"]["ep_flux_div"]
    # Extra panels: (variable, pressure) for F_p / vtheta_prime
    extra = [
        (v, p)
        for v in ("F_p", "vtheta_prime")
        if v in tem0["pred"]
        for p in tem0["pressures_hpa"][v]
    ]
    n_rows = max(len(efd_pressures), len(extra), 1)
    n_cols = 2 if extra else 1
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 2.6 * n_rows + 1), sharex=True, squeeze=False
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    # --- left column: ∇·F, TEM stream vs diagnosed on one axis --------------
    for row, p_tem in enumerate(efd_pressures):
        ax = axes[row, 0]
        ax.set_visible(True)
        tgt_drawn = False
        p_diag = None
        for data in runs_sorted:
            tem = data["tem"]
            days_rel = data["days_rel"]
            lat_mask = data["lat_centres"] >= polar_lat_min
            color, lead = data["color"], data["lead_days"]
            k_tem = _nearest_index(tem["pressures_hpa"]["ep_flux_div"], p_tem)
            k_diag = _nearest_index(data["pressures_hpa"], p_tem)
            p_diag = data["pressures_hpa"][k_diag]

            ax.plot(
                days_rel,
                _polar_cap_series(tem["pred"]["ep_flux_div"], lat_mask, k_tem),
                color=color,
                lw=1.5,
                label=f"t{lead:02d}d TEM pred",
            )
            ax.plot(
                days_rel,
                _polar_cap_series(data["pred"]["div_F"], lat_mask, k_diag),
                color=color,
                lw=1.2,
                ls=":",
                label=f"t{lead:02d}d diagnosed",
            )
            if not tgt_drawn:
                ax.plot(
                    days_rel,
                    _polar_cap_series(tem["tgt"]["ep_flux_div"], lat_mask, k_tem),
                    color="k",
                    lw=2.2,
                    label="TEM target",
                    zorder=10,
                )
                ax.plot(
                    days_rel,
                    _polar_cap_series(data["tgt"]["div_F"], lat_mask, k_diag),
                    color="k",
                    lw=1.6,
                    ls=":",
                    label="diagnosed ERA5",
                    zorder=10,
                )
                tgt_drawn = True
        ax.set_title(
            f"∇·F  TEM {p_tem:.3g} hPa  vs  diagnosed {p_diag:.3g} hPa",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel(f"∇·F ({TEM_STREAM_UNITS['ep_flux_div']})", fontsize=9)
    axes[0, 0].legend(loc="best", fontsize=7, ncol=2)

    # --- right column: F_p and v′θ′ of the TEM stream -----------------------
    for row, (var, p_var) in enumerate(extra):
        ax = axes[row, 1]
        ax.set_visible(True)
        tgt_drawn = False
        for data in runs_sorted:
            tem = data["tem"]
            if var not in tem["pred"]:
                continue
            lat_mask = data["lat_centres"] >= polar_lat_min
            k = _nearest_index(tem["pressures_hpa"][var], p_var)
            ax.plot(
                data["days_rel"],
                _polar_cap_series(tem["pred"][var], lat_mask, k),
                color=data["color"],
                lw=1.5,
                label=f"t{data['lead_days']:02d}d TEM pred",
            )
            if not tgt_drawn:
                ax.plot(
                    data["days_rel"],
                    _polar_cap_series(tem["tgt"][var], lat_mask, k),
                    color="k",
                    lw=2.2,
                    label="TEM target",
                    zorder=10,
                )
                tgt_drawn = True
        ax.set_title(f"{_TEM_TITLES[var]}  {p_var:.3g} hPa", fontsize=10, fontweight="bold")
        ax.set_ylabel(f"{var} ({TEM_STREAM_UNITS[var]})", fontsize=9)
    if extra:
        axes[0, 1].legend(loc="best", fontsize=7, ncol=2)

    for ax in axes.ravel():
        if not ax.get_visible():
            continue
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax.axvline(0.0, color="red", ls="-.", lw=1.2, alpha=0.7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Days relative to SSW onset", fontsize=9)
    fig.suptitle(
        f"{event_name}: TEM stream (model-predicted) vs diagnosed EP-flux diagnostics, "
        f"polar cap ≥{polar_lat_min:.0f}°N",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out = output_dir / f"tem_stream_vs_diagnosed_{event_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    _logger.info("Saved %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="QG-TEM diagnostics (EP flux, ∇·F, v*) for SSW analysis."
    )
    parser.add_argument(
        "--validations-config",
        type=Path,
        required=True,
        help="YAML validations config (see config/evaluate/).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing zarr result stores.  [default: results]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/qg_tem"),
        help="Directory for output plots and archives.  [default: plots/qg_tem]",
    )
    parser.add_argument(
        "--lat-width",
        type=float,
        default=2.5,
        help="Latitude bin width (°) for zonal mean.  [default: 2.5]",
    )
    parser.add_argument(
        "--clim-divF",
        type=float,
        default=5.0,
        help="Colour-scale limit for ∇·F Hovmöller (m s⁻¹ day⁻¹).  [default: 5]",
    )
    parser.add_argument(
        "--clim-vstar",
        type=float,
        default=0.3,
        help="Colour-scale limit for v* Hovmöller (m s⁻¹).  [default: 0.3]",
    )
    parser.add_argument(
        "--polar-lat-min",
        type=float,
        default=60.0,
        help="Minimum latitude for polar-cap EFD time series.  [default: 60]",
    )
    parser.add_argument(
        "--ssw-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=SSW_DATE,
        help="SSW onset date for reference lines (YYYY-MM-DD).  [default: 2018-02-12]",
    )
    parser.add_argument(
        "--p-range",
        type=float,
        nargs=2,
        metavar=("P_MIN", "P_MAX"),
        default=[1.0, 100.0],
        help="Pressure axis range in hPa for the Hovmöller/vector plots (min max).  "
        "The diagnostics themselves are computed on all common u/v/t levels "
        f"between {_AXIS_P_TOP:g} and {_AXIS_P_BOTTOM:g} hPa.  [default: 1 100]",
    )
    args = parser.parse_args(argv)

    _logger.info("=" * 60)
    _logger.info("QG-TEM ANALYSIS")
    _logger.info("=" * 60)

    cfg = load_validations_config(args.validations_config)
    events = group_validations_by_event(cfg)
    single_event = len(events) == 1

    for event_name, event_cfg in events.items():
        event_ssw_date = event_reference_date(event_cfg, args.ssw_date)
        event_output_dir = args.output_dir if single_event else args.output_dir / event_name
        _logger.info(
            "[%s] %d run(s), onset %s → %s",
            event_name,
            len(event_cfg),
            event_ssw_date.date(),
            event_output_dir,
        )

        run_specs = [
            (
                label,
                spec["id"],
                spec.get("sample", 0),
                spec.get("rank", 0),
                spec.get("lead_days", 0),
                spec.get("color", "C0"),
            )
            for label, spec in event_cfg.items()
        ]

        data_list: list[dict[str, Any]] = []

        # Phase 1: Extract all diagnostics (each run keeps its own onset-relative axis)
        for label, run_id, sample, rank, lead_days, color in run_specs:
            zarr_path = zarr_store_path(args.data_dir, run_id, rank)

            if not zarr_path.exists():
                _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
                continue

            data = extract_qg_tem(
                zarr_path,
                label,
                sample,
                lat_width=args.lat_width,
                lead_days=lead_days if lead_days is not None else 0,
                color=color or "C0",
                ssw_date=event_ssw_date,
            )
            if data is None:
                _logger.warning("Skipping %s — extraction failed.", label)
                continue

            data_list.append(data)

        if not data_list:
            _logger.warning("[%s] No data extracted — skipping event.", event_name)
            continue

        # Phase 2: Per-experiment plots (each on its own onset-relative time axis)
        for data in data_list:
            label = data["label"]
            exp_dir = event_output_dir / label
            exp_dir.mkdir(parents=True, exist_ok=True)

            p_range = tuple(args.p_range)  # type: ignore[arg-type]
            plot_ep_flux_divergence_hovmoller(
                data,
                exp_dir,
                clim=args.clim_divF,
                p_range=p_range,
            )
            plot_ep_flux_vectors(data, exp_dir, clim=args.clim_divF, p_range=p_range)
            plot_v_residual_hovmoller(data, exp_dir, clim=args.clim_vstar, p_range=p_range)

            # Save numpy archive
            npz_path = exp_dir / "qg_tem_diagnostics.npz"
            tem_arrays: dict[str, NDArray] = {}
            if data["tem"]:
                for v, p in data["tem"]["pressures_hpa"].items():
                    tem_arrays[f"tem_pressures_hpa_{v}"] = p
                    tem_arrays[f"tem_pred_{v}"] = data["tem"]["pred"][v]
                    tem_arrays[f"tem_tgt_{v}"] = data["tem"]["tgt"][v]
            np.savez_compressed(
                npz_path,
                lat_centres=data["lat_centres"],
                pressures_hpa=data["pressures_hpa"],
                **{f"pred_{k}": v for k, v in data["pred"].items()},
                **{f"tgt_{k}": v for k, v in data["tgt"].items()},
                **tem_arrays,
            )
            _logger.info("Saved %s", npz_path)

        # Phase 3: Multi-experiment time series (per event, to stay readable)
        event_output_dir.mkdir(parents=True, exist_ok=True)
        plot_polar_cap_efd_timeseries(
            data_list,
            event_output_dir,
            event_ssw_date,
            args.polar_lat_min,
        )
        # Phase 4: TEM stream vs diagnosed (only when a TEM stream exists)
        plot_tem_stream_vs_diagnosed(
            data_list, event_output_dir, event_name, polar_lat_min=args.polar_lat_min
        )

    _logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
