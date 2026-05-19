#!/usr/bin/env python3
# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
Quasi-Geostrophic Transformed Eulerian Mean (QG-TEM) Analysis.

Computes the QG-TEM diagnostics from u, v, T fields on stratospheric model
levels (IFS L137 levels where B_k ≈ 0, i.e. pure pressure levels):

  * Potential temperature  θ = T · (p₀/p)^κ
  * EP flux components     F_φ = −u′v′̄                          [m² s⁻²]
                           F_p =  f · v′θ′̄ / (∂θ̄/∂p)           [m Pa s⁻²]
  * EP flux divergence     ∇·F = (1/a cosφ)·∂(cosφ F_φ)/∂φ + ∂F_p/∂p  [m s⁻²]
  * Residual meridional    v* = v̄ − ∂/∂p(v′θ′̄ / ∂θ̄/∂p)
    velocity

Because WeatherGenerator uses an unstructured grid, zonal means are computed
by grouping grid points into latitude bands of configurable width and averaging
within each band.

Outputs (per experiment, saved to ``--output-dir/<label>/``):

  * ``ep_flux_divergence.png``  — latitude–pressure Hovmöller of ∇·F
  * ``ep_flux_vectors.png``     — quiver plot of EP flux + divergence shading
                                  (time-mean over entire rollout)
  * ``v_residual.png``          — Hovmöller of residual circulation v*
  * ``polar_cap_efd.png``       — polar-cap (60–90°N) mean ∇·F time series
  * ``qg_tem_diagnostics.npz``  — numpy archive of all computed fields

Usage::

    ssw-analyze qg-tem \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results \\
        --output-dir plots/qg_tem

Or as a module::

    python -m weathergen.stratosphere.scripts.analyze_qg_tem \\
        --validations-config config/evaluate/ssw_feb2018.yml \\
        --data-dir results
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from weathergen.stratosphere.config import load_validations_config
from weathergen.stratosphere.io import (
    convert_times_to_datetime,
    get_channels,
    get_coords,
    get_forecast_steps,
    get_stream,
    load_step,
    open_validation,
)
from weathergen.stratosphere.levels import build_full_level_map

_logger = logging.getLogger(__name__)

_ZARR_FNAME = "validation_chkpt00000_rank0000.zip"

# Physical constants
_EARTH_RADIUS = 6.371e6        # m
_OMEGA = 7.292e-5              # rad s⁻¹
_P0_PA = 1.0e5                 # Pa  (reference pressure for θ)
_KAPPA = 2.0 / 7.0             # R/cp for dry air

# Default SSW reference date
SSW_DATE = datetime(2018, 2, 12)

# Standard pressure ticks for log-pressure y-axis
_P_TICKS = [1, 3, 10, 30, 100]
_P_TICK_LABELS = [str(p) for p in _P_TICKS]


# ---------------------------------------------------------------------------
# Zonal-mean helper for unstructured grids
# ---------------------------------------------------------------------------


def _build_lat_bins(lat_width: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (edges, centres) for latitude bins spanning −90 to 90."""
    edges = np.arange(-90.0, 90.0 + lat_width, lat_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return edges, centres


def _build_lat_groups(
    lats: np.ndarray,
    edges: np.ndarray,
) -> list[np.ndarray]:
    """
    For each latitude bin return the indices of grid points that fall in it.
    """
    groups: list[np.ndarray] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((lats >= lo) & (lats < hi))[0]
        groups.append(idx)
    return groups


def _zonal_mean_field(
    data: np.ndarray,
    groups: list[np.ndarray],
) -> np.ndarray:
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
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    lats: np.ndarray,
    pressures_pa: np.ndarray,
    lat_width: float = 2.5,
) -> dict[str, np.ndarray]:
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
        Full-level pressure at each model level in **Pa**.  Shape ``(n_levels,)``.
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
        ``div_F``: EP flux divergence            [m s⁻¹ day⁻¹]  (negative = westward/easterly forcing)
        ``v_star``: residual meridional velocity [m s⁻¹]
    """
    edges, lat_centres = _build_lat_bins(lat_width)
    groups = _build_lat_groups(lats, edges)
    phi = np.deg2rad(lat_centres)                          # (n_lats,)
    f = 2.0 * _OMEGA * np.sin(phi)                        # Coriolis (n_lats,)
    cos_phi = np.cos(phi)                                   # (n_lats,)

    n_lats = len(lat_centres)
    n_lev = len(pressures_pa)

    # ---- potential temperature ------------------------------------------------
    theta = t * (_P0_PA / pressures_pa[None, :]) ** _KAPPA   # (n_pts, n_lev)

    # ---- zonal means ---------------------------------------------------------
    u_bar = _zonal_mean_field(u, groups)        # (n_lats, n_lev)
    v_bar = _zonal_mean_field(v, groups)
    theta_bar = _zonal_mean_field(theta, groups)

    # ---- eddy covariances u′v′ and v′θ′ --------------------------------------
    uv_bar = np.full((n_lats, n_lev), np.nan, dtype=np.float64)
    vtheta_bar = np.full((n_lats, n_lev), np.nan, dtype=np.float64)

    for i, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        u_prime = u[idx] - u_bar[i]               # (n_grp, n_lev)
        v_prime = v[idx] - v_bar[i]
        theta_prime = theta[idx] - theta_bar[i]
        uv_bar[i] = (u_prime * v_prime).mean(axis=0)
        vtheta_bar[i] = (v_prime * theta_prime).mean(axis=0)

    # ---- ∂θ̄/∂p — centred finite differences in pressure space ---------------
    dtheta_dp = np.full_like(theta_bar, np.nan)
    dp = pressures_pa[1:] - pressures_pa[:-1]     # (n_lev-1,) positive (ascending p)
    dtheta_dp[:, 1:-1] = (
        (theta_bar[:, 2:] - theta_bar[:, :-2])
        / (pressures_pa[2:] - pressures_pa[:-2])[None, :]
    )
    dtheta_dp[:, 0] = (theta_bar[:, 1] - theta_bar[:, 0]) / dp[0]
    dtheta_dp[:, -1] = (theta_bar[:, -1] - theta_bar[:, -2]) / dp[-1]
    # In a stable atmosphere ∂θ/∂p < 0; guard against near-zero values
    dtheta_dp_safe = np.where(np.abs(dtheta_dp) > 1e-6, dtheta_dp, np.nan)

    # ---- EP flux components --------------------------------------------------
    # Correct pressure-coordinate QG-TEM definitions (no a cosφ factor here;
    # that factor belongs only in the spherical divergence operator below).
    #
    # F_φ = −u′v′   [m² s⁻²]
    F_phi = -uv_bar

    # F_p = f · v′θ′ / (∂θ̄/∂p)   [m Pa s⁻²]
    F_p = f[:, None] * vtheta_bar / dtheta_dp_safe

    # ---- EP flux divergence ∇·F = (1/a cosφ) ∂(F_φ cosφ)/∂φ + ∂F_p/∂p ----
    div_F = np.full_like(F_phi, np.nan)

    # ∂(F_φ cosφ)/∂φ via centred differences in φ
    Fphi_cosphi = F_phi * cos_phi[:, None]
    dphi = np.deg2rad(lat_width)
    d_Fphi = np.full_like(Fphi_cosphi, np.nan)
    d_Fphi[1:-1] = (Fphi_cosphi[2:] - Fphi_cosphi[:-2]) / (2.0 * dphi)
    d_Fphi[0] = (Fphi_cosphi[1] - Fphi_cosphi[0]) / dphi
    d_Fphi[-1] = (Fphi_cosphi[-1] - Fphi_cosphi[-2]) / dphi

    # ∂F_p/∂p via centred differences in pressure
    dFp_dp = np.full_like(F_p, np.nan)
    dFp_dp[:, 1:-1] = (
        (F_p[:, 2:] - F_p[:, :-2])
        / (pressures_pa[2:] - pressures_pa[:-2])[None, :]
    )
    dFp_dp[:, 0] = (F_p[:, 1] - F_p[:, 0]) / dp[0]
    dFp_dp[:, -1] = (F_p[:, -1] - F_p[:, -2]) / dp[-1]

    # ∇·F = (1/(a cosφ)) ∂(cosφ F_φ)/∂φ + ∂F_p/∂p
    # Units: (1/m)·(m²/s²) + (m Pa/s²)/Pa = m/s²
    # Multiply by 86 400 s/day → m s⁻¹ day⁻¹
    with np.errstate(invalid="ignore"):
        div_F = (1.0 / (_EARTH_RADIUS * cos_phi[:, None])) * d_Fphi + dFp_dp
    div_F_accel = div_F * 86400.0   # m s⁻¹ day⁻¹

    # ---- residual meridional velocity ----------------------------------------
    # v* = v̄ − ∂/∂p (v′θ′ / ∂θ̄/∂p)
    heat_flux_ratio = vtheta_bar / dtheta_dp_safe   # (n_lats, n_lev)
    d_hfr_dp = np.full_like(heat_flux_ratio, np.nan)
    d_hfr_dp[:, 1:-1] = (
        (heat_flux_ratio[:, 2:] - heat_flux_ratio[:, :-2])
        / (pressures_pa[2:] - pressures_pa[:-2])[None, :]
    )
    d_hfr_dp[:, 0] = (heat_flux_ratio[:, 1] - heat_flux_ratio[:, 0]) / dp[0]
    d_hfr_dp[:, -1] = (heat_flux_ratio[:, -1] - heat_flux_ratio[:, -2]) / dp[-1]

    v_star = v_bar - d_hfr_dp

    return {
        "lat_centres": lat_centres,
        "u_bar": u_bar,
        "theta_bar": theta_bar,
        "dtheta_dp": dtheta_dp,
        "F_phi": F_phi,
        "F_p": F_p,
        "div_F": div_F_accel,
        "v_star": v_star,
    }


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_qg_tem(
    zarr_path: Path,
    label: str,
    sample: int = 0,
    lat_width: float = 2.5,
) -> dict[str, Any] | None:
    """
    Extract and compute QG-TEM diagnostics for every forecast step.

    Returns a dict with keys:
        ``label``, ``datetimes``, ``pressures_hpa``, ``lat_centres``,
        ``pred``: dict of (n_time, n_lats, n_levels) arrays,
        ``tgt``:  dict of (n_time, n_lats, n_levels) arrays.
    ``pred`` and ``tgt`` each contain: ``div_F``, ``v_star``, ``F_phi``, ``F_p``.
    """
    _logger.info("%s: extracting QG-TEM diagnostics …", label)

    with open_validation(zarr_path) as zio:
        stream_name = get_stream(zio)
        if stream_name != "ERA5ml":
            _logger.warning(
                "%s: QG-TEM requires ERA5ml stream (found %s) — skipping",
                label,
                stream_name,
            )
            return None

        channels = get_channels(zio, stream_name, sample)
        coords = get_coords(zio, stream_name, sample)
        lats = coords[:, 0]

        # Build channel → full-level pressure maps (Pa) for u, v, t
        u_map = build_full_level_map("u", channels)
        v_map = build_full_level_map("v", channels)
        t_map = build_full_level_map("t", channels)

        # Keep only stratospheric levels (p ≤ 150 hPa) that appear in all three
        strat_pressures_hpa = sorted(
            {
                p for ch, p in u_map.items()
                if p > 0.01 and p <= 150.0
                and ch.replace("u_", "v_") in v_map
                and ch.replace("u_", "t_") in t_map
            }
        )

        if len(strat_pressures_hpa) < 3:
            _logger.warning(
                "%s: fewer than 3 common stratospheric u/v/t levels — cannot compute "
                "vertical derivatives; skipping.  Found u levels: %s",
                label,
                sorted(u_map.values()),
            )
            return None

        # Invert map: pressure → channel name (use u to find the others)
        p2u: dict[float, str] = {p: ch for ch, p in u_map.items() if p in strat_pressures_hpa}
        p2v: dict[float, str] = {p: ch.replace("u_", "v_") for p, ch in p2u.items()}
        p2t: dict[float, str] = {p: ch.replace("u_", "t_") for p, ch in p2u.items()}

        pressures_hpa = np.array(strat_pressures_hpa, dtype=np.float64)
        pressures_pa = pressures_hpa * 100.0

        u_ch_idx = [channels.index(p2u[p]) for p in strat_pressures_hpa]
        v_ch_idx = [channels.index(p2v[p]) for p in strat_pressures_hpa]
        t_ch_idx = [channels.index(p2t[p]) for p in strat_pressures_hpa]

        steps = get_forecast_steps(zio, skip_source_step=True)
        _, lat_centres = _build_lat_bins(lat_width)

        n_t = len(steps)
        n_lats = len(lat_centres)
        n_lev = len(strat_pressures_hpa)

        # Accumulate per-step diagnostics
        keys = ("div_F", "v_star", "F_phi", "F_p", "u_bar")
        pred_acc: dict[str, np.ndarray] = {k: np.full((n_t, n_lats, n_lev), np.nan) for k in keys}
        tgt_acc:  dict[str, np.ndarray] = {k: np.full((n_t, n_lats, n_lev), np.nan) for k in keys}
        raw_times: list = []

        for i, step in enumerate(steps):
            pred, tgt, times = load_step(zio, stream_name, step, sample)
            pred3 = np.atleast_3d(pred)   # (n_pts, n_ch, n_ens)
            tgt3 = np.atleast_3d(tgt)

            for arr3, acc in ((pred3, pred_acc), (tgt3, tgt_acc)):
                u_field = arr3[:, u_ch_idx, 0].astype(np.float64)   # (n_pts, n_lev)
                v_field = arr3[:, v_ch_idx, 0].astype(np.float64)
                t_field = arr3[:, t_ch_idx, 0].astype(np.float64)
                diags = compute_qg_tem(u_field, v_field, t_field, lats, pressures_pa, lat_width)
                for k in keys:
                    acc[k][i] = diags[k]

            raw_times.append(times[0])

    datetimes = convert_times_to_datetime(np.array(raw_times))
    _logger.info(
        "%s: %d steps, %d lat bands, %d levels (%.1f–%.1f hPa)",
        label, n_t, n_lats, n_lev, pressures_hpa[0], pressures_hpa[-1],
    )

    return {
        "label": label,
        "datetimes": datetimes,
        "pressures_hpa": pressures_hpa,
        "lat_centres": lat_centres,
        "pred": pred_acc,
        "tgt": tgt_acc,
    }


def _smooth(field: np.ndarray, sigma: float = 1.5) -> np.ndarray:
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


def _pressure_axis(ax: Any, ssw_idx: int | None = None) -> None:
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(_P_TICKS)
    ax.set_yticklabels(_P_TICK_LABELS)
    ax.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    if ssw_idx is not None:
        ax.axvline(ssw_idx, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="SSW onset")


def _get_ssw_idx(datetimes: list[datetime], ssw_date: datetime) -> int | None:
    if datetimes[0] <= ssw_date <= datetimes[-1]:
        return int(np.argmin([abs((dt - ssw_date).total_seconds()) for dt in datetimes]))
    return None


def _time_ticks(ax: Any, datetimes: list[datetime], every: int = 14) -> None:
    n = len(datetimes)
    ticks = np.arange(0, n, every)
    ax.set_xticks(ticks)
    ax.set_xticklabels([datetimes[i].strftime("%m-%d") for i in ticks], rotation=45, fontsize=9)
    ax.set_xlabel("Date (MM-DD)", fontsize=11)


# ---------------------------------------------------------------------------
# Plot: EP flux divergence Hovmöller
# ---------------------------------------------------------------------------


def plot_ep_flux_divergence_hovmoller(
    data: dict[str, Any],
    output_dir: Path,
    ssw_date: datetime = SSW_DATE,
    lat_range: tuple[float, float] = (20.0, 90.0),
    clim: float = 5.0,
) -> None:
    """
    Hovmöller (time × pressure) of EP flux divergence at a fixed latitude band.

    A 2-column figure is produced: prediction (left) and ERA5 target (right).
    """
    label = data["label"]
    datetimes = data["datetimes"]
    pressures = data["pressures_hpa"]
    lat_centres = data["lat_centres"]
    ssw_idx = _get_ssw_idx(datetimes, ssw_date)

    # Average over latitude band
    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    if not lat_mask.any():
        _logger.warning("No latitude bins in range %s — skipping Hovmöller", lat_range)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)
    cmap = "RdBu_r"

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target")):
        # Average ∇·F over the latitude band: (n_time, n_lev)
        field = np.nanmean(data[key]["div_F"][:, lat_mask, :], axis=1)
        # Smooth in time × pressure space to suppress grid-scale noise
        field = _smooth(field, sigma=1.5)
        im = ax.contourf(
            np.arange(len(datetimes)),
            pressures,
            field.T,
            levels=np.linspace(-clim, clim, 21),
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        ax.contour(
            np.arange(len(datetimes)),
            pressures,
            field.T,
            levels=[0.0],
            colors="k",
            linewidths=0.8,
        )
        _pressure_axis(ax, ssw_idx)
        _time_ticks(ax, datetimes)
        ax.set_title(f"{label}  {title}\n∇·F ({lat_range[0]:.0f}–{lat_range[1]:.0f}°N mean)",
                     fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=axes, label="∇·F  (m s⁻¹ day⁻¹)", shrink=0.8)
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
    ssw_date: datetime = SSW_DATE,
    clim: float = 3.0,
    lat_range: tuple[float, float] = (0.0, 90.0),
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

    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    lats_plot = lat_centres[lat_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)
    cmap = "RdBu_r"

    # Quiver subsampling: ~15 arrows in each dimension
    lat_skip = max(1, lat_mask.sum() // 15)
    lev_skip = max(1, len(pressures) // 10)
    Lq = lats_plot[::lat_skip]
    Pq = pressures[::lev_skip]

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target")):
        # Time-mean then smooth
        div_F_mean = _smooth(np.nanmean(data[key]["div_F"][:, lat_mask, :], axis=0))
        F_phi_mean = _smooth(np.nanmean(data[key]["F_phi"][:, lat_mask, :], axis=0))
        F_p_mean   = _smooth(np.nanmean(data[key]["F_p"  ][:, lat_mask, :], axis=0))
        u_bar_mean = _smooth(np.nanmean(data[key]["u_bar"][:, lat_mask, :], axis=0))

        im = ax.contourf(
            lats_plot, pressures, div_F_mean.T,
            levels=np.linspace(-clim, clim, 21),
            cmap=cmap, norm=norm, extend="both",
        )
        ax.contour(lats_plot, pressures, div_F_mean.T,
                   levels=[0.0], colors="k", linewidths=0.8)

        # Overlay zonal mean wind contours (thin grey dashed)
        u_levels = np.arange(-80, 90, 10)
        cs = ax.contour(lats_plot, pressures, u_bar_mean.T,
                        levels=u_levels, colors="grey",
                        linewidths=0.6, linestyles="--", alpha=0.6)
        ax.clabel(cs, levels=[l for l in u_levels if l % 20 == 0],
                  inline=True, fontsize=7, fmt="%d")

        # EP flux arrows: normalise each component by its 95th-percentile
        # absolute value so arrows show direction+relative magnitude, not
        # absolute magnitude (which differs by orders of magnitude between
        # F_φ and F_p).
        Fq = F_phi_mean[::lat_skip, :][:, ::lev_skip]
        Gq = F_p_mean[::lat_skip,   :][:, ::lev_skip]
        ref_F = np.nanpercentile(np.abs(F_phi_mean), 95) or 1.0
        ref_G = np.nanpercentile(np.abs(F_p_mean),   95) or 1.0
        U = Fq / ref_F
        # F_p > 0 means downward (increasing p); flip sign so arrow points upward
        V = -(Gq / ref_G)
        ax.quiver(
            Lq, Pq, U.T, V.T,
            scale=15, width=0.004, color="k", alpha=0.8,
            headwidth=4, headlength=5,
        )

        _pressure_axis(ax)
        ax.set_xlim(lat_range)
        ax.set_xlabel("Latitude (°N)", fontsize=11)
        ax.set_title(
            f"{label}  {title}\nTime-mean EP flux  (arrows: direction, grey = ū contours)",
            fontsize=12, fontweight="bold",
        )

    cb = plt.colorbar(im, ax=axes, label="∇·F  (m s⁻¹ day⁻¹)", shrink=0.8)
    cb.ax.tick_params(labelsize=9)
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
    ssw_date: datetime = SSW_DATE,
    lat_range: tuple[float, float] = (50.0, 80.0),
    clim: float = 0.3,
) -> None:
    """
    Hovmöller (time × pressure) of residual meridional velocity v*.
    """
    label = data["label"]
    datetimes = data["datetimes"]
    pressures = data["pressures_hpa"]
    lat_centres = data["lat_centres"]
    ssw_idx = _get_ssw_idx(datetimes, ssw_date)

    lat_mask = (lat_centres >= lat_range[0]) & (lat_centres <= lat_range[1])
    if not lat_mask.any():
        _logger.warning("No latitude bins in range %s for v* Hovmöller", lat_range)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    norm = TwoSlopeNorm(vmin=-clim, vcenter=0.0, vmax=clim)

    for ax, key, title in zip(axes, ("pred", "tgt"), ("Prediction", "ERA5 Target")):
        field = np.nanmean(data[key]["v_star"][:, lat_mask, :], axis=1)
        field = _smooth(field, sigma=1.5)
        im = ax.contourf(
            np.arange(len(datetimes)),
            pressures,
            field.T,
            levels=np.linspace(-clim, clim, 21),
            cmap="RdBu_r",
            norm=norm,
            extend="both",
        )
        ax.contour(np.arange(len(datetimes)), pressures, field.T,
                   levels=[0.0], colors="k", linewidths=0.8)
        _pressure_axis(ax, ssw_idx)
        _time_ticks(ax, datetimes)
        ax.set_title(
            f"{label}  {title}\nv* residual ({lat_range[0]:.0f}–{lat_range[1]:.0f}°N mean)",
            fontsize=12, fontweight="bold",
        )

    plt.colorbar(im, ax=axes, label="v*  (m s⁻¹)", shrink=0.8)
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

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    fig, ax = plt.subplots(figsize=(13, 5))

    for color_idx, data in enumerate(data_list):
        label = data["label"]
        datetimes = data["datetimes"]
        lat_centres = data["lat_centres"]
        pressures = data["pressures_hpa"]
        color = colors[color_idx % len(colors)]

        lat_mask = lat_centres >= polar_lat_min
        p_mask = (pressures >= pressure_range_hpa[0]) & (pressures <= pressure_range_hpa[1])

        if not lat_mask.any() or not p_mask.any():
            continue

        for key, ls, alpha, suffix in (
            ("pred", "-",  0.9, "pred"),
            ("tgt",  "--", 0.6, "ERA5"),
        ):
            series = np.nanmean(
                data[key]["div_F"][:, lat_mask, :][:, :, p_mask],
                axis=(1, 2),
            )
            ax.plot(datetimes, series,
                    color=color, linestyle=ls, linewidth=2.0, alpha=alpha,
                    label=f"{label} {suffix}")

    ssw_in_range = any(
        d["datetimes"][0] <= ssw_date <= d["datetimes"][-1] for d in data_list
    )
    if ssw_in_range:
        ax.axvline(ssw_date, color="red", linestyle="-.", linewidth=1.8, alpha=0.7,
                   label=f"SSW onset ({ssw_date.strftime('%b %d, %Y')})")

    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("∇·F  (m s⁻¹ day⁻¹)", fontsize=12)
    ax.set_title(
        f"Polar cap (≥{polar_lat_min:.0f}°N, "
        f"{pressure_range_hpa[0]:.0f}–{pressure_range_hpa[1]:.0f} hPa) EP flux divergence",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out = output_dir / "polar_cap_efd_timeseries.png"
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
    args = parser.parse_args(argv)

    print("=" * 60)
    print("QG-TEM ANALYSIS")
    print("=" * 60)

    cfg = load_validations_config(args.validations_config)
    run_specs = [(label, spec["id"], spec["sample"]) for label, spec in cfg.items()]

    data_list: list[dict[str, Any]] = []

    for label, run_id, sample in run_specs:
        zarr_path = args.data_dir / run_id / _ZARR_FNAME

        if not zarr_path.exists():
            _logger.warning("Store not found, skipping %s: %s", label, zarr_path)
            continue

        data = extract_qg_tem(zarr_path, label, sample, lat_width=args.lat_width)
        if data is None:
            _logger.warning("Skipping %s — extraction failed.", label)
            continue

        data_list.append(data)

        # Per-experiment plots
        exp_dir = args.output_dir / label
        exp_dir.mkdir(parents=True, exist_ok=True)

        plot_ep_flux_divergence_hovmoller(
            data, exp_dir, args.ssw_date, clim=args.clim_divF,
        )
        plot_ep_flux_vectors(data, exp_dir, args.ssw_date, clim=args.clim_divF)
        plot_v_residual_hovmoller(data, exp_dir, args.ssw_date, clim=args.clim_vstar)

        # Save numpy archive
        npz_path = exp_dir / "qg_tem_diagnostics.npz"
        np.savez_compressed(
            npz_path,
            lat_centres=data["lat_centres"],
            pressures_hpa=data["pressures_hpa"],
            **{f"pred_{k}": v for k, v in data["pred"].items()},
            **{f"tgt_{k}":  v for k, v in data["tgt"].items()},
        )
        _logger.info("Saved %s", npz_path)

    # Multi-experiment time series
    if data_list:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_polar_cap_efd_timeseries(
            data_list, args.output_dir, args.ssw_date, args.polar_lat_min,
        )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
