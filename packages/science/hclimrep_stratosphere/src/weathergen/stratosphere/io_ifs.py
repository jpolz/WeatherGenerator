# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""
IFS S2S hindcast reader (combined NetCDF).

Reads the combined ensemble hindcast NetCDF file produced by merging the
GRIB downloads from ``scripts/download_forecast_ifs.py``. Expected layout::

    <xarray.Dataset>
    Dimensions:        (number: 100, time: 36, step: 47, isobaricInhPa: 1,
                         latitude: 121, longitude: 240)
    Coordinates:
      * number         (number) int64        ensemble member id
      * time           (time) datetime64[ns] forecast initialization date
      * step           (step) timedelta64[ns] forecast lead time (0..46 days)
        valid_time     (time, step) datetime64[ns]
      * isobaricInhPa  (isobaricInhPa) float64  10.0
      * latitude       (latitude) float64  90.0 ... -90.0
      * longitude      (longitude) float64  0.0 ... 358.5
    Data variables:
        t   (number, time, step, isobaricInhPa, latitude, longitude) float32
        u   (number, time, step, isobaricInhPa, latitude, longitude) float32

``number`` combines both control (cf) and perturbed (pf) ensemble members.
Only 10 hPa is present (``isobaricInhPa`` has a single value), consistent
with the u10/t10-only download config.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_logger = logging.getLogger(__name__)


def open_ifs_hindcast(path: Path, chunks: str | dict | None = "auto") -> xr.Dataset:
    """Open the combined IFS S2S hindcast NetCDF file (lazily, dask-backed)."""
    return xr.open_dataset(path, chunks=chunks)


def get_init_dates(ds: xr.Dataset) -> list[datetime]:
    """Return the sorted list of available forecast initialization dates."""
    return list(pd.to_datetime(ds["time"].values).to_pydatetime())


def nearest_init_date(
    ds: xr.Dataset,
    target: datetime,
    tolerance_days: float = 1.5,
) -> datetime | None:
    """
    Return the available init date closest to *target*, or ``None`` if the
    closest one is farther than *tolerance_days* away.
    """
    times = pd.to_datetime(ds["time"].values)
    target_ts = pd.Timestamp(target)
    diffs = np.abs((times - target_ts).total_seconds())
    idx = int(np.argmin(diffs))
    diff_days = diffs[idx] / 86400.0
    if diff_days > tolerance_days:
        _logger.warning(
            "Nearest IFS init date to %s is %s (%.1f days away) — exceeds "
            "tolerance of %.1f days",
            target,
            times[idx],
            diff_days,
            tolerance_days,
        )
        return None
    return times[idx].to_pydatetime()


def extract_ensemble_zonal_mean(
    ds: xr.Dataset,
    init_date: datetime,
    variable: str = "u",
    level_hpa: float = 10.0,
    target_lat: float = 60.0,
) -> tuple[list[datetime], np.ndarray]:
    """
    Extract the zonal-mean ensemble time series at *target_lat* for a single
    forecast initialization.

    Args:
        ds:         Dataset from :func:`open_ifs_hindcast`.
        init_date:  Forecast initialization date (matched via nearest;
                    use :func:`nearest_init_date` first to check tolerance).
        variable:   ``'u'`` or ``'t'``.
        level_hpa:  Pressure level (only 10 hPa is present in this stream).
        target_lat: Latitude for the zonal mean (nearest available row).

    Returns:
        ``(valid_times, series)`` where ``valid_times`` is a list of
        ``len(step)`` datetimes and ``series`` is a ``(n_step, n_member)``
        float64 array.
    """
    if variable not in ds.data_vars:
        raise ValueError(
            f"Variable {variable!r} not found in dataset (available: "
            f"{list(ds.data_vars)})"
        )

    da = ds[variable].sel(time=init_date, method="nearest")
    da = da.sel(isobaricInhPa=level_hpa, method="nearest")
    da = da.sel(latitude=target_lat, method="nearest")
    zonal_mean = da.mean(dim="longitude").transpose("step", "number")

    valid_time_da = ds["valid_time"].sel(time=init_date, method="nearest")
    valid_times = pd.to_datetime(valid_time_da.values).to_pydatetime().tolist()

    series = np.asarray(zonal_mean.values, dtype=np.float64)
    return valid_times, series
