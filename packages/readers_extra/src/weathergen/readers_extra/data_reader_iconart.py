# pylint: disable=bad-builtin

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from pathlib import Path
from typing import override

import dask
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_anemoi import _clip_lat, _clip_lon
from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderIconArt(DataReaderTimestep):
    "Wrapper for ICON-ART variables - Reads Zarr format datasets"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage | None = None,
    ) -> None:
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self._stage = stage
        self._initialized = False

        super().__init__(tw_handler, stream_info)

        self.lat: NDArray | None = None
        self.lon: NDArray | None = None
        self.mesh_size: int = 0

        self.init_empty()
        self._lazy_init()

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Use Dask synchronous scheduler to avoid conflicts with PyTorch DataLoader workers
        dask.config.set(scheduler="synchronous")

        try:
            ds: xr.Dataset = xr.open_zarr(self._filename, consolidated=True)
        except Exception as e:
            name = self._stream_info["name"]
            _logger.error(f"Failed to open {name} at {self._filename}: {e}")
            return

        # Column (variable) names and indices
        colnames = list(ds)
        cols_idx = np.arange(len(colnames))

        levels = self._stream_info.get("levels", [])

        # Load associated statistics file for normalization
        stats_filename = Path(self._filename).with_name(Path(self._filename).stem + "_stats.json")
        with open(stats_filename) as stats_file:
            stats = json.load(stats_file)

        stats_vars_metadata = stats["metadata"]["variables"]
        stats_vars = [v for v in stats_vars_metadata if v not in {"clat", "clon", "time"}]

        assert stats_vars == colnames, (
            f"Variables in normalization file {stats_vars} do not match "
            f"dataset columns {colnames}"
        )

        mean = np.array(stats["statistics"]["mean"], dtype=np.float32)
        stdev = np.array(stats["statistics"]["std"], dtype=np.float32)

        lon_attribute = self._stream_info["attributes"]["lon"]
        lat_attribute = self._stream_info["attributes"]["lat"]
        mesh_attribute = self._stream_info["attributes"]["grid"]

        mesh_size = len(ds[mesh_attribute])

        time_coord: NDArray = ds["time"].values
        start_ds = np.datetime64(time_coord[0])
        end_ds = np.datetime64(time_coord[-1])

        if start_ds > self._tw_handler.t_end or end_ds < self._tw_handler.t_start:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            return

        temporal_frequency = time_coord[1] - time_coord[0]

        # Patch instance attributes that DataReaderTimestep._get_dataset_idxs reads.
        self.data_start_time = start_ds
        self.data_end_time = end_ds
        self.period = temporal_frequency

        time_mask = (time_coord >= self._tw_handler.t_start) & (time_coord <= self._tw_handler.t_end)
        self.len = int(np.sum(time_mask))

        lat = _clip_lat(ds[lat_attribute].values.astype(np.float32))
        lon = _clip_lon(ds[lon_attribute].values.astype(np.float32))
        self.lat = lat
        self.lon = lon
        self.mesh_size = mesh_size

        self.properties = {"stream_id": self._stream_info.get("stream_id", 0)}

        self.colnames = colnames
        self.cols_idx = cols_idx
        self.levels = levels
        self.mean = mean
        self.stdev = stdev
        self.time = time_coord

        source_channels = self._stream_info.get("source")
        if source_channels:
            self.source_channels, self.source_idx = self.select(source_channels)
        elif levels:
            self.source_channels, self.source_idx = self.select_by_level("source")
        else:
            self.source_channels = colnames
            self.source_idx = cols_idx

        target_channels = self._stream_info.get("target")
        if target_channels:
            self.target_channels, self.target_idx = self.select(target_channels)
        elif levels:
            self.target_channels, self.target_idx = self.select_by_level("target")
        else:
            self.target_channels = colnames
            self.target_idx = cols_idx

        selected_channel_indices = list(set(self.source_idx).union(set(self.target_idx)))
        non_positive_stds = np.where(self.stdev[selected_channel_indices] <= 0)[0]
        if len(non_positive_stds) != 0:
            bad_vars = [self.colnames[selected_channel_indices[i]] for i in non_positive_stds]
            raise ValueError(
                f"Abort: Encountered non-positive standard deviations "
                f"for selected columns {bad_vars}."
            )

        self.geoinfo_channels = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        self.target_channel_weights = self.parse_target_channel_weights()

        self.ds = ds

        ds_name = self._stream_info["name"]
        _logger.info(f"{ds_name}: source channels: {self.source_channels}")
        _logger.info(f"{ds_name}: target channels: {self.target_channels}")
        _logger.info(f"{ds_name}: mesh size: {self.mesh_size}, timesteps in window: {self.len}")

    def select(self, ch_filters: list[str]) -> tuple[list[str], NDArray]:
        """
        Allow user to specify which columns they want to access.
        Get functions only returned for these specified columns.

        Parameters
        ----------
        ch_filters: list[str]
            list of patterns to access

        Returns
        -------
        selected_colnames: list[str]
            Selected columns according to the patterns specified in ch_filters
        selected_cols_idx: NDArray
            respective index of these patterns in the data array
        """
        mask = [np.array([f in c for f in ch_filters]).any() for c in self.colnames]

        selected_cols_idx = self.cols_idx[np.where(mask)[0]]
        selected_colnames = [self.colnames[int(i)] for i in np.where(mask)[0]]

        return selected_colnames, selected_cols_idx

    def select_by_level(self, ch_type: str) -> tuple[list[str], NDArray[np.int64]]:
        """
        Select channels constrained by allowed pressure levels and optional excludes.
        ch_type: "source" or "target" (for *_exclude key in stream_info)
        """
        channels_exclude = self.stream_info.get(f"{ch_type}_exclude", [])
        allowed_levels = set(self.levels) if getattr(self, "levels", None) else set()

        new_colnames: list[str] = []
        for ch in self.colnames:
            parts = ch.split("_")
            # Profile channel if exactly one level suffix exists
            if len(parts) == 2 and parts != "":
                level = parts[1]
                ch_base = parts[0]
                if (
                    not allowed_levels or level in allowed_levels
                ) and ch_base not in channels_exclude:
                    new_colnames.append(ch)
            else:
                if ch not in channels_exclude:
                    new_colnames.append(ch)

        mask = [c in new_colnames for c in self.colnames]
        selected_cols_idx = self.cols_idx[np.where(mask)]
        selected_colnames = [self.colnames[int(i)] for i in np.where(mask)[0]]

        return selected_colnames, selected_cols_idx

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.mesh_size = 0
        self.colnames = []
        self.cols_idx = np.array([], dtype=np.int64)
        self.available_vars = []

    @override
    def length(self) -> int:
        """
        Length of dataset

        Parameters
        ----------
        None

        Returns
        -------
        length of dataset
        """
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        self._lazy_init()

        if self.ds is None or self.len == 0 or len(channels_idx) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=0
            )

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if len(t_idxs) == 0:
            return ReaderData.empty(
                num_data_fields=len(channels_idx), num_geo_fields=0
            )

        t_idxs_start = t_idxs[0]
        t_idxs_end = t_idxs[-1] + 1
        n_timesteps = t_idxs_end - t_idxs_start

        channels = np.array(self.colnames)[channels_idx]

        # Load only the needed time steps; each variable has shape (time, mesh_size)
        data = [
            self.ds[ch_].isel(time=slice(t_idxs_start, t_idxs_end)).values.reshape(-1, 1)
            for ch_ in channels
        ]
        data = np.concatenate(data, axis=1).astype(np.float32)

        # coords: (n_timesteps * mesh_size, 2)
        coords_single = np.stack([self.lat, self.lon], axis=1)
        coords = np.tile(coords_single, (n_timesteps, 1))

        datetimes = np.repeat(self.time[t_idxs_start:t_idxs_end], self.mesh_size)

        geoinfos = np.zeros((data.shape[0], 0), dtype=np.float32)

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)

        return rd
