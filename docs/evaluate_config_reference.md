# Fast Evaluation Config Reference

This document is the reference for all configuration options accepted by
`uv run evaluate --config <your_config.yml>`.

A working template to copy and edit is `config/eval_config.yml`

---

## Table of Contents

1. [Overview: config layout](#1-overview-config-layout)
2. [Top-level keys](#2-top-level-keys)
3. [`global_plotting_options`](#3-global_plotting_options)
4. [`evaluation`](#4-evaluation)
5. [`default_streams`](#5-default_streams)
6. [`run_ids`](#6-run_ids)
   - [Common run keys](#61-common-run-keys)
   - [Type: `zarr` (default)](#62-type-zarr-default)
   - [Type: `json`](#63-type-json)
   - [Type: `merge`](#64-type-merge)
   - [Type: `jsonmerge`](#65-type-jsonmerge)
   - [Type: `csv`](#66-type-csv)
7. [Stream config block](#7-stream-config-block)
   - [`evaluation` sub-block](#71-evaluation-sub-block)
   - [`plotting` sub-block](#72-plotting-sub-block)
8. [Metrics reference](#8-metrics-reference)
9. [Regions reference](#9-regions-reference)
10. [Score caching (JSON files)](#10-score-caching-json-files)
11. [CSV format for pre-computed scores](#11-csv-format-for-pre-computed-scores)
12. [CLI overrides](#12-cli-overrides)

---

## 1. Overview: config layout

```
max_workers: ...                  # optional top-level cap on parallelism

global_plotting_options:          # optional — applied to all runs/streams
  ...

evaluation:                       # required — scoring and summary plot settings
  metrics: [...]
  regions: [...]
  ...

default_streams:                  # optional — stream config used when a run_id does not
  ERA5:                           #   specify its own streams
    channels: [...]
    evaluation: ...
    plotting: ...
  CERRA:
    ...

run_ids:                          # required — one entry per run to evaluate
  <run_id>:
    label: "..."
    results_base_dir: "..."
    # streams:  optional; if absent, default_streams is used
  ...
```

**Key design decisions:**

- **Regions** can be set in three places (in order of precedence, highest first):
  1. Per-stream under `default_streams.<STREAM>.regions` / `run_ids.<id>.streams.<STREAM>.regions`
  2. Under `evaluation.regions` (applies to score calculation for all streams)
  3. Under `global_plotting_options.regions` (applies to map generation for all streams)
- **Stream config** can be defined once in `default_streams` and reused by all `run_ids`.
  A run can override this entirely by supplying its own `streams` block.
- **`evaluation`** and **`plotting`** are separate sub-blocks inside every stream config,
  allowing you to score over a broad set of steps/samples while only plotting a subset.

---

## 2. Top-level keys

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `max_workers` | int | yes | — | Hard cap on parallel workers used for I/O, scoring, and plotting. Applied to all runs. Useful on shared nodes to avoid oversubscription. When absent, the number of workers is chosen automatically. |
| `private_paths` | dict | yes | — | HPC-specific private path overrides. Advanced option — only needed on certain clusters. See platform config docs. |

---

## 3. `global_plotting_options`

Applied to all runs. Stream-level blocks inside this section allow per-stream overrides
(e.g. colorscale limits). All keys are optional.

```yaml
global_plotting_options:
  regions: ["global", "europe"]
  image_format: "png"
  animation_format: "gif"
  log_colorbar: false
  dpi_val: 300
  fps: 2
  n_bins: 50
  log_x: false
  log_y: false
  ERA5:
    use_datashader: false
    marker_size: 2
    scale_marker_size: true
    marker: "o"
    alpha: 0.5
    add_healpix_grid: false
    healpix_nside: 4
    healpix_color: "black"
    healpix_linewidth: 0.2
    healpix_linestyle: "-"
    healpix_step: 64
    2t:
      vmin: 250
      vmax: 300
    10u:
      vmin: -40
      vmax: 40
```

### Global image / animation options

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `regions` | list[str] | yes | `["global"]` | Regions for which 2D map plots are generated. See [section 9](#9-regions-reference) for supported values. |
| `image_format` | str | yes | `"png"` | File format for all saved images. Options: `"png"`, `"pdf"`, `"svg"`, `"eps"`, `"jpg"`. |
| `animation_format` | str | yes | `"gif"` | File format for animations. Options: `"gif"`, `"mp4"`. |
| `log_colorbar` | bool | yes | `false` | Use a logarithmic colorscale on 2D map plots. |
| `dpi_val` | int | yes | `300` | DPI for all saved images. |
| `fps` | int | yes | `2` | Frames per second for animations. |
| `n_bins` | int | yes | `50` | Number of bins used in histogram plots. |
| `log_x` | bool | yes | `false` | Log scale on the x-axis of histogram plots. |
| `log_y` | bool | yes | `false` | Log scale on the y-axis of histogram plots. |
| `fig_size` | [float, float] | yes | `null` | Figure size `[width, height]` in inches. When unset, matplotlib's default size is used for map/histogram plots; summary line plots use `[8, 10]`. |

### Per-stream appearance options (e.g. `ERA5:`)

Any stream name can appear as a key inside `global_plotting_options` to set stream-specific
rendering defaults.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `use_datashader` | bool | yes | `false` | Use [datashader](https://datashader.org/) for rendering very dense scatter plots. Requires `datashader` to be installed. |
| `marker_size` | float | yes | stream-dependent | Base scatter-plot marker size (matplotlib `s` units, i.e. pt²). Stream defaults: ERA5 → 2.5, IMERG → 0.25, CERRA → 0.1, others → 0.5. |
| `scale_marker_size` | bool | yes | `false` | Scale marker size by `1/cos²(lat)` to compensate for point clustering at high latitudes. |
| `marker` | str | yes | `"o"` | Marker style string passed to matplotlib. |
| `alpha` | float | yes | — | Marker alpha (transparency), `0.0`–`1.0`. |
| `add_healpix_grid` | bool | yes | `false` | Overlay a HEALPix grid on map plots. |
| `healpix_nside` | int | yes | `4` | HEALPix `nside` controlling grid resolution. Higher values produce a finer grid. |
| `healpix_color` | str | yes | `"black"` | Colour of the HEALPix grid lines. |
| `healpix_linewidth` | float | yes | `0.2` | Width of the HEALPix grid lines in points. |
| `healpix_linestyle` | str | yes | `"-"` | Line style of the HEALPix grid lines (e.g. `"-"`, `"--"`). |
| `healpix_step` | int | yes | `64` | Number of interpolation points per pixel boundary edge. Higher values produce smoother curves. |

Additional keys under a stream (or per-channel block) are forwarded to [matplotlib.axes.Axes.scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html).
This can be useful for style tuning beyond the built-in options above.

Example:

```yaml
global_plotting_options:
  ERA5:
    # built-in keys parsed explicitly
    marker: "o"
    marker_size: 2.0
    alpha: 0.85

    # passed through to scatter (parsed["extra"])
    edgecolors: "none"
    zorder: 3
    rasterized: true

    # per-channel override + extra scatter kwargs
    2t:
      vmin: 250
      vmax: 305
      alpha: 0.9
      edgecolors: "black"
      linewidths: 0.05
```

"Random" pass-through examples here are: `edgecolors`, `linewidths`, `zorder`, `alpha`.
If a key conflicts with an internally managed argument (e.g. `c`, `norm`, `cmap`, `s`, `marker`,
`transform`), the internal value wins.

### Per-channel colorscale limits (e.g. `2t:`)

Under any per-stream block you can add entries keyed by channel name to set fixed colorscale
limits for 2D maps.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `vmin` | float | yes | — | Minimum value of the colorscale. When unset, matplotlib auto-scales. |
| `vmax` | float | yes | — | Maximum value of the colorscale. When unset, matplotlib auto-scales. |

---

## 4. `evaluation`

Controls what to compute and how to visualise summary scores.

```yaml
evaluation:
  metrics: ["rmse", "mae"]
  regions: ["global", "nhem"]
  summary_plots: true
  ratio_plots: false
  heat_maps: false
  score_cards: false
  bar_plots: false
  summary_dir: "./plots/"
  plot_ensemble: "members"
  plot_score_maps: false
  plot_score_animations: false
  plot_score_init_time_series: false
  print_summary: false
  log_scale: false
  add_grid: false
  baseline: "my_run_id"
  # agg_dims: ["ipoint"]
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `metrics` | list | no | — | Metrics to compute. Each item is either a metric name string or a single-key dict `{name: {param: value}}` for parametrised metrics. See [section 8](#8-metrics-reference). |
| `regions` | list[str] | yes | `["global"]` | Regions over which scores are computed. Overrides any region set on individual streams. See [section 9](#9-regions-reference). |
| `summary_dir` | str | yes | `<repo_root>/plots/` | Output directory for all summary (line/ratio/heatmap/etc.) plots. |
| `summary_plots` | bool | yes | `false` | Generate line plots of score vs forecast step, one per metric × region × stream × channel. |
| `ratio_plots` | bool | yes | `false` | Generate ratio plots (score relative to baseline). Requires `baseline` to be set. |
| `heat_maps` | bool | yes | `false` | Generate heat-map plots (score as a function of lead-time and channel). |
| `score_cards` | bool | yes | `false` | Generate score-card summary plots. |
| `bar_plots` | bool | yes | `false` | Generate bar plots of scores. |
| `baseline` | str | yes | — | `run_id` to use as the reference for ratio and improvement calculations. |
| `plot_ensemble` | str\|bool | yes | `false` | How to render ensemble spread on summary line plots. Options: `false` (no spread), `"std"` (mean ± std), `"minmax"` (shaded min–max), `"members"` (individual member lines). |
| `plot_score_maps` | bool | yes | `false` | Plot 2D spatial maps of scores per forecast step. **Slows down evaluation significantly.** |
| `plot_score_animations` | bool | yes | `false` | Animate score maps across forecast steps. Implies `plot_score_maps` must have data. |
| `plot_score_init_time_series` | bool | yes | `false` | Plot score timeseries grouped by initialisation hour of the day. |
| `print_summary` | bool | yes | `false` | Print score values to stdout. Can be very verbose for large runs. |
| `log_scale` | bool | yes | `false` | Use logarithmic y-axis on summary line plots. |
| `add_grid` | bool | yes | `false` | Add a background grid to summary line plots. |
| `agg_dims` | str\|list[str] | yes | `"ipoint"` | **Advanced.** Dimension(s) to aggregate (average) scores over. Supported values: `"ipoint"`, `"sample"`, `"forecast_step"`, `"ensemble"`. Default averages over spatial points only. Use with caution — averaging over sample or forecast_step hides temporal structure. |

---

## 5. `default_streams`

Defines the stream configuration used by any `run_id` that does not specify its own `streams`
block. The structure is identical to the per-run `streams` block described in [section 7](#7-stream-config-block).

```yaml
default_streams:
  ERA5:
    regions: ["global"]
    channels: ["2t", "10u", "z_500", "t_850"]
    regrid: true
    evaluation:
      forecast_step: "all"
      sample: "all"
      ensemble: "all"
    plotting:
      sample: [0, 1]
      forecast_step: [1, 2, 4, 8]
      ensemble: [0]
      plot_maps: true
      plot_bias: false
      plot_target: false
      plot_histograms: true
      plot_animations: false
  CERRA:
    regions: ["europe"]
    channels: ["z_500", "t_850", "u_850"]
    evaluation:
      forecast_step: "all"
      sample: "all"
    plotting:
      sample: [0]
      forecast_step: "all"
      plot_maps: true
      plot_histograms: true
      plot_animations: false
```

See [section 7](#7-stream-config-block) for a full description of all keys.

---

## 6. `run_ids`

Each key under `run_ids` is a run identifier. The value is a configuration dict whose
required and optional keys depend on the `type` field.

### 6.1 Common run keys

These apply to all run types.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `label` | str | yes | run_id | Human-readable label used in plot legends. |
| `color` | str | yes | auto | Matplotlib colour string for this run in line/bar plots (e.g. `"magenta"`, `"#2ca02c"`). When absent, colours are assigned automatically. |
| `type` | str | yes | `"zarr"` | Reader type. Options: `"zarr"`, `"json"`, `"merge"`, `"jsonmerge"`, `"csv"`. |
| `streams` | dict | yes | — | Stream-specific config for this run. If absent, `default_streams` is used. When present, **`default_streams` is completely ignored for this run** — specify all required streams explicitly. |

### 6.2 Type: `zarr` (default)

Standard run reading directly from WeatherGenerator Zarr output.

```yaml
run_ids:
  ar40mckx:
    label: "My run"
    results_base_dir: "./results/"
    mini_epoch: 0
    rank: "all"
    # streams:  optional; uses default_streams if omitted
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `results_base_dir` | str | yes | private config | Base directory containing Zarr output folders. Required if `private_paths` is not set. |
| `runplot_base_dir` | str | yes | `results_base_dir` | Base directory for 2D map and histogram plots. |
| `metrics_base_dir` | str | yes | `results_base_dir` | Base directory for cached score JSON files. |
| `metrics_dir` | str | yes | `metrics_base_dir/evaluation` | Explicit path for score JSON files. Overrides `metrics_base_dir` if set. |
| `model_base_dir` | str | yes | — | Directory containing the model config files (used when `private_paths` is not set). |
| `mini_epoch` | int | yes | `0` | Epoch number used to identify the Zarr store. In inference this is always `0`. |
| `rank` | int\|str\|list | yes | `"all"` | Rank(s) of the Zarr store to read. Use `"all"` for multi-rank inference, an integer for a single rank, or a list of integers. |

### 6.3 Type: `json`

Reads pre-computed scores from JSON files (no Zarr data required). Useful when the original
Zarr output has been deleted or is unavailable.

```yaml
run_ids:
  so67dku1:
    type: "json"
    label: "Archived run"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2]
          ensemble: "all"
```

Uses the same path keys as the `zarr` type (`results_base_dir`, `metrics_dir`, etc.).
Plotting (maps, histograms, animations) is **not** available with this type.

### 6.4 Type: `merge`

Stacks multiple Zarr runs over the ensemble dimension. Useful for creating a pseudo-ensemble
from several independent runs.

```yaml
run_ids:
  merge_test:
    type: "merge"
    merge_run_ids:
      - so67dku4
      - c9cg8ql3
    merge_metrics_dir: "./merge_test/metrics/"
    label: "Merged ensemble"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2, 3]
          ensemble: "all"
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `merge_run_ids` | list[str] | no | — | List of existing run_ids to merge. Each must be readable with the `zarr` reader. |
| `merge_metrics_dir` | str | no | — | Directory where merged score JSON files will be written and cached. |

### 6.5 Type: `jsonmerge`

Same as `merge` but reads from pre-computed JSON score files instead of Zarr data.

```yaml
run_ids:
  merge_archived:
    type: "jsonmerge"
    merge_run_ids:
      - so67dku4
      - c9cg8ql3
    merge_metrics_dir: "./merge_archived/metrics/"
    label: "Merged archived"
    results_base_dir: "./results/"
    streams:
      ERA5:
        channels: ["z_500", "t_850"]
        evaluation:
          forecast_step: [2, 4, 6]
          sample: [0, 1, 2, 3]
          ensemble: "all"
```

Same required keys as `merge`.

### 6.6 Type: `csv`

Reads pre-computed scores from CSV files generated by external tools (e.g. ECMWF Quaver).
Only score line plots are produced; no maps, histograms, or animations.

```yaml
run_ids:
  pangu:
    type: "csv"
    label: "Pangu-Weather"
    metrics_dir: "<path to folder containing run_id sub-folder>"
    streams:
      ERA5:
        channels: ["2t", "q_850", "t_850", "z_500"]
        evaluation:
          forecast_step: "all"
          sample: "all"
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `metrics_dir` | str | no | — | Path to the folder containing a sub-folder named `<run_id>/` with CSV files. |

See [section 11](#11-csv-format-for-pre-computed-scores) for the expected CSV column layout.

---

## 7. Stream config block

The same structure is used under `default_streams`, under `run_ids.<id>.streams`, and
inside `merge` type runs.

```yaml
ERA5:                                 # stream name
  regions: ["global", "nhem"]        # regions for maps (overrides global_plotting_options)
  channels: ["2t", "10u", "z_500"]
  offset: "1h"                       # optional
  regrid: true                       # optional
  climatology_path: "/path/..."      # optional
  evaluation:
    forecast_step: "all"
    sample: "all"
    ensemble: "all"
  plotting:
    forecast_step: [1, 2, 4, 8]
    sample: [0, 1]
    ensemble: [0]
    plot_maps: true
    plot_bias: false
    plot_target: false
    plot_histograms: true
    plot_animations: false
    plot_subtimesteps: false
```

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `regions` | list[str] | yes | `["global"]` | Regions for 2D maps for this stream. Overrides `global_plotting_options.regions`. |
| `channels` | list[str] | yes | all available | List of channel names to process (e.g. `["2t", "10u", "z_500"]`). |
| `offset` | str | yes | — | Timedelta offset used to infer initialisation time when `source_interval` is absent. Format examples: `"1h"`, `"30m"`, `"2h30m"`. |
| `regrid` | bool\|dict | yes | `false` | Regrid data from the native model grid to a regular lat/lon grid before scoring and plotting. `true` defaults to a 1.5°×1.5° grid. Use a dict `{target_grid: [0.25, 0.25]}` to change the resolution. Target grid can also be a string such as `"O96"`. |
| `climatology_path` | str | yes | auto | Explicit path to a climatology Zarr file used for ACC, FACT, TACT metrics. When absent, the code tries to infer it from the model config. |

### 7.1 `evaluation` sub-block

Controls which data are loaded and scored.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `forecast_step` | str\|list[int] | yes | `"all"` | Forecast steps to score. `"all"` uses every available step. A list `[1, 2, 4]` selects specific steps. A range string `"1-50"` is equivalent to `[1, 2, ..., 50]`. |
| `sample` | str\|list[int] | yes | `"all"` | Samples (initialisation times) to score. `"all"` or a list of integers. |
| `ensemble` | str\|list[int] | yes | `"all"` | Ensemble members to include in scoring. `"all"` uses every member; `"mean"` uses the ensemble mean; a list `[0, 1, 2]` selects specific members. |

### 7.2 `plotting` sub-block

Controls which subset of the evaluated data is visualised with maps, histograms, and animations.
You can score over a broad set and plot only a representative subset.

| Key | Type | Optional | Default | Description |
|-----|------|----------|---------|-------------|
| `forecast_step` | str\|list[int] | yes | `"all"` | Forecast steps for which plots are created. Same syntax as `evaluation.forecast_step`. |
| `sample` | list[int] | yes | `"all"` | Samples for which plots are created. |
| `ensemble` | str\|list[int] | yes | `"all"` | Ensemble members for which maps/histograms are created. Same syntax as `evaluation.ensemble`. |
| `plot_maps` | bool | yes | `false` | Plot a 2D scatter map for each channel, valid time, and selected sample/ensemble member. |
| `plot_bias` | bool | yes | `true` | Plot the bias (prediction − target) as a 2D map alongside the prediction map. |
| `plot_target` | bool | yes | `true` | Also plot the target (ground truth) data using the same plotting options. |
| `plot_histograms` | bool\|str | yes | `false` | Plot histograms of target vs prediction. `true` or `"per-sample"` creates one histogram per sample; `"across-samples"` aggregates all samples into a single histogram. |
| `plot_animations` | bool | yes | `false` | Build an animation (GIF/MP4) cycling through forecast steps for each channel and sample. |
| `plot_subtimesteps` | bool | yes | `false` | Create separate plots for each sub-timestep within a single forecast step (only relevant for `tokenize_spacetime` models). |

---

## 8. Metrics reference

Metrics are listed under `evaluation.metrics`. Each entry is either a plain string or a
single-key dict with a parameter sub-dict:

```yaml
evaluation:
  metrics:
    - rmse
    - mae
    - psd:
        psd_method: "sht"
    - fbi:
        thresh: 280
```

### Deterministic metrics (require prediction and target)

| Name | Description |
|------|-------------|
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `rmse` | Root Mean Squared Error |
| `vrmse` | Variance-normalised RMSE |
| `l1` | L1 error norm |
| `l2` | L2 error norm |
| `bias` | Mean bias (prediction − target) |
| `psnr` | Peak Signal-to-Noise Ratio |
| `nse` | Nash–Sutcliffe Efficiency |
| `ets` | Equitable Threat Score. Default threshold per-variable (see `score.py`). Override with `thresh`. |
| `pss` | Peirce Skill Score. Override threshold with `thresh`. |
| `fbi` | Frequency Bias Index. Override threshold with `thresh`. |
| `seeps` | Stable Equitable Error in Probability Space ([Rodwell et al., 2011](https://journals.ametsoc.org/view/journals/mwre/140/8/mwr-d-11-00301.1.pdf)). |
| `grad_amplitude` | Ratio of spatial variability (gradient amplitude) between prediction and target. Requires a regular lat/lon grid. |
| `qq_analysis` | Quantile–quantile analysis. Produces quantile plots rather than line plots. |
| `psd` | Power Spectral Density. Produces PSD plots rather than line plots. Parameters: `psd_method` (`"sht"` or `"fft"`, default `"sht"`); for `"fft"` only: `psd_regrid_resolution` (degrees, default 1.0). |

### Metrics requiring alignment between consecutive forecast steps

> Cannot be used when coordinates change between steps (shuffled data is fine).

| Name | Description |
|------|-------------|
| `froct` | Forecast Rate of Change over Time |
| `troct` | Target Rate of Change over Time |

### Metrics requiring a pre-computed climatology

> Need either `climatology_path` in the stream config or a `data_path_aux` key in the model config.

| Name | Description |
|------|-------------|
| `acc` | Anomaly Correlation Coefficient |
| `rps` | Ranked Probability Score |
| `rpss` | Ranked Probability Skill Score |
| `fact` | Forecast Activity (standard deviation of forecast anomaly) |
| `tact` | Target Activity (standard deviation of target anomaly) |

### Probabilistic metrics (require ensemble dimension)

| Name | Description |
|------|-------------|
| `ssr` | Spread–Skill Ratio |
| `crps` | Continuous Ranked Probability Score (via xskillscore) |
| `rank_histogram` | Rank Histogram (Talagrand diagram) |
| `spread` | Ensemble Spread |

### Metric parameters

Any metric that accepts a `thresh` parameter can be configured like:

```yaml
evaluation:
  metrics:
    - fbi:
        thresh: 280       # custom threshold (e.g. for 2t in Kelvin)
    - ets:
        thresh: 0.001     # custom threshold (e.g. for precipitation)
```

---

## 9. Regions reference

Predefined bounding boxes `(lat_min, lat_max, lon_min, lon_max)`:

| Name | Lat range | Lon range | Notes |
|------|-----------|-----------|-------|
| `global` | −90 to 90 | −180 to 180 | Robinson projection for maps |
| `nhem` | 0 to 90 | −180 to 180 | Northern Hemisphere |
| `shem` | −90 to 0 | −180 to 180 | Southern Hemisphere |
| `tropics` | −30 to 30 | −180 to 180 | Tropical belt |
| `europe` | 35 to 70 | −10 to 40 | |
| `belgium` | 49 to 52 | 2 to 7 | |
| `arctic` | 50 to 90 | −180 to 180 | Stereographic projection for maps |
| `uwc-west` | 39 to 63 | −26 to 41 | UWC-West domain |
| `arome` | 37 to 56 | −12 to 16 | AROME domain |
| `icon` | 42 to 51 | −1 to 18 | ICON domain |

Regions are specified as lists of strings. They can appear in three places:

```yaml
# 1. global_plotting_options — affects map region selection for all streams
global_plotting_options:
  regions: ["global", "europe"]

# 2. evaluation — affects score calculation regions for all streams
evaluation:
  regions: ["global", "nhem", "tropics"]

# 3. per stream (highest precedence) — overrides global for a specific stream
default_streams:
  CERRA:
    regions: ["europe"]
```

---

## 10. Score caching (JSON files)

To avoid recomputing scores on every run, results are saved to JSON files.
The path follows this pattern:

```
<metrics_dir>/<run_id>_<stream>_<region>_<metric>_chkpt<epoch:05d>.json
```

Where `metrics_dir` is resolved as follows (first match wins):

1. `run_ids.<id>.metrics_dir` (explicit path)
2. `run_ids.<id>.metrics_base_dir / "evaluation"`
3. `run_ids.<id>.results_base_dir / "evaluation"`
4. Platform-specific shared path from the private config

At runtime, the code checks whether a JSON file already exists for the requested
combination. If it does, the stored scores are loaded; otherwise they are computed and
saved for future use.

---

## 11. CSV format for pre-computed scores

When using `type: "csv"`, CSV files must be placed under
`<metrics_dir>/<run_id>/` and follow this column layout:

```
,parameter,level,number,score,step,date,domain_name,value
0,t,925,0,rmse,0 days 12:00:00,2022-10-01 00:00:00,n.hem,0.031371
1,t,925,0,rmse,0 days 12:00:00,2022-10-01 12:00:00,n.hem,-0.01038
```

| Column | Description |
|--------|-------------|
| `parameter` | Variable short name (e.g. `t`, `z`, `u`) |
| `level` | Pressure level in hPa (e.g. `925`, `500`) |
| `number` | Ensemble member number (use `0` for deterministic) |
| `score` | Metric name in Quaver convention |
| `step` | Lead time as a timedelta string |
| `date` | Initialisation date-time |
| `domain_name` | Region name in Quaver convention (e.g. `n.hem`, `tropics`) |
| `value` | Score value |

Channel names are constructed as `<parameter>_<level>` (e.g. `t_925`).

---

## 12. CLI overrides

Individual config values can be overridden from the command line without editing the YAML:

```bash
uv run evaluate --config myconfig.yml \
  --options evaluation.summary_plots=true evaluation.regions=[global,nhem]
```

The `--options` flag uses OmegaConf dot-notation and does **not** support overriding
`run_ids` keys (use `--run-ids` for that):

```bash
# Restrict evaluation to a subset of run_ids
uv run evaluate --config myconfig.yml --run-ids ar40mckx c8g5katp
```

Upload scores to MLFlow after evaluation:

```bash
uv run evaluate --config myconfig.yml --push-metrics
```
