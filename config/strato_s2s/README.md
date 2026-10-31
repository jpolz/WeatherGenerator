# strato_s2s

Controlled comparison of the benefit of the stratosphere for an AI extended-range model (ERA5pl + S2S
predictors), plus physical-plausibility checks of the stratospheric component.
Self-contained: copy this folder to `config/` of another checkout; paths are relative to the repo root.

## Arms (identical except for the stream directory)

| Arm | Config | Streams (`streams/<dir>/`) | Purpose |
|---|---|---|---|
| era5pl | `arm_era5pl.yml` | `era5pl/`: ERA5pl only | single-stream reference (cost/benefit of any extra stream) |
| memory | `arm_memory.yml` | `memory/`: ERA5pl + S2S memory fields | baseline without stratosphere |
| strat | `arm_strat.yml` | `strat/`: + stratosphere (t/u/v/z @ 1, 2, 5, 10, 30, 70 hPa) | benefit of the stratosphere (strat vs memory) |
| strat_w01 | `arm_strat_w01.yml` | `strat_w01/`: as strat, S2S `loss_weight: 0.1` | strat with the memory arm's ERA5pl loss share |
| strat_tem | `arm_strat_tem.yml` | `strat_tem/`: strat + TEM stream | benefit of TEM (strat_tem vs strat) |

Stack the configs in this order (later files override earlier ones):

- Pretraining: `config_base.yml`, `arm_<name>.yml`, `split_control.yml`
- Finetuning: the same three files plus `finetuning.yml` (32 mini-epochs, 8-step rollout, frozen backbone)

Pretraining: AdamW, 128 mini-epochs, 2-step forecast, uniform channel weights. Checkpoints from other runs
cannot be reused because the LR schedule differs.

## S2S memory fields

Trimmed relative to `config/strato_final/`: runoff (`ro`), precipitation/snowfall (`lsp`, `sf`, `csf`, `lsf`) and
cloud cover (`tcc`, `lcc`, `mcc`) were dropped. Their validation loss sat at a noise floor (~0.1–0.4 vs
0.001–0.01 for the slow fields) and they are redundant with ERA5pl `tp`/`cp`.

- Source and target: `sst`, `sd`, `ci`, `ttr`
- Source only (inputs, no loss): `stl1-3`, `swvl1-3`, `ssrd`, `strd`

The memory set must stay identical in `streams/memory/s2spredictors.yml` and the `strat*` variants.

## Loss weighting

- `location_weight: auto` on ERA5pl, S2Spredictors and TEM resolves to the O96 cos-latitude/points-per-ring
  weighting (grid detected from the number of grid points; `tem_reader.py` now exposes `n_latitudes`).
- Channel weights are uniform (1.0). The stream loss is a *mean over its target channels*, so the number of channels
  changes a stream's per-channel weight. S2Spredictors has 4 targets in the memory arm and 28 in the strat arms, so
  the memory arm uses `loss_weight: 0.1` (≈4/28; set when the strat stream still had 40 targets, i.e. the memory
  channels are weighted slightly lower than in the strat arms). ERA5pl is 1.0 in all arms.
- TEM: `loss_weight: 0.05`. The TEM MSE (~0.2) is 5–8× the ERA5pl/S2S means; with the previous 0.25 TEM was the
  largest term and the ERA5pl share of the training loss dropped to ~0.3 (memory ~0.95, strat ~0.55).
- The total loss is `Σ_streams loss_weight · mean_channels(mse) / n_streams`, so with AdamW + grad clipping a
  stream's *share* of the loss sets how much of each update it gets. The arms therefore differ in the effective
  number of ERA5pl updates (memory ≈ 1.75× strat at equal samples); keep this in mind when comparing at equal
  sample counts.
- Validation losses are unweighted by construction and are not comparable across arms; compare verification scores.

## ERA5pl channels and geoinfo

- Dropped vs the old streams: surface `z` (static, now geoinfo), `tcw`, `tp`, `cp`, `q_50`, `q_100`.
- `skt` is target only (redundant as input next to `2t`/`sst`).
- Geoinfo: ERA5pl keeps orography/land-sea and all time encodings; S2Spredictors only `slt` (soil type for the soil
  fields); TEM none.

## TEM channels

Levels are L137 full levels (hPa): 61=104, 55=75, 48=50, 40=28, 29=9.8, 19=2.5. The old 88 (367 hPa) was tropospheric.
Source = target: `ep_flux_div` at all six levels, plus `F_p` and `vtheta_prime` at 61 and 48 (vertical wave-activity
flux and eddy heat flux, the SSW precursors). Loss weight 0.05.

The data are zonal means in 2.5° latitude bins (72 bins) broadcast onto the 40320 O96 points. `max_num_targets` is
20000 like the other streams: a random subset of 100 points covered only ~48 bins per sample and the polar bins
(where the SSW signal is) in <20% of the samples, because the O96 polar rings have 20 points vs 384 at the equator.

## Open decisions

- TEM selection is a first proposal; trim if it is too many channels.
- Source-only memory channels (target list is a strict subset of the source list) have not been run yet; do a short
  dry run before launching the 128-epoch jobs.
- Inference streams and eval configs for these arms do not exist yet. Use identical channel lists across arms for the
  S2S verification (2018 and 2019 monthly); SSW verification only for the strat arms.
