#!/bin/bash
# ============================================================================
# launch_pretrain.sh
#
# Submit the pretraining jobs of the strato_s2s arms (4 chained 12h jobs each) and
# append "arm,run_id,slurm_job_id,timestamp" to config/strato_s2s/run_ids.csv.
#
# Run from the WeatherGenerator repo root:
#   config/strato_s2s/launch_pretrain.sh                # all arms
#   config/strato_s2s/launch_pretrain.sh strat memory   # selected arms
#   DRY_RUN=1 config/strato_s2s/launch_pretrain.sh      # only print the commands
# ============================================================================
set -euo pipefail

LAUNCH=../WeatherGenerator-private/hpc/launch-slurm.py
CFG_DIR=config/strato_s2s
OUT_CSV="${CFG_DIR}/run_ids.csv"
ALL_ARMS=(memory strat strat_tem)
ARMS=("${@:-${ALL_ARMS[@]}}")

SLURM_ARGS=(--nodes=2 --ntasks=8 --gpus-per-node=4 --account=hclimrep -t 12:00:00 --chain-jobs=4 --no-register)

[[ -f "${OUT_CSV}" ]] || echo "arm,run_id,slurm_job_id,timestamp" > "${OUT_CSV}"

for ARM in "${ARMS[@]}"; do
    ARM_CFG="${CFG_DIR}/arm_${ARM}.yml"
    if [[ ! -f "${ARM_CFG}" ]]; then
        echo "Unknown arm '${ARM}' (expected one of: ${ALL_ARMS[*]})" >&2
        exit 1
    fi

    CMD=("${LAUNCH}" "${SLURM_ARGS[@]}"
         --config "${CFG_DIR}/config_base.yml" "${ARM_CFG}" "${CFG_DIR}/split_control.yml")

    echo "=== Launching arm=${ARM} ==="
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "${CMD[*]}"
        continue
    fi

    OUTPUT=$("${CMD[@]}" 2>&1 | tee /dev/stderr)

    RUN_ID=$(grep -oP 'Using generated run id: \K\S+' <<< "${OUTPUT}" | tail -1 || true)
    SLURM_JOB_ID=$(grep -oP 'Slurm job launched with id: \K\S+' <<< "${OUTPUT}" | head -1 || true)
    if [[ -z "${RUN_ID}" ]]; then
        echo "Could not parse the run id for arm ${ARM}; check the output above." >&2
        exit 1
    fi

    echo "${ARM},${RUN_ID},${SLURM_JOB_ID},$(date -Iseconds)" >> "${OUT_CSV}"
    echo "-> arm=${ARM} run_id=${RUN_ID} slurm_job_id=${SLURM_JOB_ID}"
    echo
done

echo "Run ids in ${OUT_CSV}:"
cat "${OUT_CSV}"
