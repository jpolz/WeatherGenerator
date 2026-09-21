#!/bin/bash
# Check completion status of inference runs listed in the run-ID manifest
# (falls back to validation_submissions.log if the manifest doesn't exist yet).
# Checks weathergen.inference.*.out for "Finished inference job with exit code: <N>",
# falling back to output.inference.*.txt for "Finished inference run with id:".

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WG_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MANIFEST="${WG_ROOT}/config/evaluate/runs_manifest.yml"
LOG_FILE="$SCRIPT_DIR/validation_submissions.log"
LOGS_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)/logs"
VENV_PY="${WG_ROOT}/.venv/bin/python3"
[[ -x "${VENV_PY}" ]] || VENV_PY="python3"

if [[ -f "$MANIFEST" ]]; then
    run_ids=$("${VENV_PY}" "$SCRIPT_DIR/manage_runs.py" --manifest "$MANIFEST" list \
        | awk '{for(i=1;i<=NF;i++) if ($i ~ /^run_id=/) print substr($i,8)}' | sort -u)
else
    run_ids=$(awk -F'|' '{gsub(/ /, "", $2); print $2}' "$LOG_FILE" | sort -u)
fi

echo "$run_ids" | while read -r run_id; do
    slurm_files=("$LOGS_DIR/$run_id"/weathergen.inference.*.out)
    log_files=("$LOGS_DIR/$run_id"/output.inference.*.txt)

    slurm_exists=false
    log_exists=false
    [[ -e "${slurm_files[0]}" ]] && slurm_exists=true
    [[ -e "${log_files[0]}" ]]   && log_exists=true

    if ! $slurm_exists && ! $log_exists; then
        echo "MISSING  $run_id  (no log files found)"
        continue
    fi

    # Determine slurm exit status
    slurm_done=false
    slurm_exit_code=""
    if $slurm_exists; then
        exit_line=$(grep "Finished inference job with exit code:" "${slurm_files[0]}" 2>/dev/null | tail -1)
        if [[ -n "$exit_line" ]]; then
            slurm_exit_code=$(echo "$exit_line" | grep -o '[0-9]*$')
            [[ "$slurm_exit_code" == "0" ]] && slurm_done=true
        fi
    fi

    # Determine whether inference output confirms completion
    inference_done=false
    if $log_exists && grep -ql "Finished inference run with id:" "${log_files[@]}"; then
        inference_done=true
    fi

    ref_file="${slurm_files[0]}"
    $log_exists && ref_file="${log_files[0]}"

    if $slurm_exists && [[ -n "$slurm_exit_code" ]] && [[ "$slurm_exit_code" != "0" ]]; then
        echo "FAILED   $run_id  exit_code=$slurm_exit_code  $ref_file"
    elif $slurm_done && $inference_done; then
        echo "DONE     $run_id  $ref_file"
    elif $slurm_done && ! $inference_done; then
        echo "ERROR    $run_id  (slurm exit=0 but inference output incomplete)  $ref_file"
    else
        echo "RUNNING  $run_id  $ref_file"
    fi
done
