#!/bin/bash
#
# submit_ssw_validation.sh
#
# Submit SSW validation inference jobs for standardised lead times relative to each event.
#
# Lead times: t15d, t10d, t5d, t0d (days before the SSW central warming date)
# Rollout:    240 steps × 6h = 60 days (covers ≥45 days post-SSW from any lead)
#
# Usage:
#   ./submit_ssw_validation.sh [options]
#
# Options:
#   --dry-run        Print commands without submitting
#   --model MODEL    Submit only for a specific model key
#   --lead LEAD      Submit only for a specific lead (t15d, t10d, t5d, t0d)
#   --event EVENT    Submit for a specific event (feb2018, jan2013, jan2019, jan2021)
#   --help           Show this help message
#
# NOTE: weathergen_validate_jwb_batch.sh must use the current inference CLI:
#   srun ... inference --from-run-id ${run_id} --options \
#       test_config.start_date=${start_date} \
#       test_config.end_date=${end_date} \
#       test_config.samples_per_mini_epoch=${samples} \
#       test_config.forecast.num_steps=${fsteps}
#

set -e

# Default settings
DRY_RUN=false
SPECIFIC_MODEL=""
SPECIFIC_LEAD=""
SPECIFIC_EVENT="feb2018"

VALIDATION_SCRIPT="../WeatherGenerator-private/hpc/juwels_booster/jsc/weathergen_validate_jwb_jpstrat.sh"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=true;          shift ;;
        --model)     SPECIFIC_MODEL="$2";   shift 2 ;;
        --lead)      SPECIFIC_LEAD="$2";    shift 2 ;;
        --event)     SPECIFIC_EVENT="$2";   shift 2 ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATIONS
# ============================================================================

# Models: key → "run_id label timestep"
declare -A MODELS=(
    ["bg03rub9"]="bg03rub9 strato_ft 6h"
)

# SSW central warming dates
declare -A SSW_DATES=(
    ["feb2018"]="2018-02-12"
    ["jan2013"]="2013-01-06"
    ["jan2019"]="2019-01-02"
    ["jan2021"]="2021-01-05"
)

# Standardised lead offsets in days before SSW date
declare -A LEAD_OFFSETS=(
    ["t15d"]=15
    ["t10d"]=10
    ["t5d"]=5
    ["t0d"]=0
)

SAMPLES=1
# 240 steps × 6h = 60 days rollout → ≥45 days post-SSW even from t15d init
FSTEPS_6H=180
FSTEPS_24H=45

# ============================================================================
# HELPERS
# ============================================================================

# Compute init date: date_offset <YYYY-MM-DD> <offset_days> → YYYYMMDDhhmm
date_offset() {
    date -d "${1} -${2} days" "+%Y%m%d0000"
}

submit_validation() {
    local run_id=$1 model_name=$2 lead_key=$3 init_date=$4 fsteps=$5 timestep=$6

    local job_name="val_${model_name}_${lead_key}_${SPECIFIC_EVENT}"
    # Human-readable tag stored in general.desc of the output model JSON
    local desc="${SPECIFIC_EVENT}_${lead_key}_from_${run_id}"

    # end_date must cover start_date plus the full forecast horizon so the
    # data reader can load all target timestamps. Compute days needed from
    # fsteps × timestep, then add 2 days buffer.
    local step_hours=6
    [[ "$timestep" == "24h" ]] && step_hours=24
    local days_needed=$(( (fsteps * step_hours + 23) / 24 + 2 ))
    local end_date
    end_date=$(date -d "${init_date:0:8} +${days_needed} days" "+%Y%m%d0000")

    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}Submitting: ${model_name} | ${lead_key} | ${SPECIFIC_EVENT}${NC}"
    echo -e "  From run:  ${run_id}"
    echo -e "  Desc tag:  ${desc}"
    echo -e "  Init date: ${init_date}"
    echo -e "  End date:  ${end_date}"
    echo -e "  Steps:     ${fsteps} at ${timestep}"
    echo -e "${BLUE}================================================${NC}"

    local sbatch_cmd=(
        sbatch
        --job-name="${job_name}"
        --output="./logs/${job_name}.%j.out"
        --error="./logs/${job_name}.%j.err"
        "${VALIDATION_SCRIPT}"
        --run_id     "${run_id}"
        --samples    "${SAMPLES}"
        --start_date "${init_date}"
        --end_date   "${end_date}"
        --fsteps     "${fsteps}"
        --desc       "${desc}"
    )

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] ${sbatch_cmd[*]}${NC}\n"
    else
        local job_id
        job_id=$("${sbatch_cmd[@]}" | awk '{print $NF}')
        echo -e "${GREEN}✓ SLURM ID: ${job_id}${NC}\n"
        echo "$(date '+%Y-%m-%d %H:%M:%S') | ${job_id} | ${run_id} | ${model_name} | ${lead_key} | ${init_date} | ${fsteps} | ${desc}" >> validation_submissions.log
        sleep 1
    fi
}

# ============================================================================
# MAIN
# ============================================================================

echo -e "${BLUE}"
echo "======================================================="
echo "  SSW Validation Batch Submission"
echo "======================================================="
echo -e "${NC}"
echo "Event:   ${SPECIFIC_EVENT}"
echo "Dry Run: ${DRY_RUN}"
echo ""

ssw_date="${SSW_DATES[$SPECIFIC_EVENT]}"
if [[ -z "$ssw_date" ]]; then
    echo -e "${RED}Error: Unknown event '${SPECIFIC_EVENT}'. Choose from: ${!SSW_DATES[*]}${NC}"
    exit 1
fi

TOTAL_SUBMITTED=0

for model_key in "${!MODELS[@]}"; do
    [[ -n "$SPECIFIC_MODEL" && "$model_key" != "$SPECIFIC_MODEL" ]] && continue

    read -r run_id model_name timestep <<< "${MODELS[$model_key]}"
    [[ "$timestep" == "6h" ]] && fsteps=$FSTEPS_6H || fsteps=$FSTEPS_24H

    for lead_key in "${!LEAD_OFFSETS[@]}"; do
        [[ -n "$SPECIFIC_LEAD" && "$lead_key" != "$SPECIFIC_LEAD" ]] && continue

        offset="${LEAD_OFFSETS[$lead_key]}"
        init_date=$(date_offset "$ssw_date" "$offset")

        submit_validation "$run_id" "$model_name" "$lead_key" "$init_date" "$fsteps" "$timestep"
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}=======================================================${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}DRY RUN COMPLETE — would submit ${TOTAL_SUBMITTED} jobs${NC}"
else
    echo -e "${GREEN}SUBMISSION COMPLETE — submitted ${TOTAL_SUBMITTED} jobs${NC}"
    echo ""
    echo "Monitor:      squeue -u \$USER"
    echo "Logs:         ./logs/"
    echo "Submissions:  validation_submissions.log"
fi
echo -e "${BLUE}=======================================================${NC}"
