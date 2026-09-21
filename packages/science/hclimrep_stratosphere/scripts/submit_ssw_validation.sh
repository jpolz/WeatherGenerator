#!/bin/bash
#
# submit_ssw_validation.sh
#
# Submit SSW validation inference jobs for standardised lead times relative to each event.
#
# Lead times: t0d–t30d daily (days before the SSW central warming date)
# Rollout:    computed per lead to cover ≥45 days post-SSW
#
# Usage:
#   ./submit_ssw_validation.sh [options]
#
# Options:
#   --dry-run             Print commands without submitting
#   --model MODEL         Submit only for a specific model key
#   --lead LEAD           Submit only for a specific lead (t15d, t10d, t5d, t0d)
#   --event EVENT         Submit for a specific event (feb2018, jan2013, jan2019, jan2021)
#   --stream-dir DIR      Override streams_directory passed to inference (whole dir, all streams)
#   --force                Re-submit even if the run-ID manifest already has an entry for
#                          this (event, model, lead)
#   --no-channel-filter    Write all target channels (default: trim to the SSW-relevant
#                          write-list below, via test_config.output.channels.*)
#   --help                Show this help message
#
# NOTE: weathergen_validate_jwb_batch.sh must use the current inference CLI:
#   srun ... inference --from-run-id ${run_id} --options \
#       test_config.start_date=${start_date} \
#       test_config.end_date=${end_date} \
#       test_config.samples_per_mini_epoch=${samples} \
#       test_config.forecast.num_steps=${fsteps}
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WG_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/logs"

MANAGE_RUNS="${SCRIPT_DIR}/manage_runs.py"
VENV_PY="${WG_ROOT}/.venv/bin/python3"
[[ -x "${VENV_PY}" ]] || VENV_PY="python3"

# Default settings
DRY_RUN=false
FORCE=false
NO_CHANNEL_FILTER=false
SPECIFIC_MODEL=""
SPECIFIC_LEAD=""
SPECIFIC_EVENT="feb2018"
STREAM_DIR=""

LAUNCH_SCRIPT="${WG_ROOT}/../WeatherGenerator-private/hpc/launch-slurm.py"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)    DRY_RUN=true;          shift ;;
        --force)      FORCE=true;            shift ;;
        --no-channel-filter) NO_CHANNEL_FILTER=true; shift ;;
        --model)      SPECIFIC_MODEL="$2";   shift 2 ;;
        --lead)       SPECIFIC_LEAD="$2";    shift 2 ;;
        --event)      SPECIFIC_EVENT="$2";   shift 2 ;;
        --stream-dir) STREAM_DIR="$2";       shift 2 ;;
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
    ["x1menaw0"]="x1menaw0 strato 6h"
    ["kzt66ihm"]="kzt66ihm strato 6h latweight"
    ["qz7qhgyq"]="qz7qhgyq strato 6h"
)

# SSW central warming dates (feb2016 is a no-SSW control, same calendar anchor as feb2018)
declare -A SSW_DATES=(
    ["feb2018"]="2018-02-12"
    ["feb2016"]="2016-02-12"
    ["jan2013"]="2013-01-06"
    ["jan2019"]="2019-01-02"
    ["jan2021"]="2021-01-05"
)

# Lead range: daily increments (days before SSW central warming date)
LEAD_MIN=0
LEAD_MAX=20
# Days of forecast coverage required beyond the SSW date
POST_SSW_DAYS=45

SAMPLES=1

# Output-channel write-lists (disk-space trimming, Phase E). Applied as a
# per-run test_config.output.channels.* override (NOT written into
# config_final.yml), so training/other configs are unaffected. Drops all
# ERA5ml q_* channels and all ERA5pl v_*/q_* channels — unreferenced by any
# current stratosphere analysis script.
ERA5ML_WRITE_CHANNELS="2t,10u,10v,\
u_5,u_12,u_19,u_25,u_29,u_32,u_36,u_40,u_44,u_48,u_51,u_54,\
u_55,u_58,u_61,u_64,u_67,u_70,u_73,u_76,u_80,u_84,u_88,u_92,u_96,u_100,u_105,u_110,u_115,u_120,u_125,u_130,u_137,\
v_5,v_12,v_19,v_25,v_29,v_32,v_36,v_40,v_44,v_48,v_51,v_54,\
v_55,v_58,v_61,v_64,v_67,v_70,v_73,v_76,v_80,v_84,v_88,v_92,v_96,v_100,v_105,v_110,v_115,v_120,v_125,v_130,v_137,\
t_5,t_12,t_19,t_25,t_29,t_32,t_36,t_40,t_44,t_48,t_51,t_54,\
t_55,t_58,t_61,t_64,t_67,t_70,t_73,t_76,t_80,t_84,t_88,t_92,t_96,t_100,t_105,t_110,t_115,t_120,t_125,t_130,t_137"
ERA5PL_WRITE_CHANNELS="z_50,z_500,z_850,u_50,u_500,u_850,t_50,t_500,t_850"

# ============================================================================
# HELPERS
# ============================================================================

# Compute init date: date_offset <YYYY-MM-DD> <offset_days> → YYYYMMDDhhmm
date_offset() {
    date -d "${1} -${2} days" "+%Y%m%d0000"
}

submit_validation() {
    local run_id=$1 model_name=$2 lead_key=$3 init_date=$4 fsteps=$5 timestep=$6

    # end_date must cover start_date plus the full forecast horizon so the
    # data reader can load all target timestamps. Compute days needed from
    # fsteps × timestep, then add 2 days buffer.
    local step_hours=6
    [[ "$timestep" == "24h" ]] && step_hours=24
    local days_needed=$(( (fsteps * step_hours + 23) / 24 + 2 ))
    local end_date
    end_date=$(date -d "${init_date:0:8} +${days_needed} days" "+%Y%m%d0000")

    local streams_dir="${STREAM_DIR:-config/streams/era5_mlpl_strato_inference}"

    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}Submitting: ${model_name} | ${lead_key} | ${SPECIFIC_EVENT}${NC}"
    echo -e "  From run:   ${run_id}"
    echo -e "  Init date:  ${init_date}"
    echo -e "  End date:   ${end_date}"
    echo -e "  Steps:      ${fsteps} at ${timestep}"
    echo -e "  Streams:    ${streams_dir}"
    echo -e "${BLUE}================================================${NC}"

    local launch_cmd=(
        "${LAUNCH_SCRIPT}"
        --stage inference
        --nodes=1
        -t 02:00:00
        --account=weatherai
        --from-run-id "${run_id}"
        --no-register
        --link-venv
        --options
            "test_config.start_date=${init_date}"
            "test_config.end_date=${end_date}"
            "test_config.output.num_samples=${SAMPLES}"
            "test_config.samples_per_mini_epoch=${SAMPLES}"
            "test_config.forecast.num_steps=${fsteps}"
            "streams_directory=${streams_dir}"
    )
    if [[ "$NO_CHANNEL_FILTER" != true ]]; then
        launch_cmd+=(
            "test_config.output.channels.ERA5ml=[${ERA5ML_WRITE_CHANNELS}]"
            "test_config.output.channels.ERA5pl=[${ERA5PL_WRITE_CHANNELS}]"
        )
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] ${launch_cmd[*]}${NC}\n"
    else
        local launch_out
        launch_out=$("${launch_cmd[@]}" 2>&1)
        echo "${launch_out}"
        # Extract generated run_id from launcher output
        local gen_run_id
        gen_run_id=$(echo "${launch_out}" | grep -oP '(?<=Using generated run id: )\S+')
        echo "$(date '+%Y-%m-%d %H:%M:%S') | ${gen_run_id} | ${run_id} | ${model_name} | ${lead_key} | ${init_date} | ${fsteps}" >> "${SCRIPT_DIR}/validation_submissions.log"

        "${VENV_PY}" "${MANAGE_RUNS}" upsert \
            --event "${SPECIFIC_EVENT}" --model "${model_key}" --lead "${lead_key}" \
            --run-id "${gen_run_id}" --model-id "${run_id}" --model-label "${model_name}" \
            --init-date "${init_date}" --fsteps "${fsteps}"
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
    step_h=6; [[ "$timestep" == "24h" ]] && step_h=24

    for offset in $(seq "$LEAD_MIN" "$LEAD_MAX"); do
        lead_key="t${offset}d"
        [[ -n "$SPECIFIC_LEAD" && "$lead_key" != "$SPECIFIC_LEAD" ]] && continue

        if [[ "$FORCE" != true ]] && "${VENV_PY}" "${MANAGE_RUNS}" exists \
            --event "${SPECIFIC_EVENT}" --model "${model_key}" --lead "${lead_key}" 2>/dev/null; then
            echo "Skipping ${lead_key} (already in manifest, use --force to re-submit)"
            continue
        fi

        init_date=$(date_offset "$ssw_date" "$offset")
        fsteps=$(( (offset + POST_SSW_DAYS) * 24 / step_h ))

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
    echo "Submissions:  ${SCRIPT_DIR}/validation_submissions.log"
fi
echo -e "${BLUE}=======================================================${NC}"
