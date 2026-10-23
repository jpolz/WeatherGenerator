#!/bin/bash
#
# submit_ssw_validation.sh
#
# Submit SSW validation inference jobs for standardised lead times relative to each event.
#
# Lead times: t0d–t30d daily (days before the SSW central warming date)
# Rollout:    computed per batch to cover ≥45 days post-SSW from its longest included lead
#
# One job now covers ALL requested leads for a given (model, event) via
# test_config.date_ranges, instead of one job per (model, event, lead). The
# sampler shards the requested init dates chronologically across SLURM ranks
# (world_size = nodes*4, hardcoded by launch-slurm.py), and each rank writes
# its own zarr store. Because of this, the number of leads submitted in one
# batch MUST be a non-zero multiple of WORLD_SIZE below, or the launcher's
# check_samples() assertion will fail (or, for a single lead, silently pad
# samples_per_mini_epoch up and then hit an unsatisfiable n_duplicates
# assertion) — see --leads below.
#
# Usage:
#   ./submit_ssw_validation.sh [options]
#
# Options:
#   --dry-run             Print commands without submitting
#   --model MODEL         Submit only for a specific model key
#   --leads LEADS         Comma-separated list of lead offsets (days before SSW) to submit
#                          as one batch, e.g. "0,1,2,3". Must be a non-zero multiple of
#                          WORLD_SIZE (4). Default: LEAD_MIN..LEAD_MAX (21 values — pass
#                          --leads explicitly, or adjust LEAD_MIN/LEAD_MAX, to get a
#                          multiple of 4)
#   --event EVENT         Submit for a specific event (feb2018, jan2013, jan2019, jan2021)
#   --stream-dir DIR      Override streams_directory passed to inference (whole dir, all streams)
#   --force                Re-submit even if the run-ID manifest already has entries for
#                          every lead in this batch (event, model, leads)
#   --no-channel-filter    Write all target channels (default: trim to the SSW-relevant
#                          write-list below, via test_config.output.channels.*)
#   --account ACCOUNT     Slurm account to submit under (default: weatherai). NOTE: if
#                          the weatherai QOS is suspended (0-node allowance), pass an
#                          active account, e.g. --account e-ext-2025e01-128
#   --help                Show this help message
#
# NOTE: inference must be launched with the current CLI, using a single
#   consolidated date_ranges list (flat list of init dates) rather than a
#   contiguous start_date/end_date range:
#   srun ... inference --from-run-id ${run_id} --options \
#       test_config.date_ranges=[${init_date_1},${init_date_2},...] \
#       test_config.samples_per_mini_epoch=${num_leads} \
#       test_config.output.num_samples=${num_leads / WORLD_SIZE} \
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
SPECIFIC_LEADS=""
SPECIFIC_EVENT="feb2018"
STREAM_DIR=""
ACCOUNT="weatherai"

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
        --leads)      SPECIFIC_LEADS="$2";   shift 2 ;;
        --event)      SPECIFIC_EVENT="$2";   shift 2 ;;
        --stream-dir) STREAM_DIR="$2";       shift 2 ;;
        --account)    ACCOUNT="$2";          shift 2 ;;
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

# Number of SLURM ranks a batch is sharded across (nodes*4, nodes is hardcoded
# to 1 below / in the launch_cmd). Must match launch-slurm.py's inference
# --ntasks formula, or the rank/sample bookkeeping here will be wrong.
WORLD_SIZE=4

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

# Submit ONE consolidated batch (all requested leads for a single model+event)
# as a single inference job with test_config.date_ranges, then upsert one
# manifest entry per lead with its computed rank/sample.
#
# Args: run_id model_name timestep  <space-separated "offset:lead_key:init_date" triples...>
submit_batch() {
    local run_id=$1 model_name=$2 timestep=$3
    shift 3
    local -a triples=("$@")
    local num_leads=${#triples[@]}

    local step_hours=6
    [[ "$timestep" == "24h" ]] && step_hours=24

    # batch-wide fsteps: must cover the longest-lead (largest offset) init date
    local max_offset=-1
    local t
    for t in "${triples[@]}"; do
        local offset=${t%%:*}
        (( offset > max_offset )) && max_offset=$offset
    done
    local fsteps=$(( (max_offset + POST_SSW_DAYS) * 24 / step_hours ))

    # rank/sample assignment: sort by offset descending == chronological
    # ascending order (larger offset == earlier calendar date), matching the
    # sampler's worker_workset() sharding (rank r owns chronological indices
    # [r*per_rank, (r+1)*per_rank)).
    local -a sorted_triples
    mapfile -t sorted_triples < <(printf '%s\n' "${triples[@]}" | sort -t: -k1,1nr)

    local per_rank=$(( num_leads / WORLD_SIZE ))
    local -a date_list=()
    for t in "${sorted_triples[@]}"; do
        local init_date=${t#*:*:}
        date_list+=("${init_date}")
    done
    local date_ranges_opt
    date_ranges_opt=$(IFS=,; echo "${date_list[*]}")

    local streams_dir="${STREAM_DIR:-config/streams/era5_mlpl_strato_inference}"

    echo -e "${BLUE}================================================${NC}"
    echo -e "${GREEN}Submitting batch: ${model_name} | ${SPECIFIC_EVENT} | ${num_leads} leads${NC}"
    echo -e "  From run:    ${run_id}"
    echo -e "  Init dates:  ${date_ranges_opt}"
    echo -e "  Steps:       ${fsteps} at ${timestep}"
    echo -e "  Streams:     ${streams_dir}"
    echo -e "${BLUE}================================================${NC}"

    local launch_cmd=(
        "${LAUNCH_SCRIPT}"
        --stage inference
        --nodes=1
        -t 02:00:00
        --account="${ACCOUNT}"
        --from-run-id "${run_id}"
        --no-register
        --link-venv
        --options
            "test_config.date_ranges=[${date_ranges_opt}]"
            "test_config.output.num_samples=${per_rank}"
            "test_config.samples_per_mini_epoch=${num_leads}"
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
        return 0
    fi

    local launch_out launch_rc=0
    launch_out=$("${launch_cmd[@]}" 2>&1) || launch_rc=$?
    echo "${launch_out}"
    if [[ ${launch_rc} -ne 0 ]]; then
        echo -e "${RED}ERROR: launch-slurm.py failed (exit ${launch_rc}) for ${model_name}/${SPECIFIC_EVENT} batch. Skipping manifest/log update.${NC}"
        return 1
    fi
    # Extract generated run_id from launcher output
    local gen_run_id
    gen_run_id=$(echo "${launch_out}" | grep -oP '(?<=Using generated run id: )\S+')

    # One manifest entry per lead, all sharing gen_run_id, each with its own
    # chronological rank/sample within that run_id's output.
    local rank=0 sample=0
    for t in "${sorted_triples[@]}"; do
        local offset=${t%%:*}
        local rest=${t#*:}
        local lead_key=${rest%%:*}
        local init_date=${rest#*:}

        echo "$(date '+%Y-%m-%d %H:%M:%S') | ${gen_run_id} | ${run_id} | ${model_name} | ${lead_key} | ${init_date} | ${fsteps} | rank=${rank} sample=${sample}" >> "${SCRIPT_DIR}/validation_submissions.log"

        "${VENV_PY}" "${MANAGE_RUNS}" upsert \
            --event "${SPECIFIC_EVENT}" --model "${model_key}" --lead "${lead_key}" \
            --run-id "${gen_run_id}" --model-id "${run_id}" --model-label "${model_name}" \
            --init-date "${init_date}" --fsteps "${fsteps}" \
            --rank "${rank}" --sample "${sample}"

        sample=$((sample + 1))
        if (( sample == per_rank )); then
            sample=0
            rank=$((rank + 1))
        fi
    done
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

# Build the requested offset list: explicit --leads CSV, or the full
# LEAD_MIN..LEAD_MAX range.
declare -a offsets=()
if [[ -n "$SPECIFIC_LEADS" ]]; then
    IFS=',' read -r -a offsets <<< "$SPECIFIC_LEADS"
else
    for offset in $(seq "$LEAD_MIN" "$LEAD_MAX"); do
        offsets+=("$offset")
    done
fi

num_offsets=${#offsets[@]}
if (( num_offsets == 0 || num_offsets % WORLD_SIZE != 0 )); then
    echo -e "${RED}Error: ${num_offsets} leads requested, but a batch must be a non-zero" \
         "multiple of WORLD_SIZE (${WORLD_SIZE}).${NC}"
    echo "Pass --leads with an explicit comma-separated list whose count is a multiple" \
         "of ${WORLD_SIZE}, or adjust LEAD_MIN/LEAD_MAX in this script."
    exit 1
fi

TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0

for model_key in "${!MODELS[@]}"; do
    [[ -n "$SPECIFIC_MODEL" && "$model_key" != "$SPECIFIC_MODEL" ]] && continue

    read -r run_id model_name timestep <<< "${MODELS[$model_key]}"

    declare -a triples=()
    all_exist=true
    for offset in "${offsets[@]}"; do
        lead_key="t${offset}d"
        init_date=$(date_offset "$ssw_date" "$offset")
        triples+=("${offset}:${lead_key}:${init_date}")

        if ! "${VENV_PY}" "${MANAGE_RUNS}" exists \
            --event "${SPECIFIC_EVENT}" --model "${model_key}" --lead "${lead_key}" 2>/dev/null; then
            all_exist=false
        fi
    done

    if [[ "$FORCE" != true && "$all_exist" == true ]]; then
        echo "Skipping ${model_key} batch (all ${num_offsets} leads already in manifest, use --force to re-submit)"
        TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        continue
    fi

    if submit_batch "$run_id" "$model_name" "$timestep" "${triples[@]}"; then
        TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
    else
        echo -e "${RED}Batch submission failed for ${model_key}/${SPECIFIC_EVENT}, continuing with remaining models.${NC}"
    fi
done

# ============================================================================
# SUMMARY
# ============================================================================

echo -e "${BLUE}=======================================================${NC}"
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}DRY RUN COMPLETE — would submit ${TOTAL_SUBMITTED} batch job(s), skip ${TOTAL_SKIPPED}${NC}"
else
    echo -e "${GREEN}SUBMISSION COMPLETE — submitted ${TOTAL_SUBMITTED} batch job(s) (${num_offsets} leads each), skipped ${TOTAL_SKIPPED}${NC}"
    echo ""
    echo "Monitor:      squeue -u \$USER"
    echo "Submissions:  ${SCRIPT_DIR}/validation_submissions.log"
fi
echo -e "${BLUE}=======================================================${NC}"
