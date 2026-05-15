#!/bin/bash
# ============================================================================
# submit_ssw_analyze.sh
#
# Run one or more ssw-analyze subcommands on JUWELS Booster.
# Submit from the WeatherGenerator repo root:
#
#   sbatch --job-name=ssw_feb2018 \
#       packages/science/hclimrep_stratosphere/scripts/submit_ssw_analyze.sh \
#       [--config CONFIG] [--data-dir DIR] [--output-dir DIR] \
#       [--run SUBCOMMAND [SUBCOMMAND...]] [--dry-run] [--devel]
#
# Subcommands (default: all):
#   polar-vortex         Zonal mean u-wind at 60°N, SSW detection
#   ssw-lead-times       Prediction skill vs lead time
#   polar-maps           Polar stereographic animations + surface impact
#   vertical-structure   Height-time cross-sections
#
# Options:
#   --config PATH        Validations config YAML  [default: config/evaluate/ssw_feb2018.yml]
#   --data-dir DIR       Zarr results directory   [default: results]
#   --output-dir DIR     Plot output directory    [default: plots/ssw_analyze]
#   --climatology PATH   Climatology zarr for anomaly computation (optional)
#   --run SUBCMD...      Subcommands to run       [default: all four]
#   --polar-vortex-extra Extra args for polar-vortex (e.g. '--channels u_29 u_30')
#   --dry-run            Print commands, do not execute
#   --devel              Use develbooster partition (short jobs)
#   --help               Show this help
# ============================================================================
#SBATCH --account=hclimrep
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=packages/science/hclimrep_stratosphere/scripts/logs/%x.%j.out
#SBATCH --error=packages/science/hclimrep_stratosphere/scripts/logs/%x.%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CONFIG="config/evaluate/ssw_feb2018.yml"
DATA_DIR="results"
OUTPUT_DIR="plots/ssw_analyze"
CLIMATOLOGY=""
DRY_RUN=false
POLAR_VORTEX_EXTRA=""

# Subcommands to run (populated from --run or defaults to all)
declare -a RUN_CMDS=()

# ---------------------------------------------------------------------------
# Argument parsing  (works both as sbatch script and direct bash invocation)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2";            shift 2 ;;
        --data-dir)         DATA_DIR="$2";           shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";         shift 2 ;;
        --climatology)      CLIMATOLOGY="$2";        shift 2 ;;
        --polar-vortex-extra) POLAR_VORTEX_EXTRA="$2"; shift 2 ;;
        --run)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RUN_CMDS+=("$1"); shift
            done ;;
        --dry-run)          DRY_RUN=true;            shift ;;
        --devel)
            # Switch partition to develbooster (max 2h, useful for testing)
            sed -i 's/^#SBATCH --partition=booster/#SBATCH --partition=booster/' "$0" 2>/dev/null || true
            shift ;;
        --help|-h)
            grep "^#" "$0" | grep -v "^#!/\|^#SBATCH" | sed 's/^# \{0,2\}//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Default: run all four subcommands
if [[ ${#RUN_CMDS[@]} -eq 0 ]]; then
    RUN_CMDS=(polar-vortex ssw-lead-times polar-maps vertical-structure)
fi

# ---------------------------------------------------------------------------
# Environment setup (only needed when running inside a SLURM job)
# ---------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    VENV_DIR="${SLURM_SUBMIT_DIR}"

    ml --force purge
    ml use "$OTHERSTAGES"
    ml Stages/2025
    ml GCC/13.3.0
    ml GCCcore/.13.3.0
    ml OpenMPI/5.0.5
    ml git/2.45.1
    ml Python/3.12.3

    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -f "${VENV_DIR}/.venv/bin/activate" ]]; then
            source "${VENV_DIR}/.venv/bin/activate"
        else
            echo "ERROR: .venv not found in ${VENV_DIR}" >&2
            exit 1
        fi
    fi

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

    RUNNER="srun --label"
else
    # Interactive / local run — just use the current Python env
    RUNNER=""
fi

mkdir -p packages/science/hclimrep_stratosphere/scripts/logs

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_cmd() {
    local subcmd="$1"; shift
    local extra_args=("$@")

    local cmd=(
        ${RUNNER} ssw-analyze "${subcmd}"
        --validations-config "${CONFIG}"
        --data-dir            "${DATA_DIR}"
        --output-dir          "${OUTPUT_DIR}/${subcmd}"
        "${extra_args[@]}"
    )

    echo ""
    echo ">>> ${cmd[*]}"
    if [[ "${DRY_RUN}" == false ]]; then
        "${cmd[@]}"
    fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "======================================================="
echo "  ssw-analyze batch run"
echo "  Config:     ${CONFIG}"
echo "  Data dir:   ${DATA_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Climatology:${CLIMATOLOGY:-none}"
echo "  Commands:   ${RUN_CMDS[*]}"
echo "  Dry run:    ${DRY_RUN}"
echo "======================================================="
date

for subcmd in "${RUN_CMDS[@]}"; do
    case "${subcmd}" in
        polar-vortex)
            pv_args=()
            [[ -n "${CLIMATOLOGY:-}" ]] && pv_args+=(--climatology "${CLIMATOLOGY}")
            # shellcheck disable=SC2086
            run_cmd polar-vortex "${pv_args[@]}" ${POLAR_VORTEX_EXTRA} ;;
        ssw-lead-times)
            run_cmd ssw-lead-times ;;
        polar-maps)
            run_cmd polar-maps --model-level 29 --fps 10 --skip-animation ;;
        vertical-structure)
            run_cmd vertical-structure ;;
        *)
            echo "WARNING: unknown subcommand '${subcmd}', skipping" >&2 ;;
    esac
done

echo ""
echo "======================================================="
echo "  Done."
date
