#!/bin/bash -x
#SBATCH --account=hclimrep
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --chdir=.
#SBATCH --partition=booster
##SBATCH --partition=develbooster
#SBATCH --output=packages/science/hclimrep_stratosphere/scripts/logs/%x.%j.out
#SBATCH --error=packages/science/hclimrep_stratosphere/scripts/logs/%x.%j.err

# important paths and directories (adapt if required!)
VENV_NAME=.venv
BASE_DIR=${SLURM_SUBMIT_DIR}/
VENV_DIR=${BASE_DIR}

# Load basic modules from software stack
ml --force purge
ml use $OTHERSTAGES
ml Stages/2025

ml GCC/13.3.0
ml GCCcore/.13.3.0

ml OpenMPI/5.0.5

ml git/2.45.1
ml Python/3.12.3

# Activate virtual environment
if [ -z ${VIRTUAL_ENV} ]; then
   if [[ -f ${VENV_DIR}/${VENV_NAME}/bin/activate ]]; then
      echo "Run virtual env via uv."
      source ${VENV_DIR}/${VENV_NAME}/bin/activate
   else
      echo ${VENV_DIR}
      echo "ERROR: Requested virtual environment ${VENV_NAME} not found..."
      exit 1
   fi
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Starting job."
date

srun --label ssw-analyze polar-vortex \
    --validations-config config/evaluate/ssw_jan2021.yml \
    --data-dir results \
    --output-dir plots/tmp_strat/test_plots \
    --channels u_29

echo "Finished job."
date
