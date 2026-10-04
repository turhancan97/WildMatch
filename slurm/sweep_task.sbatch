#!/bin/bash -l
# One element of a sweep array. Submitted by `wildmatch sweep <spec> --submit`, which freezes
# the submission and passes WILDMATCH_SWEEP_MANIFEST; the array index selects the task.
# The #SBATCH lines are the GMUM cluster defaults; override them with
# `wildmatch sweep ... --submit --sbatch-arg=--partition=<name>` (command-line options win).
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c13,c15,dgx1
#SBATCH --job-name=wildmatch_sweep
#SBATCH --time=24:00:00
#SBATCH --output=logs/parallel_run/%x_%A_%a.out
#SBATCH --error=logs/parallel_run/%x_%A_%a.err
#SBATCH --export=ALL

set -euo pipefail
# Slurm runs a copy of this script from its spool directory; work from the submission directory.
cd "${SLURM_SUBMIT_DIR:?run through sbatch}"
: "${WILDMATCH_SWEEP_MANIFEST:?submit with wildmatch sweep <spec> --submit}"

# The wildmatch uv environment (README, "Installation"); override with WILDMATCH_ENV.
WILDMATCH_ENV="${WILDMATCH_ENV:-${UV_ENV_ROOT:-/shared/results/common/kargin/projects/uv-environment}/wildmatch}"
source "${WILDMATCH_ENV}/bin/activate"
nvidia-smi -L || true
exec wildmatch sweep-task --manifest "${WILDMATCH_SWEEP_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}"
