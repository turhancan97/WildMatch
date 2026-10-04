#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=finetune_reid
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

nvidia-smi -L

# The wildmatch uv environment (README, "Installation"); override with WILDMATCH_ENV.
WILDMATCH_ENV="${WILDMATCH_ENV:-${UV_ENV_ROOT:-/shared/results/common/kargin/projects/uv-environment}/wildmatch}"
source "${WILDMATCH_ENV}/bin/activate"

python train/finetune.py paths=gmum