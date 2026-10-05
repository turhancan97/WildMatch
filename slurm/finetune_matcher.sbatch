#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --exclude=c11,c15,c22
#SBATCH --job-name=finetune_matcher
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x_%j.log
# One matcher fine-tuning run: `sbatch slurm/finetune_matcher.sbatch dataset=salamander matcher_finetune=rdd ...`
# (arguments are Hydra overrides of conf/finetune_matcher.yaml). The recipe trains on 4 GPUs
# (matcher_finetune.num_processes=4); when you request another GPU count, pass the same number as
# matcher_finetune.num_processes and keep num_processes x batch_size (x grad_accum_steps) = 32.
# Submit from the repository root.
set -euo pipefail
nvidia-smi -L

# The wildmatch uv environment with the `train` extra; override with WILDMATCH_ENV.
WILDMATCH_ENV="${WILDMATCH_ENV:-${UV_ENV_ROOT:-/shared/results/common/kargin/projects/uv-environment}/wildmatch}"
source "${WILDMATCH_ENV}/bin/activate"

wildmatch finetune-matcher paths=gmum "$@"
