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

# Descriptor and joint RDD runs feed full-resolution CzechLynx images of varying size, which fragments
# the CUDA caching allocator until a 24 GB RTX 4090 runs out of memory with about 5 GB reserved but
# unused (jobs 527423 and 528063, 2026-10-07/08). Expandable segments only change how memory is
# reserved, not what is computed; set PYTORCH_CUDA_ALLOC_CONF yourself to override.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wildmatch finetune-matcher paths=gmum "$@"
