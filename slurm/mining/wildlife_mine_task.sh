#!/usr/bin/env bash
#SBATCH --job-name=wildlife-mine
#SBATCH --output=logs/wildlife-mine/wildlife-mine-%j.out
#SBATCH --error=logs/wildlife-mine/wildlife-mine-%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=c11,c15
#SBATCH --time=23:59:00

set -euo pipefail
dataset_id=${1:?dataset id is required}
dataset_root=${2:?canonical dataset root is required}
cache_dir=${3:?feature cache is required}
weights=${4:?matcher checkpoint is required}
dump_report=${5:?report prefix is required}
split=${6:?query split is required}
backend=${7:-rdd}
variant=${8:-loma-b}
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
if [[ "${backend}" == loma ]]; then conda activate loma; else conda activate rdd; fi

python -m scripts.wildlife_mine \
  --dataset_id "${dataset_id}" \
  --dataset_root "${dataset_root}" \
  --cache_dir "${cache_dir}" \
  --lg_weights "${weights}" \
  --backend "${backend}" \
  --variant "${variant}" \
  --split "${split}" \
  --query_id "${SLURM_ARRAY_TASK_ID}" \
  --frames_per_collection "${WILDLIFE_FRAMES_PER_COLLECTION:-20}" \
  --top_k_frames "${WILDLIFE_TOP_K_FRAMES:-5}" \
  --top_m "${WILDLIFE_TOP_M:-10}" \
  --dump_report "${dump_report}"
