#!/bin/bash
#SBATCH --job-name=czechlynx-loma-cache
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=c11,c15
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-loma-cache-%j.out
#SBATCH --error=logs/czechlynx-loma-cache-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
cache_dir=${CZECHLYNX_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-cache}
loma_weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
benchmark_root=/home/kargin/Projects/repositories/rdd-parallel-benchmark

mkdir -p logs
cd "${benchmark_root}"
python -m scripts.lynx_build_loma_cache \
  --dataset_root "${dataset_root}" \
  --cache_dir "${cache_dir}" \
  --weights "${loma_weights}" \
  --variant loma-b \
  --all_frames \
  --splits train val test \
  --resize_max 512 \
  --num_keypoints 512 \
  --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}" \
  --num_workers 16 \
  --resume
