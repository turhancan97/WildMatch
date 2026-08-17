#!/bin/bash
#SBATCH --job-name=czechlynx-rdd-cache
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-rdd-cache-%j.out
#SBATCH --error=logs/czechlynx-rdd-cache-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
cache_root=${CZECHLYNX_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}

mkdir -p logs
python -m contrastive_finetuning.build_keypoint_cache \
  --data_root "${dataset_root}" \
  --cache_root "${cache_root}" \
  --rdd_weights "${rdd_weights}" \
  --splits train val test \
  --resize 512 \
  --top_k 512 \
  --batch_size "${RDD_CACHE_BATCH_SIZE:-4}" \
  --num_workers 16 \
  --resume
