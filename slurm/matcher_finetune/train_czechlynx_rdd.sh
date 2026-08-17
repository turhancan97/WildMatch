#!/bin/bash
#SBATCH --job-name=czechlynx-rdd-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-rdd-ft-%j.out
#SBATCH --error=logs/czechlynx-rdd-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd
export WANDB_MODE=online

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
train_index=${CZECHLYNX_TRAIN_INDEX:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/indices/strong_matches_train_combined.json}
val_index=${CZECHLYNX_VAL_INDEX:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/indices/strong_matches_val_combined.json}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
lg_weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
cache_root=${CZECHLYNX_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache}
output_dir=${CZECHLYNX_RDD_OUTPUT:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned}

mkdir -p logs
if [[ ! -f "${cache_root}/manifest.json" ]]; then
  python -m contrastive_finetuning.build_keypoint_cache \
    --data_root "${dataset_root}" --cache_root "${cache_root}" \
    --rdd_weights "${rdd_weights}" --splits train val test \
    --resize 512 --top_k 512 \
    --batch_size "${RDD_CACHE_BATCH_SIZE:-32}" --num_workers 16 --resume
fi

accelerate launch --num_processes 2 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  --train_index "${train_index}" --val_index "${val_index}" \
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}" \
  --lg_weights "${lg_weights}" --output_dir "${output_dir}" \
  --project lynx-czechlynx-rdd --run_name czechlynx-time-closed-rdd \
  --trained_model lg --epochs 300 --batch_size 8 --lr 1e-5 \
  --weight_decay 1e-4 --num_workers 8 --lg_margin 0.5 \
  --random_negative_prob 0.3 --resize 512 --top_k 512 \
  --keypoint_cache "${cache_root}" --seed 0
