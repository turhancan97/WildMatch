#!/usr/bin/env bash
#SBATCH --job-name=wildlife-rdd-cache
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=logs/wildlife-rdd-cache/wildlife-rdd-cache-%j.out
#SBATCH --error=logs/wildlife-rdd-cache/wildlife-rdd-cache-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

benchmark_root=${WILDLIFE_BENCHMARK_ROOT:-/home/kargin/Projects/repositories/rdd-parallel-benchmark}
config=${WILDLIFE_CONFIG:-${benchmark_root}/configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(cd "${benchmark_root}" && python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
cache_root=${WILDLIFE_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-cache}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
splits=(train test)
if [[ "${protocol}" == strict ]]; then
  splits=(train val test)
fi

python -m contrastive_finetuning.build_keypoint_cache \
  --data_root "${dataset_root}" --cache_root "${cache_root}" \
  --rdd_weights "${rdd_weights}" --splits "${splits[@]}" \
  --resize 512 --top_k 512 \
  --batch_size "${WILDLIFE_RDD_CACHE_BATCH_SIZE:-8}" \
  --num_workers "${WILDLIFE_CACHE_WORKERS:-16}" --resume
