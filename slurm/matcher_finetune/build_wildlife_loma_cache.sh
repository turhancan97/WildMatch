#!/usr/bin/env bash
#SBATCH --job-name=wildlife-loma-cache
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=logs/wildlife-loma-cache/wildlife-loma-cache-%j.out
#SBATCH --error=logs/wildlife-loma-cache/wildlife-loma-cache-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

benchmark_root=${WILDLIFE_BENCHMARK_ROOT:-/home/kargin/Projects/repositories/rdd-parallel-benchmark}
config=${WILDLIFE_CONFIG:-${benchmark_root}/configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(cd "${benchmark_root}" && python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
cache_dir=${WILDLIFE_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/loma-cache}
loma_weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
splits=(train test)
if [[ "${protocol}" == strict ]]; then
  splits=(train val test)
fi

(cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
  --dataset_root "${dataset_root}" --cache_dir "${cache_dir}" \
  --weights "${loma_weights}" --variant "${WILDLIFE_LOMA_VARIANT:-loma-b}" \
  --all_frames --splits "${splits[@]}" --resize_max 512 --num_keypoints 512 \
  --batch_size "${WILDLIFE_LOMA_CACHE_BATCH_SIZE:-4}" \
  --num_workers "${WILDLIFE_CACHE_WORKERS:-16}" --resume)
