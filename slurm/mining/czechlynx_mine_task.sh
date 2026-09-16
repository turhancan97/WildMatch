#!/bin/bash
#SBATCH --job-name=czechlynx-mine
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=c11,c15
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-mine/czechlynx-mine-%A_%a.out
#SBATCH --error=logs/czechlynx-mine/czechlynx-mine-%A_%a.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
backend=${7:-rdd}
variant=${8:-loma-b}
if [[ "${backend}" == "loma" ]]; then conda activate loma; else conda activate rdd; fi

dataset_root=${1:?canonical CzechLynx root is required}
cache_dir=${2:?RDD feature cache is required}
rdd_weights=${3:?RDD weights are required}
lg_weights=${4:?LightGlue weights are required}
dump_report=${5:?report prefix is required}
split=${6:?query split is required}
weights=${4}

python -m scripts.czechlynx_mine \
  --dataset_root "${dataset_root}" \
  --cache_dir "${cache_dir}" \
  --rdd_weights "${rdd_weights}" \
  --lg_weights "${lg_weights}" \
  --weights "${weights}" \
  --backend "${backend}" \
  --variant "${variant}" \
  --split "${split}" \
  --query_id "${SLURM_ARRAY_TASK_ID}" \
  --frames_per_collection "${CZECHLYNX_FRAMES_PER_COLLECTION:-20}" \
  --top_k_frames "${CZECHLYNX_TOP_K_FRAMES:-5}" \
  --top_m "${CZECHLYNX_TOP_M:-10}" \
  --dump_report "${dump_report}"
