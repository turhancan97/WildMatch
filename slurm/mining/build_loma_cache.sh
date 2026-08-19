#!/bin/bash -l
#SBATCH --job-name=lynx-loma-cache
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00
#SBATCH --partition=rtx4090_batch
#SBATCH --exclude=c11,c15
#SBATCH --qos=batch
#SBATCH --output=logs/loma-cache-%j.out
#SBATCH --error=logs/loma-cache-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma

dataset_root=$1
cache_dir=$2
weights=$3
variant=${4:-loma-b}
sample=${5:-20}
resize_max=${6:-512}
num_keypoints=${7:-512}

python -m scripts.lynx_build_loma_cache \
    --dataset_root "${dataset_root}" \
    --cache_dir "${cache_dir}" \
    --weights "${weights}" \
    --variant "${variant}" \
    --frames_per_seq "${sample}" \
    --resize_max "${resize_max}" \
    --num_keypoints "${num_keypoints}"
