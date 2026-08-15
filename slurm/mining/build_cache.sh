#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

dataset_root=$1
cache_dir=$2
top_k=$3
resize_max=$4
sample=$5

python -m scripts.lynx_build_cache \
    --dataset_root ${dataset_root} \
    --cache_dir ${cache_dir} \
    --frames_per_seq ${sample} \
    --top_k ${top_k} \
    --resize_max ${resize_max} \
