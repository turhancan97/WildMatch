#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgittossl2-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

source helios_scripts/_activate.sh

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
