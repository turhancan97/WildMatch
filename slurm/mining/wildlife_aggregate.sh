#!/usr/bin/env bash
#SBATCH --job-name=wildlife-aggregate
#SBATCH --output=logs/wildlife-aggregate/wildlife-aggregate-%j.out
#SBATCH --error=logs/wildlife-aggregate/wildlife-aggregate-%j.err
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

report=${1:?report prefix is required}
dataset_root=${2:?canonical dataset root is required}
dataset_id=${3:?dataset id is required}
protocol=${4:?protocol is required}
python -m scripts.wildlife_aggregate \
  --dataset_id "${dataset_id}" \
  --protocol "${protocol}" \
  --dump_report "${report}" \
  --dataset_root "${dataset_root}"
