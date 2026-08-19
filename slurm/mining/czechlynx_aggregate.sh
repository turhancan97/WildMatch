#!/bin/bash
#SBATCH --job-name=czechlynx-aggregate
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=c11,c15
#SBATCH --output=logs/czechlynx-aggregate-%j.out
#SBATCH --error=logs/czechlynx-aggregate-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

dump_report=${1:?report prefix is required}
dataset_root=${2:?canonical CzechLynx root is required}

python -m scripts.czechlynx_aggregate \
  --dump_report "${dump_report}" \
  --dataset_root "${dataset_root}" \
  --splits train val test
