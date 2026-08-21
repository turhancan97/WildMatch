#!/bin/bash
#SBATCH --job-name=czechlynx-prepare
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=c11,c15
#SBATCH --output=logs/czechlynx-prepare/czechlynx-prepare-%j.out
#SBATCH --error=logs/czechlynx-prepare/czechlynx-prepare-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

source_root=${CZECHLYNX_SOURCE_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_v2}
output_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
validation_fraction=${CZECHLYNX_VALIDATION_FRACTION:-0.20}
seed=${CZECHLYNX_SEED:-0}

mkdir -p logs
python -m scripts.czechlynx_dataset \
  --source_root "${source_root}" \
  --output_root "${output_root}" \
  --validation_fraction "${validation_fraction}" \
  --seed "${seed}"
