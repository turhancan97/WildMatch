#!/usr/bin/env bash
#SBATCH --job-name=wildlife-prepare
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/wildlife-prepare/wildlife-prepare-%j.out
#SBATCH --error=logs/wildlife-prepare/wildlife-prepare-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

config=${WILDLIFE_CONFIG:-configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
output_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}

echo "Preparing ${WILDLIFE_DATASET_ID} with ${protocol} protocol"
echo "config=${config}"
echo "output_root=${output_root}"
python -m scripts.wildlife_dataset \
  --config "${config}" \
  --output_root "${output_root}" \
  --protocol "${protocol}" \
  ${WILDLIFE_FORCE:+--force}
