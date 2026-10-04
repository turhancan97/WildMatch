#!/bin/bash -l
# SAM 3 background masks for `wildmatch prepare build` (A100/H100 only: the SAM 3 environment's
# CUDA 13 PyTorch has no V100 kernels). Pass the arguments that `wildmatch prepare build` prints:
#   sbatch slurm/sam3_masks.sbatch --root ... --csv ... --out-dir ... [--masks-csv ...] --segment
# Override the partition/QOS on the command line (sbatch -p dgxa100 ...); the defaults are GMUM's.
#SBATCH -p dgxh100
#SBATCH --qos=big
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sam3_masks
#SBATCH --time=24:00:00
#SBATCH --output=logs/prepare/%x_%j.out
#SBATCH --export=ALL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run through sbatch}"
# The SAM 3 environment; override with SAM3_ENV (a conda env name or prefix).
source "${CONDA_SH:-/shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh}"
conda activate "${SAM3_ENV:-lynx-app}"
nvidia-smi -L
python src/wildmatch/data/prepare/sam3_masks.py "$@"
