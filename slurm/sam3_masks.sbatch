#!/bin/bash -l
# SAM 3 background masks for `wildmatch prepare build`. The SAM 3 environment's CUDA 13 PyTorch
# (lynx-app: 2.13.0+cu130) ships kernels for sm_75/80/86/90/100/120, so V100s (sm_70) cannot run
# it; RTX 4090s (sm_89) use the sm_86 kernels. Pass the arguments that `wildmatch prepare build` prints:
#   sbatch slurm/sam3_masks.sbatch --root ... --csv ... --out-dir ... [--masks-csv ...] --segment
# Override the partition/QOS on the command line (sbatch -p dgxh100 --qos=big ...); the defaults are GMUM's.
#SBATCH -p rtx4090_batch
#SBATCH --qos=batch
#SBATCH --exclude=c22,c11,c13,c15,dgx1
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
