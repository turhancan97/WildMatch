#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --exclude=c11,c15,c22
#SBATCH --job-name=mine
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.log
# One `wildmatch mine` step on one GPU: `sbatch slurm/mine.sbatch cache --dataset salamander --backend loma`.
# `wildmatch mine submit ...` submits this script as the per-split query arrays (step `task`) and the
# dependent aggregation. Submit from the repository root.
set -euo pipefail
WILDMATCH_ENV="${WILDMATCH_ENV:-${UV_ENV_ROOT:-/shared/results/common/kargin/projects/uv-environment}/wildmatch}"
source "${WILDMATCH_ENV}/bin/activate"
wildmatch mine "$@"
