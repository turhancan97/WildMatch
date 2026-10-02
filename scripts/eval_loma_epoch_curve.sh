#!/bin/bash -l
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --exclude=c13,c22,c15,dgx1
#SBATCH --job-name=loma_epoch_curve
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/compute_efficiency/%x_%A_%a.out
#SBATCH --error=logs/compute_efficiency/%x_%A_%a.err
#SBATCH --export=ALL
#
# Training-cost ablation, LoMa arm: evaluate the intermediate checkpoints of the
# CzechLynx closed fine-tuned LoMa run (epoch 299 is the paper run
# 20260920T122915Z_0015f14a) with exactly that run's configuration. The only
# changes are the checkpoint path, a separate feature-cache directory, which can be
# deleted afterwards (about 20 GB per epoch), and the output locations: runs, the
# legacy aggregate CSV and the run index go under experiments/compute-efficiency/,
# because the paper-table exporter keys runs by checkpoint label ("custom") and k
# only, so an intermediate epoch under experiments/probe/ would replace epoch 299 in
# the tables. Test-split curves only: no epoch is selected from them.
#
# Submit from the repository root:
#   mkdir -p logs/compute_efficiency && sbatch scripts/eval_loma_epoch_curve.sh

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

EPOCHS=(000 050 100 150 200 250)
EPOCH="${EPOCHS[${SLURM_ARRAY_TASK_ID:-0}]}"
# Immutable config copy of the submission that produced the epoch-299 run.
CONFIG_DIR="${REPO}/logs/parallel_run/submissions/20260920T111901Z-2476485"
CHECKPOINT_ROOT=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-finetuned-loma-mined-legacy
CHECKPOINT="${CHECKPOINT_ROOT}/epoch_${EPOCH}/model.safetensors"
OUT_ROOT=experiments/compute-efficiency
CACHE_DIR="${LOMA_EPOCH_CURVE_CACHE:-/shared/results/common/kargin/lynx/results/CzechLynx_v2/CzechLynx/cache/vismatch_epoch_curve}"

[[ -f "${CONFIG_DIR}/probe.yaml" ]] || { echo "missing config copy ${CONFIG_DIR}/probe.yaml" >&2; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "missing checkpoint ${CHECKPOINT}" >&2; exit 1; }

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate ex-reid
nvidia-smi -L
echo "[loma-epoch-curve] epoch ${EPOCH} checkpoint ${CHECKPOINT} sha256 $(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)"

python train/probe.py --config-dir "${CONFIG_DIR}" --config-name probe \
    dataset.name=CzechLynx_v2 dataset.animal=CzechLynx \
    dataset.root=/shared/sets/datasets/vision/czechlynx/CzechLynx_v2 \
    dataset.metadata_file=CzechLynxDataset-Metadata-Real.csv dataset.label_col=unique_name \
    dataset.mask_col=mask dataset.no_background=true dataset.image_variant=no_background \
    dataset.split_col=split-time_closed dataset.database_split_value=train \
    dataset.query_split_value=test dataset.calibration_size=100 \
    benchmark.method=vismatch benchmark.candidate_k=250 \
    benchmark.methods.vismatch.matcher=loma benchmark.methods.vismatch.loma_arch=LoMa-B \
    benchmark.methods.vismatch.checkpoint_source=custom \
    "benchmark.methods.vismatch.checkpoint_path=${CHECKPOINT}" \
    benchmark.methods.vismatch.checkpoint_components=matcher_only \
    "benchmark.methods.vismatch.cache_dir=${CACHE_DIR}" \
    "output.experiment_root=${OUT_ROOT}" "output.csv_path=${OUT_ROOT}/benchmark_results.csv" \
    "output.run_dir=${OUT_ROOT}/benchmark_runs" "reporting.index_path=${OUT_ROOT}/runs.csv" \
    "wandb.name=probe-czechlynx-closed-loma-epoch-curve-e${EPOCH}"
