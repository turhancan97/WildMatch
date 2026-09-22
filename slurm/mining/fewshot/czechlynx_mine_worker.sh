#!/usr/bin/env bash
#SBATCH --job-name=fs-cz-mine
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=c11
#SBATCH --output=logs/fewshot/%x-%A_%a.out
#SBATCH --error=logs/fewshot/%x-%A_%a.err
#
# One worker of the CzechLynx few-shot mining: array task i of N mines every query
# collection of one split with id % N == i (scripts.czechlynx_mine --all_queries), loading the
# matcher and the train gallery once. The backend (rdd = RDD-LightGlue, loma = LoMa matcher)
# selects the conda env, the checkpoint and the feature cache. Submitted by
# pipeline_czechlynx.sh as
#   sbatch --array=0-$((N-1)) czechlynx_mine_worker.sh <view_root> <cache> <report_prefix> <split> <N> [rdd|loma]
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
view_root=${1:?view root}; cache=${2:?cache}; report=${3:?report prefix}; split=${4:?split}; workers=${5:?workers}; backend=${6:-rdd}
if [[ "${backend}" == loma ]]; then activate_conda_env "${CONDA_ENV_LOMA}"; weights=${LOMA_WEIGHTS}; else activate_conda_env "${CONDA_ENV_RDD}"; weights=${LG_WEIGHTS}; fi
python -m scripts.czechlynx_mine \
  --dataset_root "${view_root}" --cache_dir "${cache}" \
  --rdd_weights "${RDD_WEIGHTS}" --lg_weights "${weights}" \
  --backend "${backend}" --variant "${CZECHLYNX_LOMA_VARIANT:-loma-b}" \
  --split "${split}" --all_queries --num_workers "${workers}" --worker_index "${SLURM_ARRAY_TASK_ID}" \
  --frames_per_collection "${CZECHLYNX_FRAMES_PER_COLLECTION:-20}" \
  --top_k_frames "${CZECHLYNX_TOP_K_FRAMES:-5}" --top_m "${CZECHLYNX_TOP_M:-10}" \
  --dump_report "${report}"
