#!/usr/bin/env bash
#SBATCH --job-name=fs-bench-staging
#SBATCH --partition=dgxh100
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
# Timing experiment: the same short fine-tuning (WILDLIFE_EPOCHS epochs, no checkpoint of
# value) once with the feature cache on NFS and once with it staged on the node-local disk.
#   sbatch slurm_scripts/fewshot/bench_cache_staging.sh configs/wildlife/NyalaData.json 1.0
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
activate_conda_env "${CONDA_ENV_RDD}"
source slurm_scripts/fewshot/fewshot_paths.sh
fewshot_resolve "${1:?config}" "${2:?fraction}" "${3:-0}"
epochs=${WILDLIFE_EPOCHS:-5}
scratch=${FEWSHOT_ROOT}/scratch/bench-${dataset}-${view}-${SLURM_JOB_ID:-local}
staged=${TMPDIR:-/tmp}/fewshot-bench-${SLURM_JOB_ID:-$$}
trap 'rm -rf "${staged}"' EXIT
mkdir -p "${scratch}"
t0=$(date +%s); cp -r "${cache}/." "${staged}/"; echo "staging: $(( $(date +%s) - t0 ))s for $(du -sm "${cache}" | cut -f1) MB"
for variant in nfs local; do
  if [[ "${variant}" == local ]]; then use_cache=${staged}; else use_cache=${cache}; fi
  echo "[$(date '+%F %T')] ==== ${variant} cache: ${use_cache}"
  t0=$(date +%s)
  (fewshot_export_train_env
   export WILDLIFE_RDD_CACHE=${use_cache} WILDLIFE_RDD_OUTPUT=${scratch}/${variant} WILDLIFE_EPOCHS=${epochs}
   export WILDLIFE_NUM_PROCESSES=1 WILDLIFE_BATCH_SIZE=32 WILDLIFE_NUM_WORKERS=8 WANDB_MODE=offline
   export WILDLIFE_RDD_RUN_NAME=bench-${variant}-${dataset}-${view}
   cd "${LYNX_FINETUNING_ROOT}" && bash slurm_scripts/train_wildlife_rdd.sh) > "${scratch}/${variant}.log" 2>&1 || echo "${variant}: training exited with an error, see ${scratch}/${variant}.log"
  echo "[$(date '+%F %T')] ==== ${variant}: ${epochs} epochs in $(( $(date +%s) - t0 ))s"
  tr '\r' '\n' < "${scratch}/${variant}.log" | grep -E "^Epoch [0-9]+: 100%" | tail -3 | cut -c1-100
done
rm -rf "${scratch}/nfs/epoch_"* "${scratch}/local/epoch_"* 2>/dev/null || true
