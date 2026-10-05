#!/usr/bin/env bash
#SBATCH --job-name=fs-cz-aggregate
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
# Combine the CzechLynx few-shot mining shards into the trainer indices (train + test; the
# legacy protocol validates on the test index, so val is not mined).
#   sbatch --dependency=afterok:<mining arrays> czechlynx_aggregate_job.sh <report_prefix> <view_root>
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
activate_conda_env "${CONDA_ENV_RDD}"
report=${1:?report prefix}; view_root=${2:?view root}
python -m scripts.czechlynx_aggregate --dump_report "${report}" --dataset_root "${view_root}" --splits train test
for split in train test; do
  [[ -s "${report}_${split}_combined.json" ]] || { echo "missing ${report}_${split}_combined.json" >&2; exit 1; }
done
