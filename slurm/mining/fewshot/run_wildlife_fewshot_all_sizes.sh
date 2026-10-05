#!/usr/bin/env bash
# Submit the few-shot pipeline for every training-data fraction of one dataset.
# Each fraction is an independent chain (view -> mining -> fine-tuning -> evaluation); the
# chains run concurrently on the cluster. See run_wildlife_fewshot.sh for the per-run options.
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/run_wildlife_fewshot_all_sizes.sh configs/wildlife/StripeSpotter.json [seed]
#   FEWSHOT_FRACTIONS="0.125 0.25" bash slurm_scripts/fewshot/run_wildlife_fewshot_all_sizes.sh configs/wildlife/ATRW.json
set -euo pipefail
config=${1:?usage: $0 configs/wildlife/<dataset>.json [seed]}
seed=${2:-${FEWSHOT_SEED:-0}}
fractions=${FEWSHOT_FRACTIONS:-0.125 0.25 0.5 1.0}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for fraction in ${fractions}; do
  echo
  echo "################ fraction ${fraction} seed ${seed}"
  bash "${script_dir}/run_wildlife_fewshot.sh" "${config}" "${fraction}" "${seed}"
done
