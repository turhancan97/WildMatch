#!/usr/bin/env bash
# Build one few-shot view of a WildlifeReID dataset (symlinks + probe metadata column).
# Fast (seconds, no GPU): run it directly from the repository root, or let
# run_wildlife_fewshot.sh call it.
#
#   FEWSHOT_FRACTION=0.125 WILDLIFE_CONFIG=configs/wildlife/StripeSpotter.json \
#   bash slurm_scripts/fewshot/prepare_wildlife_fewshot.sh
#
# Inputs (environment): WILDLIFE_CONFIG, WILDLIFE_PROTOCOL (legacy), FEWSHOT_FRACTION,
# FEWSHOT_SEED (0), FEWSHOT_MIN_PER_IDENTITY (2), FEWSHOT_VIEW_ROOT (derived),
# FEWSHOT_METADATA_OUT (derived; "none" disables the probe metadata), FEWSHOT_FORCE,
# FEWSHOT_PREPARE_DRY_RUN (print the selection only).
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
activate_conda_env "${CONDA_ENV_RDD}"
cd "${RDD_BENCHMARK_ROOT}"

config=${WILDLIFE_CONFIG:?set WILDLIFE_CONFIG to configs/wildlife/<dataset>.json}
protocol=${WILDLIFE_PROTOCOL:-legacy}
fraction=${FEWSHOT_FRACTION:?set FEWSHOT_FRACTION (e.g. 0.125, 0.25, 0.5, 1.0)}
seed=${FEWSHOT_SEED:-0}
eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
source_view=${WILDLIFE_VIEW_ROOT:-${WILDLIFE_PROCESSED_ROOT}/${WILDLIFE_DATASET_ID}/${protocol}}
output_root=${FEWSHOT_VIEW_ROOT:-${FEWSHOT_ROOT}/views/${WILDLIFE_DATASET_ID}/${protocol}/${view}}
metadata_out=${FEWSHOT_METADATA_OUT:-${WILDLIFE_SOURCE_ROOT}/metadata_fewshot/metadata_${WILDLIFE_DATASET_ID}.csv}

echo "dataset=${WILDLIFE_DATASET_ID} protocol=${protocol} view=${view}"
echo "source view=${source_view}"
echo "few-shot view=${output_root}"
echo "probe metadata=${metadata_out}"
python -m scripts.wildlife_fewshot \
  --config "${config}" --protocol "${protocol}" \
  --fraction "${fraction}" --seed "${seed}" \
  --min_per_identity "${FEWSHOT_MIN_PER_IDENTITY:-2}" \
  --source_view "${source_view}" --output_root "${output_root}" \
  --metadata_out "${metadata_out}" \
  ${FEWSHOT_FORCE:+--force} ${FEWSHOT_PREPARE_DRY_RUN:+--dry_run}
