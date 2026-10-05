#!/usr/bin/env bash
# Queue the fine-tuning variants on the CzechLynx fraction-1.0 view — 2 backends x 3 training
# modes — each through pipeline_czechlynx_components.sh (training on rtx4090_batch + probes
# + collector):
#   rdd   lg                   LightGlue (matcher), RDD frozen           --trained_model lg
#   rdd   descriptor           RDD descriptor network, LightGlue frozen  --trained_model rdd    --rdd_train_component descriptor
#   rdd   lg+descriptor        RDD descriptor network + LightGlue        --trained_model lg+rdd --rdd_train_component descriptor
#   loma  matcher              LoMa matcher, DeDoDe frozen               --loma_train_component matcher
#   loma  descriptor           DeDoDe descriptor, LoMa matcher frozen    --loma_train_component descriptor
#   loma  descriptor+matcher   DeDoDe descriptor + LoMa matcher          --loma_train_component descriptor+matcher
# FEWSHOT_EPOCHS (300) sets the training length; a different length (e.g. 30) is tagged ep<N>
# everywhere (checkpoint directories, job names, probe labels), so it lives next to the
# 300-epoch runs instead of overwriting them. With 300 epochs the matcher combinations are
# skipped when pipeline_czechlynx.sh already produced or queued them.
# The RDD runs use the RDD-mined indices; the LoMa runs need the LoMa-mined ones, i.e.
# `FEWSHOT_BACKEND=loma bash slurm_scripts/fewshot/pipeline_czechlynx.sh` must have been run —
# while its mining is queued the LoMa trainings wait for the aggregation job. Combinations
# whose indices are neither present nor being mined are skipped with a message.
#
# usage (from the rdd-parallel-benchmark root):
#   FEWSHOT_EPOCHS=30 bash slurm_scripts/fewshot/spawn_czechlynx_components.sh        # all six, 30 epochs
#   bash slurm_scripts/fewshot/spawn_czechlynx_components.sh rdd                       # only the RDD trio
#   FEWSHOT_RDD_COMPONENTS="descriptor" FEWSHOT_LOMA_COMPONENTS="" bash slurm_scripts/fewshot/spawn_czechlynx_components.sh
#   FEWSHOT_DRY_RUN=1 FEWSHOT_EPOCHS=30 bash slurm_scripts/fewshot/spawn_czechlynx_components.sh
# environment: FEWSHOT_RDD_COMPONENTS ("lg descriptor lg+descriptor"), FEWSHOT_LOMA_COMPONENTS
#   ("matcher descriptor descriptor+matcher"), FEWSHOT_FRACTION (1.0), FEWSHOT_SEED (0) and
#   everything pipeline_czechlynx_components.sh reads (FEWSHOT_EPOCHS, FEWSHOT_CZ_COMPONENT_SBATCH, ...)
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
only=${1:-}
fraction=${FEWSHOT_FRACTION:-1.0}
seed=${FEWSHOT_SEED:-0}
tag=${FEWSHOT_TAG-$([[ "${FEWSHOT_EPOCHS:-300}" != 300 ]] && echo "ep${FEWSHOT_EPOCHS}")}
combos=()
for component in ${FEWSHOT_RDD_COMPONENTS-lg descriptor lg+descriptor}; do combos+=("rdd ${component}"); done
for component in ${FEWSHOT_LOMA_COMPONENTS-matcher descriptor descriptor+matcher}; do combos+=("loma ${component}"); done
failed=(); names=()
for combo in "${combos[@]}"; do
  read -r backend component <<< "${combo}"
  [[ -z "${only}" || "${only}" == "${backend}" ]] || continue
  echo
  if bash slurm_scripts/fewshot/pipeline_czechlynx_components.sh "${backend}" "${component}" "${fraction}" "${seed}"; then
    names+=("fs-CzechLynx-${fraction}-${backend}-${component//+/-}-train${tag:+-${tag}}")
  else
    failed+=("${backend}/${component}")
  fi
done
echo
if (( ${#failed[@]} > 0 )); then echo "not submitted: ${failed[*]}"; fi
(( ${#names[@]} > 0 )) && echo "queue: squeue -u ${USER} -n $(IFS=,; echo "${names[*]}")"
(( ${#failed[@]} == 0 ))
