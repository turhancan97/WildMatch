#!/usr/bin/env bash
# End-to-end few-shot run for one WildlifeReID dataset and one training-data fraction:
#
#   1. few-shot view of the canonical dataset (symlinks + probe metadata column)  inline, seconds
#   2. positive/negative mining on that view (train + test query arrays, aggregate)  Slurm
#   3. fine-tuning on the mined index                                                Slurm, after 2
#   4. full-gallery evaluation of the pretrained and the fine-tuned checkpoint       Slurm, after 3
#
# Each fraction has its own gallery, therefore its own index, checkpoints and reports:
#
#   $FEWSHOT_ROOT/views/<dataset>/<protocol>/<view>/                the few-shot view
#   $FEWSHOT_ROOT/indices/<dataset>/<protocol>/<view>/<backend>/    strong-matches_*_combined.json
#   $FEWSHOT_ROOT/checkpoints/<dataset>/<protocol>/<view>/<backend>-finetuned/epoch_*/
#   $FEWSHOT_ROOT/eval/<dataset>/<protocol>/<view>/<backend>-{pretrained,epoch<E>}-full.json
#   $FEWSHOT_ROOT/runs/<dataset>/<protocol>/<view>-<backend>.json   job ids and paths of this run
#
# with <view> = frac<fraction>-seed<seed>. The RDD/LoMa feature caches of the full dataset
# are reused (canonical frame names are unchanged).
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/run_wildlife_fewshot.sh configs/wildlife/StripeSpotter.json 0.125 [seed]
# environment:
#   FEWSHOT_BACKEND=rdd|loma      (rdd)   which matcher is mined with and fine-tuned
#   WILDLIFE_PROTOCOL             (legacy)
#   FEWSHOT_FORCE=1               re-mine / re-train even when the index / checkpoint exist
#   FEWSHOT_SKIP_EVAL=1           do not submit step 4
#   FEWSHOT_EVAL_EPOCH            (299)   fine-tuned checkpoint epoch evaluated in step 4
#   WILDLIFE_MAX_CONCURRENT       (30)    mining array throttle
#   WILDLIFE_EPOCHS, WILDLIFE_BATCH_SIZE, WILDLIFE_NUM_PROCESSES  forwarded to the trainer
#   FEWSHOT_DRY_RUN=1             build the view and validate the cache, print the sbatch calls
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
if [[ -n "${FEWSHOT_DRY_RUN:-}" ]]; then
  # Replace sbatch (also inside the spawn script, a child bash) with a stub returning job id 0.
  sbatch() { echo "[dry-run] sbatch $*" >&2; echo 0; }
  export -f sbatch
fi

backend=${FEWSHOT_BACKEND:-rdd}
if [[ "${backend}" == loma ]]; then activate_conda_env "${CONDA_ENV_LOMA}"; else activate_conda_env "${CONDA_ENV_RDD}"; fi
source slurm_scripts/fewshot/fewshot_paths.sh
fewshot_resolve "${1:?usage: $0 configs/wildlife/<dataset>.json <fraction> [seed]}" \
  "${2:?usage: $0 configs/wildlife/<dataset>.json <fraction> [seed]}" "${3:-${FEWSHOT_SEED:-0}}"

echo "== few-shot ${dataset} ${protocol} ${view} backend=${backend}"
echo "view       ${view_root}"
echo "indices    ${index_root}/${backend}"
echo "checkpoint ${output_dir}"
echo "cache      ${cache}"

# 1. view ------------------------------------------------------------------------------------
FEWSHOT_FRACTION=${fraction} FEWSHOT_SEED=${seed} FEWSHOT_VIEW_ROOT=${view_root} \
  bash slurm_scripts/fewshot/prepare_wildlife_fewshot.sh

# 2. mining ----------------------------------------------------------------------------------
mkdir -p logs/wildlife-aggregate logs/wildlife-eval logs/wildlife-mine
train_index=${index_root}/${backend}/strong-matches_train_combined.json
mining_dependency=""
aggregate_job=""
if [[ -f "${train_index}" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "index exists, mining skipped: ${train_index} (FEWSHOT_FORCE=1 to redo)"
else
  spawn_output=$(WILDLIFE_VIEW_ROOT=${view_root} WILDLIFE_INDEX_ROOT=${index_root} \
    WILDLIFE_MINING_BACKEND=${backend} WILDLIFE_RDD_CACHE=${cache} WILDLIFE_LOMA_CACHE=${cache} \
    bash slurm_scripts/spawn_wildlife_mining.sh)
  echo "${spawn_output}"
  aggregate_job=$(sed -n 's/^aggregation: \([0-9]\+\).*/\1/p' <<< "${spawn_output}")
  [[ -n "${aggregate_job}" ]] || { echo "could not find the aggregation job id in spawn output" >&2; exit 1; }
  mining_dependency="afterok:${aggregate_job}"
fi

# 3. fine-tuning -----------------------------------------------------------------------------
train_job=""
if [[ -f "${finetuned}" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "checkpoint exists, training skipped: ${finetuned} (FEWSHOT_FORCE=1 to redo)"
else
  train_job=$(fewshot_export_train_env; cd "${LYNX_FINETUNING_ROOT}" && mkdir -p "${train_log_dir}" && \
    sbatch --parsable ${mining_dependency:+--dependency=${mining_dependency}} "${train_script}")
  echo "training: ${train_job}${mining_dependency:+ (after ${aggregate_job})}"
fi

# 4. evaluation ------------------------------------------------------------------------------
eval_pretrained_job=""
eval_finetuned_job=""
if [[ -z "${FEWSHOT_SKIP_EVAL:-}" ]]; then
  mkdir -p "${eval_root}"
  eval_pretrained_job=$(fewshot_export_eval_env "${pretrained}" "${eval_root}/${backend}-pretrained-full.json"
    sbatch --parsable slurm_scripts/wildlife_evaluate.sh)
  echo "evaluation (pretrained): ${eval_pretrained_job}"
  eval_finetuned_job=$(fewshot_export_eval_env "${finetuned}" "${eval_root}/${backend}-epoch${eval_epoch}-full.json"
    sbatch --parsable ${train_job:+--dependency=afterok:${train_job}} slurm_scripts/wildlife_evaluate.sh)
  echo "evaluation (epoch ${eval_epoch}): ${eval_finetuned_job}${train_job:+ (after ${train_job})}"
fi

fewshot_write_record run_wildlife_fewshot.sh "\"jobs\": {\"aggregate\": \"${aggregate_job}\", \"train\": \"${train_job}\", \"eval_pretrained\": \"${eval_pretrained_job}\", \"eval_finetuned\": \"${eval_finetuned_job}\"}"
