#!/usr/bin/env bash
# The complete few-shot experiment for one WildlifeReID dataset, queued in one go:
#
#   1. few-shot views for every fraction (inline, seconds); fractions whose view is identical
#      to the previous one (minimum-of-two-images rule) are dropped;
#   2. one self-contained fine-tuning job per remaining fraction (fewshot_job.sh: mining ->
#      training -> evaluation), skipped when its checkpoint already exists;
#   3. a CPU "chain" job that waits for those jobs (--dependency) and submits the probe
#      benchmark for the reduced-gallery and the full-gallery setting
#      (explainable_individual_reidentification/probe-fewshot-wildlife.sh);
#   4. a CPU collector job that waits for the probe arrays and writes
#      $EXREID_ROOT/reports/fewshot/<dataset>/ (CSV, markdown summary, figures).
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/pipeline_dataset.sh configs/wildlife/NyalaData.json [seed]
# environment:
#   FEWSHOT_FRACTIONS ("0.125 0.25 0.5 1.0"), FEWSHOT_BACKEND (rdd), WILDLIFE_PROTOCOL (legacy)
#   FEWSHOT_PARTITION (dgxh100) FEWSHOT_QOS (quick) FEWSHOT_GPUS (1) FEWSHOT_CPUS (10)
#   FEWSHOT_MEM (64G) FEWSHOT_TIME (24:00:00)            resources of the fine-tuning jobs
#   FEWSHOT_FORCE=1                                       re-train existing checkpoints
#   FEWSHOT_SKIP_PROBE=1                                  stop after the fine-tuning jobs
#   PROBE_SBATCH_ARGS ("-p rtx4090_batch --qos=batch"), PROBE_CANDIDATE_K ("50")   probe settings
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
activate_conda_env "${CONDA_ENV_RDD}"
config_arg=${1:?usage: $0 configs/wildlife/<dataset>.json [seed]}
seed=${2:-${FEWSHOT_SEED:-0}}
config=$(readlink -f "${config_arg}")
[[ -f "${config}" ]] || { echo "config not found: ${config_arg}" >&2; exit 1; }
[[ -d "${EXREID_ROOT}" ]] || { echo "EXREID_ROOT does not exist: ${EXREID_ROOT}" >&2; exit 1; }
backend=${FEWSHOT_BACKEND:-rdd}
protocol=${WILDLIFE_PROTOCOL:-legacy}
epoch=${FEWSHOT_EVAL_EPOCH:-299}
eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
dataset=${WILDLIFE_DATASET_ID}
mkdir -p logs/fewshot
echo "== few-shot pipeline: ${dataset} (seed ${seed}, backend ${backend}, protocol ${protocol})"

has_progress() {  # <fraction>: checkpoint present or a fine-tuning job queued/running for it
  local view finetuned
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${1}, ${seed}))")
  finetuned=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-finetuned/epoch_${epoch}/model.safetensors
  [[ -f "${finetuned}" ]] && return 0
  [[ -n "$(squeue -h -u "${USER}" -n "fs-${dataset}-${1}-${backend}" -o '%i' 2>/dev/null)" ]]
}

# 1. views ---------------------------------------------------------------------------------
selected=()
previous_kept=""
for fraction in ${FEWSHOT_FRACTIONS:-0.125 0.25 0.5 1.0}; do
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
  view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
  WILDLIFE_CONFIG=${config} WILDLIFE_PROTOCOL=${protocol} FEWSHOT_FRACTION=${fraction} FEWSHOT_SEED=${seed} \
    FEWSHOT_VIEW_ROOT=${view_root} bash slurm_scripts/fewshot/prepare_wildlife_fewshot.sh > "logs/fewshot/prepare-${dataset}-${view}.log" 2>&1 \
    || { echo "view preparation failed for ${view}, see logs/fewshot/prepare-${dataset}-${view}.log" >&2; exit 1; }
  read -r kept effective feasible < <(python - "${view_root}/fewshot.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["fewshot"]
print(s["train_frames_kept"], f"{s['effective_fraction']:.3f}", s["budget_feasible"])
PY
)
  if [[ "${kept}" == "${previous_kept}" ]]; then
    # Identical views (minimum-per-identity rule): keep the one that already has a
    # checkpoint or a running job, otherwise the first one.
    if has_progress "${fraction}" && ! has_progress "${selected[-1]}"; then
      echo "fraction ${fraction}: identical to ${selected[-1]} but already trained/queued -> replaces it"
      selected[-1]=${fraction}
    else
      echo "fraction ${fraction}: identical to the previous view (${kept} training images, minimum-per-identity rule) -> skipped"
    fi
    continue
  fi
  echo "fraction ${fraction}: ${view} keeps ${kept} training images (effective ${effective}, budget feasible: ${feasible})"
  selected+=("${fraction}")
  previous_kept=${kept}
done
(( ${#selected[@]} > 0 )) || { echo "no fractions left to run" >&2; exit 1; }

# 2. fine-tuning jobs ------------------------------------------------------------------------
train_jobs=()
for fraction in "${selected[@]}"; do
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
  finetuned=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-finetuned/epoch_${epoch}/model.safetensors
  if [[ -f "${finetuned}" && -z "${FEWSHOT_FORCE:-}" ]]; then
    echo "fraction ${fraction}: checkpoint exists (${finetuned}) -> no training job"
    continue
  fi
  job_name="fs-${dataset}-${fraction}-${backend}"
  running=$(squeue -h -u "${USER}" -n "${job_name}" -o "%i" 2>/dev/null | tr '\n' ' ')
  if [[ -n "${running// /}" ]]; then
    echo "fraction ${fraction}: job ${job_name} already queued/running (${running}) -> reused as dependency"
    train_jobs+=(${running})
    continue
  fi
  job=$(sbatch --parsable \
    --job-name="${job_name}" \
    --partition="${FEWSHOT_PARTITION:-dgxh100}" --qos="${FEWSHOT_QOS:-quick}" \
    --gres="gpu:${FEWSHOT_GPUS:-1}" --cpus-per-task="${FEWSHOT_CPUS:-10}" \
    --mem="${FEWSHOT_MEM:-64G}" --time="${FEWSHOT_TIME:-24:00:00}" \
    slurm_scripts/fewshot/fewshot_job.sh "${config}" "${fraction}" "${seed}")
  train_jobs+=("${job}")
  echo "fraction ${fraction}: fine-tuning job ${job} (logs/fewshot/fs-${dataset}-${fraction}-${backend}-${job}.out)"
done

# 3./4. probes + collection, after the fine-tuning jobs -----------------------------------------
if [[ -n "${FEWSHOT_SKIP_PROBE:-}" ]]; then
  echo "FEWSHOT_SKIP_PROBE set: fine-tuning jobs only (${train_jobs[*]:-none})"
  exit 0
fi
dependency=""
if (( ${#train_jobs[@]} > 0 )); then dependency="--dependency=afterany:$(IFS=:; echo "${train_jobs[*]}")"; fi
# shellcheck disable=SC2086
chain=$(cd "${EXREID_ROOT}" && mkdir -p logs/fewshot && \
  FEWSHOT_SEED=${seed} FEWSHOT_PROTOCOL=${protocol} FEWSHOT_EVAL_EPOCH=${epoch} \
  sbatch --parsable ${dependency} --job-name="fs-probe-${dataset}" slurm/fewshot_probe_chain.sh "${dataset}" "${selected[@]}")
echo "probe chain job ${chain}${dependency:+ (${dependency})}: submits the probe arrays, then the collector"
echo "results: ${EXREID_ROOT}/reports/fewshot/${dataset}/ (fewshot_summary.md, fewshot_results.csv, fewshot_*.png)"

record_dir=${FEWSHOT_ROOT}/runs/${dataset}/${protocol}
mkdir -p "${record_dir}"
cat > "${record_dir}/pipeline-seed${seed}-${backend}-$(date -u +%Y%m%dT%H%M%SZ).json" <<EOF
{"dataset": "${dataset}", "seed": ${seed}, "backend": "${backend}", "protocol": "${protocol}",
 "fractions": "${selected[*]}", "train_jobs": "${train_jobs[*]:-}", "probe_chain_job": "${chain}",
 "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
