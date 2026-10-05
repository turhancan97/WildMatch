#!/usr/bin/env bash
# The complete few-shot experiment for CzechLynx (time-closed, legacy protocol), queued in
# one go — the CzechLynx counterpart of pipeline_dataset.sh. CzechLynx is ~30x the mining
# work of the small wildlife datasets (train/test queries x 7,200 gallery frames), so the
# steps are separate Slurm jobs chained with dependencies instead of one job per fraction:
#
#   1. few-shot views for every fraction (inline; scripts.czechlynx_fewshot), identical views
#      dropped;
#   2. per fraction: mining as two arrays of strided workers (train and test queries, each
#      worker loads the gallery once; rdd: rtx4090, loma: dgxh100) -> aggregation (CPU) ->
#      fine-tuning (rdd: dgxh100, 2 GPUs x batch 8 = the reference global batch 16; loma:
#      rtx4090, 4 GPUs x batch 8 = the reference global batch 32); every step is skipped
#      when its output already exists;
#   3. a CPU chain job waiting for the trainings submits the probe benchmark (reduced and
#      full gallery; explainable_individual_reidentification/probe-fewshot-wildlife.sh CzechLynx)
#      and a collector writing $EXREID_ROOT/reports/fewshot/CzechLynx/.
#
# The benchmark-repository evaluation (czechlynx_evaluate.sh, ~50M pairs per checkpoint) is
# not part of the chain; the probe is the reported metric.
#
# The backend (FEWSHOT_BACKEND=rdd|loma) selects the matcher that mines the pairs and is
# fine-tuned: the indices are mined with that matcher and live under
# $FEWSHOT_ROOT/indices/CzechLynx/<protocol>/<view>/<backend>/, the checkpoints under
# .../checkpoints/.../<view>/<backend>-finetuned/ — RDD and LoMa never share an index.
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/pipeline_czechlynx.sh [seed]                  # RDD-LightGlue
#   FEWSHOT_BACKEND=loma bash slurm_scripts/fewshot/pipeline_czechlynx.sh   # LoMa-B
# environment:
#   FEWSHOT_BACKEND (rdd), FEWSHOT_FRACTIONS ("0.125 0.25 0.5 1.0")
#   FEWSHOT_CZ_MINE_WORKERS (16)        workers per split (array size); mining throughput scales with it
#   FEWSHOT_CZ_MINE_SBATCH              resources of the mining arrays; default rdd: "-p rtx4090_batch
#                                       --qos=batch", loma: "-p dgxh100 --qos=quick --cpus-per-task=10 --mem=64G"
#   FEWSHOT_CZ_TRAIN_SBATCH             default rdd: "-p dgxh100 --qos=normal --gres=gpu:2 --cpus-per-task=20
#                                       --mem=128G --time=2-00:00:00"; loma: "-p rtx4090_batch --qos=batch --gres=gpu:4
#                                       --cpus-per-task=40 --mem=125G --time=12:00:00 --exclude=c11" (the reference
#                                       LoMa run: one rtx4090 node, 4 x batch 8, 300 epochs in ~2-3 h)
#   FEWSHOT_GLOBAL_BATCH (rdd 16, loma 32), FEWSHOT_EVAL_EPOCH (299), FEWSHOT_FORCE=1, FEWSHOT_SKIP_PROBE=1
#   PROBE_SBATCH_ARGS ("-p rtx4090_batch --qos=batch --exclude=c11,c15,c22"), PROBE_CANDIDATE_K ("50")
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
activate_conda_env "${CONDA_ENV_RDD}"
seed=${1:-${FEWSHOT_SEED:-0}}
dataset=CzechLynx
protocol=legacy
backend=${FEWSHOT_BACKEND:-rdd}
epoch=${FEWSHOT_EVAL_EPOCH:-299}
workers=${FEWSHOT_CZ_MINE_WORKERS:-16}
case "${backend}" in
  rdd)
    cache=${CZECHLYNX_RDD_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/rdd-cache}; weights=${LG_WEIGHTS}
    mine_sbatch=${FEWSHOT_CZ_MINE_SBATCH:--p rtx4090_batch --qos=batch}
    train_sbatch=${FEWSHOT_CZ_TRAIN_SBATCH:--p dgxh100 --qos=normal --gres=gpu:2 --cpus-per-task=20 --mem=128G --time=2-00:00:00} ;;
  loma)
    cache=${CZECHLYNX_LOMA_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-cache}; weights=${LOMA_WEIGHTS}
    mine_sbatch=${FEWSHOT_CZ_MINE_SBATCH:--p dgxh100 --qos=quick --cpus-per-task=10 --mem=64G}
    train_sbatch=${FEWSHOT_CZ_TRAIN_SBATCH:--p rtx4090_batch --qos=batch --gres=gpu:4 --cpus-per-task=40 --mem=125G --time=12:00:00 --exclude=c11} ;;
  *) echo "FEWSHOT_BACKEND must be rdd or loma (got ${backend})" >&2; exit 1 ;;
esac
[[ -f "${cache}/manifest.json" ]] || { echo "${backend} cache not found: ${cache}" >&2; exit 1; }
[[ -e "${weights}" ]] || { echo "${backend} checkpoint not found: ${weights}" >&2; exit 1; }
[[ -d "${EXREID_ROOT}" && -d "${LYNX_FINETUNING_ROOT}" ]] || { echo "EXREID_ROOT / LYNX_FINETUNING_ROOT must exist" >&2; exit 1; }
mkdir -p logs/fewshot
echo "== few-shot pipeline: ${dataset} (backend ${backend}, seed ${seed}, ${workers} mining workers per split)"

has_progress() {  # <fraction>: checkpoint present or a training job queued/running
  local view
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${1}, ${seed}))")
  [[ -f "${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-finetuned/epoch_${epoch}/model.safetensors" ]] && return 0
  [[ -n "$(squeue -h -u "${USER}" -n "fs-${dataset}-${1}-${backend}-train" -o '%i' 2>/dev/null)" ]]
}
queued() {  # <job name> -> ids of queued/running jobs with that name
  squeue -h -u "${USER}" -n "${1}" -o "%i" 2>/dev/null | tr '\n' ' '
}

# 1. views ---------------------------------------------------------------------------------
selected=(); previous_kept=""
for fraction in ${FEWSHOT_FRACTIONS:-0.125 0.25 0.5 1.0}; do
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
  view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
  python -m scripts.czechlynx_fewshot --fraction "${fraction}" --seed "${seed}" --protocol "${protocol}" \
    --output_root "${view_root}" > "logs/fewshot/prepare-${dataset}-${view}.log" 2>&1 \
    || { echo "view preparation failed for ${view}, see logs/fewshot/prepare-${dataset}-${view}.log" >&2; exit 1; }
  read -r kept effective feasible < <(python - "${view_root}/fewshot.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["fewshot"]
print(s["train_frames_kept"], f"{s['effective_fraction']:.3f}", s["budget_feasible"])
PY
)
  if [[ "${kept}" == "${previous_kept}" ]]; then
    if has_progress "${fraction}" && ! has_progress "${selected[-1]}"; then
      echo "fraction ${fraction}: identical to ${selected[-1]} but already trained/queued -> replaces it"; selected[-1]=${fraction}
    else
      echo "fraction ${fraction}: identical to the previous view (${kept} training frames) -> skipped"
    fi
    continue
  fi
  echo "fraction ${fraction}: ${view} keeps ${kept} training frames (effective ${effective}, budget feasible: ${feasible})"
  selected+=("${fraction}"); previous_kept=${kept}
done

# 2. mining -> aggregation -> training per fraction ----------------------------------------------
train_jobs=()
for fraction in "${selected[@]}"; do
  view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
  view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
  index_root=${FEWSHOT_ROOT}/indices/${dataset}/${protocol}/${view}/${backend}
  report=${index_root}/strong-matches
  output_dir=${FEWSHOT_ROOT}/checkpoints/${dataset}/${protocol}/${view}/${backend}-finetuned
  finetuned=${output_dir}/epoch_${epoch}/model.safetensors
  echo "-- fraction ${fraction} (${view}, ${backend})"
  if [[ -f "${finetuned}" && -z "${FEWSHOT_FORCE:-}" ]]; then echo "   checkpoint exists -> nothing to do"; continue; fi
  running=$(queued "fs-${dataset}-${fraction}-${backend}-train")
  if [[ -n "${running// /}" ]]; then echo "   training job already queued/running (${running}) -> reused as dependency"; train_jobs+=(${running}); continue; fi

  train_dependency=""
  if [[ -f "${report}_train_combined.json" && -f "${report}_test_combined.json" && -z "${FEWSHOT_FORCE:-}" ]]; then
    echo "   indices exist -> mining skipped"
  else
    aggregate_running=$(queued "fs-${dataset}-${fraction}-${backend}-aggregate")
    if [[ -n "${aggregate_running// /}" ]]; then
      echo "   mining/aggregation already queued (${aggregate_running}) -> reused"
      train_dependency="--dependency=afterok:${aggregate_running// /}"
    else
      mkdir -p "${index_root}"
      python -m scripts.wildlife_cache_check "${backend}" "${view_root}" "${cache}" "${weights}" "${CZECHLYNX_LOMA_VARIANT:-loma-b}" --fast
      mine_jobs=()
      for split in train test; do
        # shellcheck disable=SC2086
        job=$(sbatch --parsable ${mine_sbatch} \
          --job-name="fs-${dataset}-${fraction}-${backend}-mine-${split}" --array="0-$(( workers - 1 ))" \
          slurm_scripts/fewshot/czechlynx_mine_worker.sh "${view_root}" "${cache}" "${report}" "${split}" "${workers}" "${backend}")
        mine_jobs+=("${job}"); echo "   mining ${split} (${backend}): array ${job} (${workers} workers)"
      done
      aggregate=$(sbatch --parsable --dependency="afterok:$(IFS=:; echo "${mine_jobs[*]}")" \
        --job-name="fs-${dataset}-${fraction}-${backend}-aggregate" slurm_scripts/fewshot/czechlynx_aggregate_job.sh "${report}" "${view_root}")
      echo "   aggregation: ${aggregate}"
      train_dependency="--dependency=afterok:${aggregate}"
    fi
  fi
  # shellcheck disable=SC2086
  job=$(sbatch --parsable ${train_dependency} ${train_sbatch} \
    --job-name="fs-${dataset}-${fraction}-${backend}-train" \
    slurm_scripts/fewshot/czechlynx_train_job.sh "${view_root}" "${index_root}" "${output_dir}" "${dataset}-${backend}-${protocol}-${view}" "${backend}")
  train_jobs+=("${job}"); echo "   training: ${job}${train_dependency:+ (${train_dependency})}"
done

# 3. probes + collection ----------------------------------------------------------------------
if [[ -n "${FEWSHOT_SKIP_PROBE:-}" ]]; then echo "FEWSHOT_SKIP_PROBE set: no probe chain (training jobs: ${train_jobs[*]:-none})"; exit 0; fi
dependency=""
if (( ${#train_jobs[@]} > 0 )); then dependency="--dependency=afterany:$(IFS=:; echo "${train_jobs[*]}")"; fi
# shellcheck disable=SC2086
chain=$(cd "${EXREID_ROOT}" && mkdir -p logs/fewshot && \
  FEWSHOT_SEED=${seed} FEWSHOT_PROTOCOL=${protocol} FEWSHOT_EVAL_EPOCH=${epoch} \
  sbatch --parsable ${dependency} --job-name="fs-probe-${dataset}-${backend}" slurm/fewshot_probe_chain.sh "${dataset}" "${selected[@]}")
echo "probe chain job ${chain}${dependency:+ (${dependency})}: submits the probe arrays, then the collector"
echo "results: ${EXREID_ROOT}/reports/fewshot/${dataset}/"
record_dir=${FEWSHOT_ROOT}/runs/${dataset}/${protocol}
mkdir -p "${record_dir}"
cat > "${record_dir}/pipeline-seed${seed}-${backend}-$(date -u +%Y%m%dT%H%M%SZ).json" <<EOF
{"dataset": "${dataset}", "seed": ${seed}, "backend": "${backend}", "protocol": "${protocol}",
 "fractions": "${selected[*]}", "train_jobs": "${train_jobs[*]:-}", "probe_chain_job": "${chain}",
 "mining_workers": ${workers}, "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
