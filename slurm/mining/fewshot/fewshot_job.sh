#!/usr/bin/env bash
#SBATCH --job-name=fewshot
#SBATCH --partition=dgxh100
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=c11
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
#
# The whole few-shot chain for one dataset and one training-data fraction inside a single
# allocation (no job arrays, no cross-job dependencies): view -> cache check -> mining
# (all query collections of a split in one process per GPU) -> aggregation -> fine-tuning
# -> full-gallery evaluation (pretrained and fine-tuned). Meant for partitions with tight
# per-user job/resource limits such as dgxh100; submit one job per fraction with
# submit_fewshot_dgxh100.sh, or directly:
#
#   sbatch slurm_scripts/fewshot/fewshot_job.sh configs/wildlife/CowDataset.json 0.125 [seed]
#
# Outputs go to the same $FEWSHOT_ROOT layout as run_wildlife_fewshot.sh. Options (env):
#   FEWSHOT_BACKEND=rdd|loma (rdd), WILDLIFE_PROTOCOL (legacy), FEWSHOT_FORCE=1,
#   FEWSHOT_SKIP_EVAL=1, FEWSHOT_EVAL_EPOCH (299), FEWSHOT_GLOBAL_BATCH (32: the trainer's
#   global batch, split over the job's GPUs so results stay comparable with the 4x8 runs),
#   WILDLIFE_EPOCHS (300), WILDLIFE_FRAMES_PER_COLLECTION/TOP_K_FRAMES/TOP_M (20/5/10),
#   FEWSHOT_EVAL_EVERY (50: pseudo-accuracy evaluation interval in epochs during training; the
#   reference runs used 10, which costs ~25% of the training time and only changes the W&B curve),
#   FEWSHOT_STOP_AFTER=view|mining|aggregate|training (stop early, e.g. to test the setup),
#   FEWSHOT_STAGE_CACHE=1 (copy the feature cache to node-local $TMPDIR before use: the
#   trainer reads ~100 cache files per step and is I/O bound on NFS; 0 disables).
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
backend=${FEWSHOT_BACKEND:-rdd}
if [[ "${backend}" == loma ]]; then activate_conda_env "${CONDA_ENV_LOMA}"; else activate_conda_env "${CONDA_ENV_RDD}"; fi
source slurm_scripts/fewshot/fewshot_paths.sh
fewshot_resolve "${1:?usage: fewshot_job.sh <config> <fraction> [seed]}" "${2:?fraction}" "${3:-${FEWSHOT_SEED:-0}}"

IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
n_gpus=${#gpus[@]}
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
global_batch=${FEWSHOT_GLOBAL_BATCH:-32}
(( global_batch % n_gpus == 0 )) || { echo "FEWSHOT_GLOBAL_BATCH=${global_batch} is not divisible by ${n_gpus} GPUs" >&2; exit 1; }
per_process_batch=$(( global_batch / n_gpus ))
loader_workers=$(( (cpus - n_gpus) / n_gpus )); (( loader_workers > 8 )) && loader_workers=8; (( loader_workers < 1 )) && loader_workers=1
variant=${WILDLIFE_LOMA_VARIANT:-loma-b}
report=${index_root}/${backend}/strong-matches
shared_cache=${cache}
if [[ "${FEWSHOT_STAGE_CACHE:-1}" != 0 ]]; then
  stage_parent=${FEWSHOT_STAGE_DIR:-${TMPDIR:-/tmp}}
  stage_dir=${stage_parent}/fewshot-cache-${SLURM_JOB_ID:-$$}
  need_mb=$(du -sm "${shared_cache}" | cut -f1)
  avail_mb=$(df -Pm "${stage_parent}" | awk 'NR==2 {print $4}')
  if (( avail_mb > need_mb * 12 / 10 + 2048 )); then
    echo "staging ${need_mb} MB of ${backend} cache to ${stage_dir} (node-local)"
    mkdir -p "${stage_dir}"
    stage_started=$(date +%s)
    cp -r "${shared_cache}/." "${stage_dir}/"
    cache=${stage_dir}
    echo "cache staged in $(( $(date +%s) - stage_started ))s"
  else
    echo "not enough local space to stage the cache (${avail_mb} MB free, ${need_mb} MB needed); using ${shared_cache}"
    unset stage_dir
  fi
fi
job_log_dir=logs/fewshot/${dataset}-${protocol}-${view}-${backend}
mkdir -p "${job_log_dir}" "${eval_root}"
# Private copies of the wrappers this job runs for hours: bash reads scripts lazily, and on
# NFS neither an in-place edit nor an atomic replacement of the repository file is safe for
# a script that another node is executing (stale file handle). Slurm copies this batch
# script itself; the wrappers it calls are copied here.
script_dir=${stage_parent:-${TMPDIR:-/tmp}}/fewshot-scripts-${SLURM_JOB_ID:-$$}
mkdir -p "${script_dir}"
cp "${LYNX_FINETUNING_ROOT}/${train_script}" "${script_dir}/train.sh"
cp slurm_scripts/wildlife_evaluate.sh "${script_dir}/evaluate.sh"
trap 'rm -rf "${script_dir}" ${stage_dir:+"${stage_dir}"}' EXIT

declare -A timings
step_started=0
step_begin() { step_started=$(date +%s); echo; echo "[$(date '+%F %T')] ==== ${1}"; }
step_end() {
  local took=$(( $(date +%s) - step_started ))
  timings[$1]=${took}
  echo "[$(date '+%F %T')] ==== ${1}: done in ${took}s"
  if [[ "${FEWSHOT_STOP_AFTER:-}" == "$1" ]]; then echo "FEWSHOT_STOP_AFTER=${1}: stopping here"; exit 0; fi
}

echo "== few-shot ${dataset} ${protocol} ${view} backend=${backend}  job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "gpus=${n_gpus} (${CUDA_VISIBLE_DEVICES:-0}) cpus=${cpus} global_batch=${global_batch} per_process_batch=${per_process_batch} loader_workers=${loader_workers}"
echo "view       ${view_root}"
echo "indices    ${index_root}/${backend}"
echo "checkpoint ${output_dir}"
echo "cache      ${shared_cache}${stage_dir:+ (staged at ${cache})}"
nvidia-smi -L || true

# 1. view ------------------------------------------------------------------------------------
step_begin view
FEWSHOT_FRACTION=${fraction} FEWSHOT_SEED=${seed} FEWSHOT_VIEW_ROOT=${view_root} \
  bash slurm_scripts/fewshot/prepare_wildlife_fewshot.sh
python -m scripts.wildlife_cache_check "${backend}" "${view_root}" "${cache}" "${pretrained}" "${variant}"
step_end view

# 2. mining ----------------------------------------------------------------------------------
train_index=${report}_train_combined.json
if [[ -f "${train_index}" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "index exists, mining skipped: ${train_index} (FEWSHOT_FORCE=1 to redo)"
else
  step_begin mining
  mkdir -p "$(dirname "${report}")"
  for split in train test; do
    pids=()
    for (( w = 0; w < n_gpus; w++ )); do
      CUDA_VISIBLE_DEVICES=${gpus[w]} python -m scripts.wildlife_mine \
        --dataset_id "${dataset}" --dataset_root "${view_root}" --cache_dir "${cache}" \
        --lg_weights "${pretrained}" --backend "${backend}" --variant "${variant}" \
        --split "${split}" --all_queries --num_workers "${n_gpus}" --worker_index "${w}" \
        --frames_per_collection "${WILDLIFE_FRAMES_PER_COLLECTION:-20}" \
        --top_k_frames "${WILDLIFE_TOP_K_FRAMES:-5}" --top_m "${WILDLIFE_TOP_M:-10}" \
        --dump_report "${report}" > "${job_log_dir}/mine-${split}-worker${w}.log" 2>&1 &
      pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    if (( failed )); then echo "mining failed for split ${split}; see ${job_log_dir}/mine-${split}-worker*.log" >&2; tail -n 30 "${job_log_dir}"/mine-"${split}"-worker*.log >&2; exit 1; fi
    echo "${split}: $(ls "${report}"_"${split}"_*.json 2>/dev/null | grep -v -e combined -e metadata | wc -l) query reports"
  done
  step_end mining

  step_begin aggregate
  python -m scripts.wildlife_aggregate \
    --dataset_id "${dataset}" --protocol "${protocol}" --dump_report "${report}" \
    --dataset_root "${view_root}" --backend "${backend}" --variant "${variant}" \
    --weights "${pretrained}" --cache_dir "${cache}" \
    --frames_per_collection "${WILDLIFE_FRAMES_PER_COLLECTION:-20}" \
    --top_k_frames "${WILDLIFE_TOP_K_FRAMES:-5}" --top_m "${WILDLIFE_TOP_M:-10}"
  step_end aggregate
fi

# 3. fine-tuning -----------------------------------------------------------------------------
if [[ -f "${finetuned}" && -z "${FEWSHOT_FORCE:-}" ]]; then
  echo "checkpoint exists, training skipped: ${finetuned} (FEWSHOT_FORCE=1 to redo)"
else
  step_begin training
  (fewshot_export_train_env
   export WILDLIFE_NUM_PROCESSES=${n_gpus} WILDLIFE_BATCH_SIZE=${per_process_batch} WILDLIFE_NUM_WORKERS=${loader_workers}
   export WILDLIFE_EVAL_EVERY=${FEWSHOT_EVAL_EVERY:-50}
   cd "${LYNX_FINETUNING_ROOT}" && mkdir -p "${train_log_dir}" && bash "${script_dir}/train.sh")
  [[ -f "${finetuned}" ]] || { echo "training finished but ${finetuned} is missing" >&2; exit 1; }
  step_end training
fi

# 4. evaluation ------------------------------------------------------------------------------
if [[ -z "${FEWSHOT_SKIP_EVAL:-}" ]]; then
  step_begin eval_pretrained
  (fewshot_export_eval_env "${pretrained}" "${eval_root}/${backend}-pretrained-full.json"; bash "${script_dir}/evaluate.sh")
  step_end eval_pretrained
  step_begin eval_finetuned
  (fewshot_export_eval_env "${finetuned}" "${eval_root}/${backend}-epoch${eval_epoch}-full.json"; bash "${script_dir}/evaluate.sh")
  step_end eval_finetuned
fi

fragment="\"jobs\": {\"single_job\": \"${SLURM_JOB_ID:-local}\"}, \"timings_s\": {"
sep=""
for key in "${!timings[@]}"; do fragment+="${sep}\"${key}\": ${timings[$key]}"; sep=", "; done
fragment+="}"
cache=${shared_cache}   # the record must point at the shared cache, not the staged copy
fewshot_write_record fewshot_job.sh "${fragment}"
echo; echo "== done: ${dataset} ${view} ${backend}"
for key in view mining aggregate training eval_pretrained eval_finetuned; do
  if [[ -n "${timings[$key]:-}" ]]; then printf '  %-16s %6ss\n' "${key}" "${timings[$key]}"; fi
done
exit 0
