#!/usr/bin/env bash
#SBATCH --job-name=fs-cz-train
#SBATCH --partition=dgxh100
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=c11
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
#
# Fine-tune the matcher on one CzechLynx few-shot view: backend rdd = RDD-LightGlue
# (train_czechlynx_rdd.sh, reference run 2 processes x batch 8 = global 16), backend loma =
# LoMa-B matcher (train_czechlynx_loma.sh, reference run 4 x 8 = global 32). The wrapper from
# the lynx-finetuning repository runs as a private copy on a node-local copy of the backend's
# feature cache; FEWSHOT_GLOBAL_BATCH is split over the job's GPUs so the effective batch
# stays the reference one.
#   sbatch [--dependency=afterok:<aggregate>] czechlynx_train_job.sh <view_root> <index_root> <output_dir> <run_name> [rdd|loma] [component]
# component (6th argument) = the wrapper's training component: rdd: lg (default), descriptor,
# lg+descriptor; loma: matcher (default), descriptor, descriptor+matcher. The descriptor
# components recompute features every step (no feature cache; LoMa keeps only its DaD keypoint
# cache) and take days on rtx4090, so the job resumes: when <output_dir> already holds
# epoch checkpoints but not the final one, training continues from the newest complete epoch
# (CZECHLYNX_RESUME_FROM); when the final checkpoint exists the job exits at once. Submit
# several copies chained with --dependency=afterany to cover the wall-time limit.
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
view_root=${1:?view root}; index_root=${2:?index root}; output_dir=${3:?output dir}; run_name=${4:?run name}; backend=${5:-rdd}; component=${6:-}
# FEWSHOT_EPOCHS (300) is the training length (the cosine LR schedule spans it); the final
# checkpoint is epoch FEWSHOT_EVAL_EPOCH (default: the last one). FEWSHOT_EVAL_EVERY overrides
# the wrapper's evaluation cadence (CZECHLYNX_EVAL_EVERY).
epochs=${FEWSHOT_EPOCHS:-300}
final_epoch=${FEWSHOT_EVAL_EPOCH:-$(( epochs - 1 ))}
if [[ "${backend}" == loma ]]; then
  activate_conda_env "${CONDA_ENV_LOMA}"
  shared_cache=${CZECHLYNX_LOMA_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/loma-b-cache}
  wrapper=train_czechlynx_loma.sh; default_global_batch=32; component=${component:-matcher}
  case "${component}" in matcher) uses_cache=1 ;; descriptor|descriptor+matcher) uses_cache=0 ;; *) echo "unknown loma component ${component}" >&2; exit 2 ;; esac
  epoch_dir=$(printf 'epoch_%03d' "${final_epoch}")
  resume_marker=accelerate_state/random_states_0.pkl   # last file of a complete LoMa epoch
else
  activate_conda_env "${CONDA_ENV_RDD}"
  shared_cache=${CZECHLYNX_RDD_CACHE:-${CHECKPOINTS_ROOT}/czechlynx-time-closed/rdd-cache}
  wrapper=train_czechlynx_rdd.sh; default_global_batch=16; component=${component:-lg}
  case "${component}" in lg) uses_cache=1 ;; descriptor|lg+descriptor|rdd|lg+rdd) uses_cache=0 ;; *) echo "unknown rdd component ${component}" >&2; exit 2 ;; esac
  epoch_dir=$(printf 'epoch_%02d' "${final_epoch}")
  resume_marker=scheduler.pt                              # written after accelerator.save_state
fi
if [[ -f "${output_dir}/${epoch_dir}/model.safetensors" ]]; then
  echo "final checkpoint present (${output_dir}/${epoch_dir}); nothing to do"; exit 0
fi
resume_from=""
for candidate in $(ls -d "${output_dir}"/epoch_[0-9]* 2>/dev/null | sort -V -r); do
  if [[ -f "${candidate}/${resume_marker}" ]]; then resume_from=${candidate}; break; fi
done
(( uses_cache )) && { [[ -f "${shared_cache}/manifest.json" ]] || { echo "${backend} cache not found: ${shared_cache}" >&2; exit 1; }; }
[[ -f "${index_root}/strong-matches_train_combined.json" && -f "${index_root}/strong-matches_test_combined.json" ]] || { echo "indices missing under ${index_root}" >&2; exit 1; }

IFS=',' read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0}"
n_gpus=${#gpus[@]}
global_batch=${FEWSHOT_GLOBAL_BATCH:-${default_global_batch}}
(( global_batch % n_gpus == 0 )) || { echo "FEWSHOT_GLOBAL_BATCH=${global_batch} not divisible by ${n_gpus} GPUs" >&2; exit 1; }
cpus=${SLURM_CPUS_PER_TASK:-$(nproc)}
loader_workers=$(( (cpus - n_gpus) / n_gpus )); (( loader_workers > 8 )) && loader_workers=8; (( loader_workers < 1 )) && loader_workers=1

work=${TMPDIR:-/tmp}/fewshot-cz-${SLURM_JOB_ID:-$$}
mkdir -p "${work}"
trap 'rm -rf "${work}"' EXIT
cache=${shared_cache}
if (( uses_cache )) && [[ "${FEWSHOT_STAGE_CACHE:-1}" != 0 ]]; then
  need_mb=$(du -sm "${shared_cache}" | cut -f1); avail_mb=$(df -Pm "${work}" | awk 'NR==2 {print $4}')
  if (( avail_mb > need_mb * 12 / 10 + 2048 )); then
    t0=$(date +%s); mkdir -p "${work}/cache"; cp -r "${shared_cache}/." "${work}/cache/"; cache=${work}/cache
    echo "cache staged (${need_mb} MB) in $(( $(date +%s) - t0 ))s"
  else
    echo "not enough local space for the cache (${avail_mb} MB free, ${need_mb} MB needed); using ${shared_cache}"
  fi
fi
cp "${LYNX_FINETUNING_ROOT}/slurm_scripts/${wrapper}" "${work}/train.sh"   # private copy, see fewshot_job.sh
echo "== CzechLynx few-shot training (${backend}, component ${component}, ${epochs} epochs): ${run_name}  gpus=${n_gpus} batch=$(( global_batch / n_gpus ))x${n_gpus} workers=${loader_workers} cache=${cache}${resume_from:+ resume=${resume_from}}"
started=$(date +%s)
(export CZECHLYNX_SPLIT_PROTOCOL=legacy CZECHLYNX_ROOT=${view_root} CZECHLYNX_INDEX_ROOT=${index_root}
 export CZECHLYNX_RDD_CACHE=${cache} CZECHLYNX_LOMA_CACHE=${cache}
 export CZECHLYNX_RDD_TRAIN_COMPONENT=${component} CZECHLYNX_LOMA_TRAIN_COMPONENT=${component}
 export CZECHLYNX_RESUME_FROM=${resume_from} CZECHLYNX_EPOCHS=${epochs}
 [[ -n "${FEWSHOT_EVAL_EVERY:-}" ]] && export CZECHLYNX_EVAL_EVERY=${FEWSHOT_EVAL_EVERY}
 export CZECHLYNX_RDD_OUTPUT=${output_dir} CZECHLYNX_LOMA_OUTPUT=${output_dir}
 export CZECHLYNX_RDD_RUN_NAME=${run_name} CZECHLYNX_LOMA_RUN_NAME=${run_name}
 export CZECHLYNX_WANDB_PROJECT=${CZECHLYNX_WANDB_PROJECT:-czechlynx-fewshot-${backend}}
 export CZECHLYNX_NUM_PROCESSES=${n_gpus} CZECHLYNX_BATCH_SIZE=$(( global_batch / n_gpus )) CZECHLYNX_NUM_WORKERS=${loader_workers}
 cd "${LYNX_FINETUNING_ROOT}" && mkdir -p logs/czechlynx-rdd-ft logs/czechlynx-loma-ft && bash "${work}/train.sh")
[[ -f "${output_dir}/${epoch_dir}/model.safetensors" ]] || { echo "training finished but the final checkpoint is missing" >&2; exit 1; }
echo "== training done in $(( $(date +%s) - started ))s"
