#!/bin/bash
#SBATCH --job-name=czechlynx-loma-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclude=c11,c15,c22
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-loma-ft/czechlynx-loma-ft-%j.out
#SBATCH --error=logs/czechlynx-loma-ft/czechlynx-loma-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma
export WANDB_MODE=online
export CZECHLYNX_MINING_BACKEND=${CZECHLYNX_MINING_BACKEND:-loma}
source /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/czechlynx_protocol.sh
czechlynx_resolve_protocol

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_${CZECHLYNX_RESOLVED_SPLIT_SUFFIX}}
train_index=${CZECHLYNX_RESOLVED_TRAIN_INDEX}
val_index=${CZECHLYNX_RESOLVED_VAL_INDEX}
loma_weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
cache_dir=${CZECHLYNX_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/loma-b-cache}
train_component=${CZECHLYNX_LOMA_TRAIN_COMPONENT:-matcher}
keypoint_cache=${CZECHLYNX_LOMA_KEYPOINT_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/loma-b-keypoint-cache}
if [[ -n "${CZECHLYNX_LOMA_OUTPUT:-}" ]]; then
  output_dir=${CZECHLYNX_LOMA_OUTPUT}
elif [[ "${train_component}" == descriptor ]]; then
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/loma-b-descriptor-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
elif [[ "${train_component}" == joint ]]; then
  # DeDoDe + matcher trained together (DaD frozen); the name records the pair source.
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/loma-b-joint-finetuned-${CZECHLYNX_RESOLVED_BACKEND}-mined-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
else
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/loma-b-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
fi
if [[ "${train_component}" == descriptor ]]; then
  default_run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-loma-descriptor-${CZECHLYNX_RESOLVED_PROTOCOL}
elif [[ "${train_component}" == joint ]]; then
  default_run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-loma-joint-${CZECHLYNX_RESOLVED_PROTOCOL}
else
  default_run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-loma-${CZECHLYNX_RESOLVED_PROTOCOL}
fi
run_name=${CZECHLYNX_LOMA_RUN_NAME:-${default_run_name}}
benchmark_root=/home/kargin/Projects/repositories/rdd-parallel-benchmark

# Resume: CZECHLYNX_LOMA_RESUME=auto picks the newest epoch_* directory with a
# metadata.json; an explicit epoch directory is used as is. Without it, an
# output directory that already holds epoch checkpoints is refused rather than
# silently overwritten (archive it first, or resume).
resume_dir=${CZECHLYNX_LOMA_RESUME:-}
if [[ "${resume_dir}" == auto ]]; then
  resume_dir=$(ls -d "${output_dir}"/epoch_* 2>/dev/null | while read -r d; do
    [[ -f "${d}/metadata.json" ]] && echo "${d}"; done | sort -V | tail -n 1 || true)
  if [[ -z "${resume_dir}" ]]; then
    echo "CZECHLYNX_LOMA_RESUME=auto: no resumable epoch_* directory in ${output_dir}" >&2
    exit 2
  fi
elif [[ -z "${resume_dir}" ]] && compgen -G "${output_dir}/epoch_*" > /dev/null; then
  echo "${output_dir} already contains epoch checkpoints; set CZECHLYNX_LOMA_RESUME=auto to continue it," >&2
  echo "or archive the directory before starting a new run." >&2
  exit 2
fi

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "CzechLynx split column: ${CZECHLYNX_RESOLVED_SPLIT_COLUMN}"
echo "CzechLynx mining backend: ${CZECHLYNX_RESOLVED_BACKEND}"
echo "LoMa training component: ${train_component}"
echo "training index: ${train_index}"
echo "validation index: ${val_index}"
echo "output directory: ${output_dir}"
echo "resume from: ${resume_dir:-<fresh run>}"

mkdir -p "${output_dir}"
cat > "${output_dir}/czechlynx_protocol.json" <<EOF
{
  "split_column": "${CZECHLYNX_RESOLVED_SPLIT_COLUMN}",
  "backend": "${CZECHLYNX_RESOLVED_BACKEND}",
  "protocol": "${CZECHLYNX_RESOLVED_PROTOCOL}",
  "train_index": "${train_index}",
  "validation_index": "${val_index}",
  "final_evaluation_split": "test",
  "loma_train_component": "${train_component}",
  "training_score": "relaxed_v1",
  "optimizer": "adamw"
}
EOF

mkdir -p logs
# Under sbatch, Slurm runs a copy of this script from its spool directory
# (e.g. /var/spool/slurmd/job<id>/), so the script's own location is not the
# repository. Accept only a directory that contains contrastive_finetuning/:
# the explicit override, the script location, then the submission directory.
repo_root=""
for candidate in "${LYNX_FINETUNING_ROOT:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "${candidate}" && -d "${candidate}/contrastive_finetuning" ]]; then
    repo_root=${candidate}
    break
  fi
done
if [[ -z "${repo_root}" ]]; then
  echo "cannot locate the lynx-finetuning repository; submit from its root or set LYNX_FINETUNING_ROOT" >&2
  exit 2
fi
cd "${repo_root}"
if [[ "${train_component}" == matcher ]]; then
  if [[ ! -f "${cache_dir}/manifest.json" ]]; then
    (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
      --dataset_root "${dataset_root}" --cache_dir "${cache_dir}" \
      --weights "${loma_weights}" --variant loma-b --all_frames \
      --splits train val test --resize_max 512 --num_keypoints 512 \
      --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}" --num_workers 16 --resume)
  fi
elif [[ "${train_component}" == descriptor || "${train_component}" == joint ]]; then
  # DaD is frozen in both modes, so its keypoints can be cached once.
  python -m contrastive_finetuning.build_loma_keypoint_cache \
    --data_root "${dataset_root}" --cache_dir "${keypoint_cache}" \
    --weights "${loma_weights}" --variant loma-b --splits train val test \
    --resize 512 --num_keypoints 512 --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}"
else
  echo "CZECHLYNX_LOMA_TRAIN_COMPONENT must be matcher, descriptor, or joint, got ${train_component}" >&2
  exit 2
fi

args=(
  --trained_model loma --loma_train_component "${train_component}"
  --train_index "${train_index}" --val_index "${val_index}"
  --data_root "${dataset_root}" --loma_weights "${loma_weights}"
  --output_dir "${output_dir}"
  --project "${CZECHLYNX_LOMA_PROJECT:-lynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-loma}"
  --run_name "${run_name}" --split_protocol "${CZECHLYNX_RESOLVED_PROTOCOL}" --wandb_mode online
  --loma_variant loma-b --epochs 300 --batch_size 8 --lr 1e-5
  --weight_decay 1e-4 --margin 0.5 --random_negative_prob 0.3
  --num_workers 10 --eval_every_epochs 10 --seed 0 --resize 512
  --num_keypoints 512 --descriptor_microbatch_size "${CZECHLYNX_LOMA_DESCRIPTOR_MICROBATCH_SIZE:-1}"
  --keep_every "${LOMA_KEEP_EVERY:-50}"
)
if [[ "${train_component}" == matcher ]]; then
  args+=(--loma_cache "${cache_dir}")
else
  args+=(--loma_keypoint_cache "${keypoint_cache}")
fi
if [[ -n "${resume_dir}" ]]; then
  args+=(--resume "${resume_dir}")
fi

accelerate launch --num_processes 4 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_loma_matches \
  "${args[@]}"
