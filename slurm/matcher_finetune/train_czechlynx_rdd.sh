#!/bin/bash
#SBATCH --job-name=czechlynx-rdd-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --exclude=c11,c15
#SBATCH --output=logs/czechlynx-rdd-ft/czechlynx-rdd-ft-%j.out
#SBATCH --error=logs/czechlynx-rdd-ft/czechlynx-rdd-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd
export WANDB_MODE=online
# LoMa-mined pairs for every RDD run (2026-09-29): the same pair source as the
# LoMa runs and the paper checkpoints. Set CZECHLYNX_MINING_BACKEND=rdd explicitly
# for RDD-mined pairs.
export CZECHLYNX_MINING_BACKEND=${CZECHLYNX_MINING_BACKEND:-loma}
source /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/czechlynx_protocol.sh
czechlynx_resolve_protocol

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_${CZECHLYNX_RESOLVED_SPLIT_SUFFIX}}
train_index=${CZECHLYNX_RESOLVED_TRAIN_INDEX}
val_index=${CZECHLYNX_RESOLVED_VAL_INDEX}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
lg_weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
cache_root=${CZECHLYNX_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/rdd-cache}
component=${CZECHLYNX_RDD_TRAIN_COMPONENT:-lg}
case "${component}" in
  lg) trained_model=lg; rdd_component=all ;;
  descriptor) trained_model=rdd; rdd_component=descriptor ;;
  rdd|all) trained_model=rdd; rdd_component=all ;;
  lg+rdd) trained_model=lg+rdd; rdd_component=all ;;
  # joint: LightGlue and RDD's descriptor trained together, RDD's detector frozen.
  joint) trained_model=lg+rdd; rdd_component=descriptor ;;
  *) echo "CZECHLYNX_RDD_TRAIN_COMPONENT must be lg, descriptor, joint, rdd, or lg+rdd (got ${component})" >&2; exit 2 ;;
esac
# Default names encode the pair source, e.g. rdd-finetuned-loma-mined-legacy —
# the directory names the probe launchers already point to.
mined_suffix=${CZECHLYNX_RESOLVED_BACKEND}-mined-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
checkpoint_root=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}
if [[ -n "${CZECHLYNX_RDD_OUTPUT:-}" ]]; then
  output_dir=${CZECHLYNX_RDD_OUTPUT}
elif [[ "${component}" == descriptor ]]; then
  output_dir=${checkpoint_root}/rdd-descriptor-finetuned-${mined_suffix}
elif [[ "${component}" == joint ]]; then
  output_dir=${checkpoint_root}/rdd-joint-finetuned-${mined_suffix}
elif [[ "${component}" == lg ]]; then
  output_dir=${checkpoint_root}/rdd-finetuned-${mined_suffix}
else
  output_dir=${checkpoint_root}/rdd-full-finetuned-${mined_suffix}
fi
if [[ -n "${CZECHLYNX_RDD_BATCH_SIZE:-}" ]]; then
  batch_size=${CZECHLYNX_RDD_BATCH_SIZE}
elif [[ "${component}" == descriptor || "${component}" == joint ]]; then
  # Descriptor (and joint) training keeps three RDD autograd graphs alive (query,
  # positive, negative).  With the PyTorch deformable-attention fallback,
  # batch_size=8 exceeds the 24 GB GPU limit on CzechLynx images, so it
  # loads 1 per GPU and accumulates 8 of them per step (below).
  batch_size=1
else
  batch_size=8
fi
# Effective batch = batch_size x grad_accum_steps x 4 GPUs = 32 in every mode,
# the same as the LoMa trainer (whose descriptor mode microbatches 8 -> 1).
if [[ -n "${CZECHLYNX_RDD_GRAD_ACCUM_STEPS:-}" ]]; then
  grad_accum_steps=${CZECHLYNX_RDD_GRAD_ACCUM_STEPS}
elif [[ "${component}" == descriptor || "${component}" == joint ]]; then
  grad_accum_steps=8
else
  grad_accum_steps=1
fi

# Resume: CZECHLYNX_RDD_RESUME=auto picks the newest epoch_* directory in the
# output directory that has a train_state.json; an explicit path is used as is.
# Without it, an output directory that already holds epoch checkpoints is
# refused rather than silently overwritten (archive it first, or resume).
resume_dir=${CZECHLYNX_RDD_RESUME:-}
if [[ "${resume_dir}" == auto ]]; then
  resume_dir=$(ls -d "${output_dir}"/epoch_* 2>/dev/null | while read -r d; do
    [[ -f "${d}/train_state.json" ]] && echo "${d}"; done | sort -V | tail -n 1 || true)
  if [[ -z "${resume_dir}" ]]; then
    echo "CZECHLYNX_RDD_RESUME=auto: no resumable epoch_* directory in ${output_dir}" >&2
    exit 2
  fi
elif [[ -z "${resume_dir}" ]] && compgen -G "${output_dir}/epoch_*" > /dev/null; then
  echo "${output_dir} already contains epoch checkpoints; set CZECHLYNX_RDD_RESUME=auto to continue it," >&2
  echo "or archive the directory before starting a new run." >&2
  exit 2
fi
if [[ -n "${CZECHLYNX_RDD_RUN_NAME:-}" ]]; then
  run_name=${CZECHLYNX_RDD_RUN_NAME}
elif [[ "${component}" == descriptor ]]; then
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd-descriptor-${CZECHLYNX_RESOLVED_PROTOCOL}-relaxed
elif [[ "${component}" == joint ]]; then
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd-joint-${CZECHLYNX_RESOLVED_PROTOCOL}-relaxed
elif [[ "${component}" == lg ]]; then
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd-${CZECHLYNX_RESOLVED_PROTOCOL}-relaxed
else
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-${component}-${CZECHLYNX_RESOLVED_PROTOCOL}-relaxed
fi

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "CzechLynx split column: ${CZECHLYNX_RESOLVED_SPLIT_COLUMN}"
echo "CzechLynx mining backend: ${CZECHLYNX_RESOLVED_BACKEND}"
echo "RDD training component: ${component} (trained_model=${trained_model}, rdd component=${rdd_component})"
echo "training index: ${train_index}"
echo "validation index: ${val_index}"
echo "output directory: ${output_dir}"
echo "batch size per GPU: ${batch_size} (grad_accum_steps=${grad_accum_steps})"
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
  "rdd_train_component": "${component}",
  "trained_model": "${trained_model}",
  "rdd_component": "${rdd_component}",
  "training_score": "relaxed_v1",
  "optimizer": "adamw",
  "batch_size_per_gpu": ${batch_size},
  "grad_accum_steps": ${grad_accum_steps},
  "num_gpus": 4
}
EOF

mkdir -p logs
if [[ "${trained_model}" == lg && ! -f "${cache_root}/manifest.json" ]]; then
  python -m contrastive_finetuning.build_keypoint_cache \
    --data_root "${dataset_root}" --cache_root "${cache_root}" \
    --rdd_weights "${rdd_weights}" --splits train val test \
    --resize 512 --top_k 512 \
    --batch_size "${RDD_CACHE_BATCH_SIZE:-32}" --num_workers 16 --resume
fi

args=(
  --train_index "${train_index}" --val_index "${val_index}"
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}"
  --lg_weights "${lg_weights}" --output_dir "${output_dir}"
  --project "${CZECHLYNX_RDD_PROJECT:-lynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd}"
  --run_name "${run_name}" --split_protocol "${CZECHLYNX_RESOLVED_PROTOCOL}"
  --trained_model "${trained_model}" --rdd_train_component "${rdd_component}"
  --epochs 300 --batch_size "${batch_size}" --grad_accum_steps "${grad_accum_steps}" --keep_every "${RDD_KEEP_EVERY:-50}"
  --lr 1e-5 --weight_decay 1e-4
  --num_workers 8 --lg_margin 0.5 --random_negative_prob 0.3
  --resize 512 --top_k 512 --seed 0
)
if [[ "${trained_model}" == lg ]]; then
  args+=(--keypoint_cache "${cache_root}")
fi
if [[ -n "${resume_dir}" ]]; then
  args+=(--resume "${resume_dir}")
fi

accelerate launch --num_processes 4 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  "${args[@]}"
