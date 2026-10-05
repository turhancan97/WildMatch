#!/usr/bin/env bash
#SBATCH --job-name=wildlife-rdd-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=logs/wildlife-rdd-ft/wildlife-rdd-ft-%j.out
#SBATCH --error=logs/wildlife-rdd-ft/wildlife-rdd-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd
export WANDB_MODE=online

benchmark_root=${WILDLIFE_BENCHMARK_ROOT:-/home/kargin/Projects/repositories/rdd-parallel-benchmark}
config=${WILDLIFE_CONFIG:-${benchmark_root}/configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(cd "${benchmark_root}" && python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
index_base=${WILDLIFE_INDEX_ROOT:-${benchmark_root}/outputs/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/indices}
# Pair source is explicit (2026-09-29). This used to prefer indices/loma when it
# existed and silently fall back to indices/ otherwise, so datasets without LoMa
# mining (NyalaData, WhaleSharkID) were trained on RDD-mined pairs. Every RDD run
# now uses LoMa-mined pairs unless WILDLIFE_MINING_BACKEND says otherwise, and a
# missing index is an error instead of a fallback.
mining_backend=${WILDLIFE_MINING_BACKEND:-loma}
case "${mining_backend}" in
  loma|rdd) ;;
  *) echo "WILDLIFE_MINING_BACKEND must be loma or rdd (got ${mining_backend})" >&2; exit 2 ;;
esac
if [[ -n "${WILDLIFE_INDEX_ROOT:-}" ]]; then
  index_root=${index_base}
else
  index_root="${index_base}/${mining_backend}"
fi
train_index=${WILDLIFE_TRAIN_INDEX:-${index_root}/strong-matches_train_combined.json}
if [[ "${protocol}" == legacy ]]; then
  default_val_index=${index_root}/strong-matches_test_combined.json
else
  default_val_index=${index_root}/strong-matches_val_combined.json
fi
val_index=${WILDLIFE_VAL_INDEX:-${default_val_index}}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
lg_weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
cache_root=${WILDLIFE_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-cache}
component=${WILDLIFE_RDD_TRAIN_COMPONENT:-lg}
case "${component}" in
  lg) trained_model=lg; rdd_component=all ;;
  descriptor) trained_model=rdd; rdd_component=descriptor ;;
  rdd|all) trained_model=rdd; rdd_component=all ;;
  lg+rdd) trained_model=lg+rdd; rdd_component=all ;;
  *) echo "WILDLIFE_RDD_TRAIN_COMPONENT must be lg, descriptor, rdd, or lg+rdd (got ${component})" >&2; exit 2 ;;
esac
if [[ -n "${WILDLIFE_RDD_OUTPUT:-}" ]]; then
  output_dir=${WILDLIFE_RDD_OUTPUT}
elif [[ "${component}" == descriptor ]]; then
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-descriptor-finetuned/${protocol}
elif [[ "${component}" == lg ]]; then
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-finetuned/${protocol}
else
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-full-finetuned/${protocol}
fi
if [[ -n "${WILDLIFE_RDD_RUN_NAME:-}" ]]; then
  run_name=${WILDLIFE_RDD_RUN_NAME}
elif [[ "${component}" == descriptor ]]; then
  run_name=${WILDLIFE_DATASET_ID}-rdd-descriptor-${protocol}-finetuned-relaxed
elif [[ "${component}" == lg ]]; then
  run_name=${WILDLIFE_DATASET_ID}-rdd-${protocol}-finetuned-relaxed
else
  run_name=${WILDLIFE_DATASET_ID}-${component}-${protocol}-finetuned-relaxed
fi
# Effective batch = batch_size x grad_accum_steps x GPUs = 32 by default. Descriptor
# mode loads 1 per GPU (memory) and accumulates 8, as in train_czechlynx_rdd.sh.
if [[ "${component}" == descriptor ]]; then
  batch_size=${WILDLIFE_BATCH_SIZE:-1}
  grad_accum_steps=${WILDLIFE_GRAD_ACCUM_STEPS:-8}
else
  batch_size=${WILDLIFE_BATCH_SIZE:-8}
  grad_accum_steps=${WILDLIFE_GRAD_ACCUM_STEPS:-1}
fi
# Resume (see train_czechlynx_rdd.sh): WILDLIFE_RDD_RESUME=auto or a path; a
# directory that already holds epoch checkpoints is never silently overwritten.
resume_dir=${WILDLIFE_RDD_RESUME:-}
if [[ "${resume_dir}" == auto ]]; then
  resume_dir=$(ls -d "${output_dir}"/epoch_* 2>/dev/null | while read -r d; do
    [[ -f "${d}/train_state.json" ]] && echo "${d}"; done | sort -V | tail -n 1 || true)
  if [[ -z "${resume_dir}" ]]; then
    echo "WILDLIFE_RDD_RESUME=auto: no resumable epoch_* directory in ${output_dir}" >&2
    exit 2
  fi
elif [[ -z "${resume_dir}" ]] && compgen -G "${output_dir}/epoch_*" > /dev/null; then
  echo "${output_dir} already contains epoch checkpoints; set WILDLIFE_RDD_RESUME=auto to continue it," >&2
  echo "or archive the directory before starting a new run." >&2
  exit 2
fi
wandb_project=${WILDLIFE_WANDB_PROJECT:-wildlife-reid-rdd-${WILDLIFE_DATASET_ID}-${protocol}}

echo "dataset=${WILDLIFE_DATASET_ID} protocol=${protocol} component=${component} trained_model=${trained_model}"
echo "training index=${train_index}"
echo "validation index=${val_index}"
echo "mining backend=${mining_backend}"
echo "output directory=${output_dir}"
echo "batch size per GPU=${batch_size} grad_accum_steps=${grad_accum_steps} resume=${resume_dir:-<fresh run>}"
[[ -f "${train_index}" && -f "${val_index}" ]] || { echo "missing training or validation index" >&2; exit 1; }
if [[ "${trained_model}" == lg ]]; then
  [[ -f "${cache_root}/manifest.json" ]] || bash "${benchmark_root}/../lynx-finetuning/slurm_scripts/build_wildlife_rdd_cache.sh"
fi

mkdir -p "${output_dir}"
cat > "${output_dir}/wildlife_protocol.json" <<EOF
{"dataset": "${WILDLIFE_DATASET_ID}", "protocol": "${protocol}", "backend": "${mining_backend}", "train_component": "${component}", "train_index": "${train_index}", "validation_index": "${val_index}", "final_evaluation_split": "test", "training_score": "relaxed_v1", "optimizer": "adamw", "batch_size_per_gpu": ${batch_size}, "grad_accum_steps": ${grad_accum_steps}, "num_gpus": ${WILDLIFE_NUM_PROCESSES:-4}}
EOF

args=(
  --train_index "${train_index}" --val_index "${val_index}"
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}"
  --lg_weights "${lg_weights}" --output_dir "${output_dir}"
  --project "${wandb_project}" --run_name "${run_name}"
  --split_protocol "${protocol}" --trained_model "${trained_model}"
  --rdd_train_component "${rdd_component}"
  --epochs "${WILDLIFE_EPOCHS:-300}" --batch_size "${batch_size}"
  --grad_accum_steps "${grad_accum_steps}" --keep_every "${RDD_KEEP_EVERY:-50}"
  --lr 1e-5 --weight_decay 1e-4 --num_workers "${WILDLIFE_NUM_WORKERS:-8}"
  --lg_margin 0.5 --random_negative_prob 0.3 --resize 512 --top_k 512
  --eval_every_epochs 10 --seed 0
)
if [[ "${trained_model}" == lg ]]; then
  args+=(--keypoint_cache "${cache_root}")
fi
if [[ -n "${resume_dir}" ]]; then
  args+=(--resume "${resume_dir}")
fi

accelerate launch --num_processes "${WILDLIFE_NUM_PROCESSES:-4}" --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  "${args[@]}"
