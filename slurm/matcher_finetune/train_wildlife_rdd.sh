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
if [[ -n "${WILDLIFE_INDEX_ROOT:-}" ]]; then
  index_root=${index_base}
elif [[ -f "${index_base}/loma/strong-matches_train_combined.json" ]]; then
  index_root="${index_base}/loma"
else
  # Fall back to the original shared path for existing experiments.
  index_root=${index_base}
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
  run_name=${WILDLIFE_DATASET_ID}-rdd-descriptor-${protocol}-finetuned
elif [[ "${component}" == lg ]]; then
  run_name=${WILDLIFE_DATASET_ID}-rdd-${protocol}-finetuned
else
  run_name=${WILDLIFE_DATASET_ID}-${component}-${protocol}-finetuned
fi
wandb_project=${WILDLIFE_WANDB_PROJECT:-wildlife-reid-rdd-${WILDLIFE_DATASET_ID}-${protocol}}

echo "dataset=${WILDLIFE_DATASET_ID} protocol=${protocol} component=${component} trained_model=${trained_model}"
echo "training index=${train_index}"
echo "validation index=${val_index}"
echo "output directory=${output_dir}"
[[ -f "${train_index}" && -f "${val_index}" ]] || { echo "missing training or validation index" >&2; exit 1; }
if [[ "${trained_model}" == lg ]]; then
  [[ -f "${cache_root}/manifest.json" ]] || bash "${benchmark_root}/../lynx-finetuning/slurm_scripts/build_wildlife_rdd_cache.sh"
fi

mkdir -p "${output_dir}"
cat > "${output_dir}/wildlife_protocol.json" <<EOF
{"dataset": "${WILDLIFE_DATASET_ID}", "protocol": "${protocol}", "train_component": "${component}", "train_index": "${train_index}", "validation_index": "${val_index}", "final_evaluation_split": "test"}
EOF

args=(
  --train_index "${train_index}" --val_index "${val_index}"
  --data_root "${dataset_root}" --rdd_weights "${rdd_weights}"
  --lg_weights "${lg_weights}" --output_dir "${output_dir}"
  --project "${wandb_project}" --run_name "${run_name}"
  --split_protocol "${protocol}" --trained_model "${trained_model}"
  --rdd_train_component "${rdd_component}"
  --epochs "${WILDLIFE_EPOCHS:-300}" --batch_size "${WILDLIFE_BATCH_SIZE:-8}"
  --lr 1e-5 --weight_decay 1e-4 --num_workers "${WILDLIFE_NUM_WORKERS:-8}"
  --lg_margin 0.5 --random_negative_prob 0.3 --resize 512 --top_k 512
  --eval_every_epochs 10 --seed 0
)
if [[ "${trained_model}" == lg ]]; then
  args+=(--keypoint_cache "${cache_root}")
fi

accelerate launch --num_processes "${WILDLIFE_NUM_PROCESSES:-4}" --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  "${args[@]}"
