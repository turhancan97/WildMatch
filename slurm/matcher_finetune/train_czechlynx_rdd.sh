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
export CZECHLYNX_MINING_BACKEND=${CZECHLYNX_MINING_BACKEND:-rdd}
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
  *) echo "CZECHLYNX_RDD_TRAIN_COMPONENT must be lg, descriptor, rdd, or lg+rdd (got ${component})" >&2; exit 2 ;;
esac
if [[ -n "${CZECHLYNX_RDD_OUTPUT:-}" ]]; then
  output_dir=${CZECHLYNX_RDD_OUTPUT}
elif [[ "${component}" == descriptor ]]; then
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/rdd-descriptor-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
elif [[ "${component}" == lg ]]; then
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/rdd-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
else
  output_dir=/shared/sets/datasets/vision/czechlynx/checkpoints/${CZECHLYNX_RESOLVED_EXPERIMENT}/rdd-full-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}
fi
if [[ -n "${CZECHLYNX_RDD_RUN_NAME:-}" ]]; then
  run_name=${CZECHLYNX_RDD_RUN_NAME}
elif [[ "${component}" == descriptor ]]; then
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd-descriptor-${CZECHLYNX_RESOLVED_PROTOCOL}
elif [[ "${component}" == lg ]]; then
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-rdd-${CZECHLYNX_RESOLVED_PROTOCOL}
else
  run_name=czechlynx-${CZECHLYNX_RESOLVED_EXPERIMENT}-${component}-${CZECHLYNX_RESOLVED_PROTOCOL}
fi

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "CzechLynx split column: ${CZECHLYNX_RESOLVED_SPLIT_COLUMN}"
echo "CzechLynx mining backend: ${CZECHLYNX_RESOLVED_BACKEND}"
echo "RDD training component: ${component} (trained_model=${trained_model}, rdd component=${rdd_component})"
echo "training index: ${train_index}"
echo "validation index: ${val_index}"
echo "output directory: ${output_dir}"

mkdir -p "${output_dir}"
cat > "${output_dir}/czechlynx_protocol.json" <<EOF
{
  "split_column": "${CZECHLYNX_RESOLVED_SPLIT_COLUMN}",
  "backend": "${CZECHLYNX_RESOLVED_BACKEND}",
  "protocol": "${CZECHLYNX_RESOLVED_PROTOCOL}",
  "train_index": "${train_index}",
  "validation_index": "${val_index}",
  "final_evaluation_split": "test",
  "rdd_train_component": "${component}"
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
  --epochs 300 --batch_size 8 --lr 1e-5 --weight_decay 1e-4
  --num_workers 8 --lg_margin 0.5 --random_negative_prob 0.3
  --resize 512 --top_k 512 --seed 0
)
if [[ "${trained_model}" == lg ]]; then
  args+=(--keypoint_cache "${cache_root}")
fi

accelerate launch --num_processes 4 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_by_lg_matches \
  "${args[@]}"
