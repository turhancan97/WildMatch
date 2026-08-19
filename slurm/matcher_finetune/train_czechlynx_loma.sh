#!/bin/bash
#SBATCH --job-name=czechlynx-loma-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclude=c11,c15,c22
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --output=logs/czechlynx-loma-ft-%j.out
#SBATCH --error=logs/czechlynx-loma-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma
export WANDB_MODE=online
source /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/czechlynx_protocol.sh
czechlynx_resolve_protocol

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
train_index=${CZECHLYNX_RESOLVED_TRAIN_INDEX}
val_index=${CZECHLYNX_RESOLVED_VAL_INDEX}
loma_weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
cache_dir=${CZECHLYNX_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-cache}
output_dir=${CZECHLYNX_LOMA_OUTPUT:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/loma-b-finetuned-${CZECHLYNX_RESOLVED_OUTPUT_SUFFIX}}
run_name=${CZECHLYNX_LOMA_RUN_NAME:-czechlynx-time-closed-loma-${CZECHLYNX_RESOLVED_PROTOCOL}}
benchmark_root=/home/kargin/Projects/repositories/rdd-parallel-benchmark

echo "CzechLynx split protocol: ${CZECHLYNX_RESOLVED_PROTOCOL}"
echo "training index: ${train_index}"
echo "validation index: ${val_index}"
echo "output directory: ${output_dir}"

mkdir -p "${output_dir}"
cat > "${output_dir}/czechlynx_protocol.json" <<EOF
{
  "protocol": "${CZECHLYNX_RESOLVED_PROTOCOL}",
  "train_index": "${train_index}",
  "validation_index": "${val_index}",
  "final_evaluation_split": "test"
}
EOF

mkdir -p logs
if [[ ! -f "${cache_dir}/manifest.json" ]]; then
  (cd "${benchmark_root}" && python -m scripts.lynx_build_loma_cache \
    --dataset_root "${dataset_root}" --cache_dir "${cache_dir}" \
    --weights "${loma_weights}" --variant loma-b --all_frames \
    --splits train val test --resize_max 512 --num_keypoints 512 \
    --batch_size "${LOMA_CACHE_BATCH_SIZE:-4}" --num_workers 16 --resume)
fi

accelerate launch --num_processes 4 --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_loma_matches \
  --trained_model loma --train_index "${train_index}" \
  --val_index "${val_index}" --data_root "${dataset_root}" \
  --loma_weights "${loma_weights}" --loma_cache "${cache_dir}" \
  --output_dir "${output_dir}" --project lynx-czechlynx-loma \
  --run_name "${run_name}" --split_protocol "${CZECHLYNX_RESOLVED_PROTOCOL}" --wandb_mode online \
  --loma_variant loma-b --epochs 300 --batch_size 8 --lr 1e-5 \
  --weight_decay 1e-4 --margin 0.5 --random_negative_prob 0.3 \
  --num_workers 10 --eval_every_epochs 10 --seed 0 --resize 512 \
  --num_keypoints 512
