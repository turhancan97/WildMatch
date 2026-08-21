#!/usr/bin/env bash
#SBATCH --job-name=wildlife-loma-ft
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=125G
#SBATCH --time=23:59:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=logs/wildlife-loma-ft/wildlife-loma-ft-%j.out
#SBATCH --error=logs/wildlife-loma-ft/wildlife-loma-ft-%j.err

set -euo pipefail
source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate loma
export WANDB_MODE=online

benchmark_root=${WILDLIFE_BENCHMARK_ROOT:-/home/kargin/Projects/repositories/rdd-parallel-benchmark}
config=${WILDLIFE_CONFIG:-${benchmark_root}/configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(cd "${benchmark_root}" && python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
index_root=${WILDLIFE_INDEX_ROOT:-${benchmark_root}/outputs/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/indices}
train_index=${WILDLIFE_TRAIN_INDEX:-${index_root}/strong-matches_train_combined.json}
if [[ "${protocol}" == legacy ]]; then default_val_index=${index_root}/strong-matches_test_combined.json; else default_val_index=${index_root}/strong-matches_val_combined.json; fi
val_index=${WILDLIFE_VAL_INDEX:-${default_val_index}}
loma_weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
cache_dir=${WILDLIFE_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/loma-cache}
output_dir=${WILDLIFE_LOMA_OUTPUT:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/loma-finetuned/${protocol}}
run_name=${WILDLIFE_LOMA_RUN_NAME:-${WILDLIFE_DATASET_ID}-loma-${protocol}-finetuned}
resume_args=()
if [[ -n "${WILDLIFE_RESUME:-}" ]]; then
  resume_args+=(--resume "${WILDLIFE_RESUME}")
fi

echo "dataset=${WILDLIFE_DATASET_ID} protocol=${protocol}"
echo "training index=${train_index}"
echo "validation index=${val_index}"
echo "output directory=${output_dir}"
[[ -f "${train_index}" && -f "${val_index}" ]] || { echo "missing training or validation index" >&2; exit 1; }
[[ -f "${cache_dir}/manifest.json" ]] || bash "${benchmark_root}/../lynx-finetuning/slurm_scripts/build_wildlife_loma_cache.sh"

mkdir -p "${output_dir}"
cat > "${output_dir}/wildlife_protocol.json" <<EOF
{"dataset": "${WILDLIFE_DATASET_ID}", "protocol": "${protocol}", "train_index": "${train_index}", "validation_index": "${val_index}", "final_evaluation_split": "test"}
EOF

accelerate launch --num_processes "${WILDLIFE_NUM_PROCESSES:-4}" --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  -m contrastive_finetuning.train_loma_matches \
  --trained_model loma --train_index "${train_index}" --val_index "${val_index}" \
  --data_root "${dataset_root}" --loma_weights "${loma_weights}" \
  --loma_cache "${cache_dir}" --output_dir "${output_dir}" \
  --project "${WILDLIFE_WANDB_PROJECT:-wildlife-reid-loma}" \
  --run_name "${run_name}" --split_protocol "${protocol}" --wandb_mode online \
  --loma_variant "${WILDLIFE_LOMA_VARIANT:-loma-b}" --epochs "${WILDLIFE_EPOCHS:-300}" \
  --batch_size "${WILDLIFE_BATCH_SIZE:-8}" --lr 1e-5 --weight_decay 1e-4 \
  --margin 0.5 --random_negative_prob 0.3 --num_workers "${WILDLIFE_NUM_WORKERS:-10}" \
  --eval_every_epochs 10 --seed 0 --resize 512 --num_keypoints 512 "${resume_args[@]}"
