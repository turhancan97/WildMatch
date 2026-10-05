#!/usr/bin/env bash
#SBATCH --job-name=wm-spike-train
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%j.out
#SBATCH --error=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%j.err
# Integration plan D3 spike (parity L5): one epoch of Salamander LoMa and RDD-LightGlue matcher
# training, old code (lynx-finetuning 6958d3d, conda envs loma/rdd, torch 2.13) twice and new code
# (wildmatch integrate/matcher-finetune, uv env wildmatch-integration, torch 2.8) once, on one GPU.
# Arguments are those recorded by the paper's Salamander runs, except epochs/eval/W&B/output.
set -euo pipefail
SPIKE=/shared/results/common/kargin/projects/wildmatch-integration/spike
OUT=$SPIKE/runs/${SLURM_JOB_ID}
OLD_REPO=/home/kargin/Projects/repositories/lynx-finetuning
NEW_REPO=/shared/results/common/kargin/projects/wildmatch-integration/wt
NEW_ENV=/shared/results/common/kargin/projects/uv-environment/wildmatch-integration
mkdir -p "$OUT"
export WANDB_MODE=disabled WANDB_DIR=$OUT HF_HOME=/shared/results/common/kargin/.cache/huggingface
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
git -C "$OLD_REPO" rev-parse HEAD > "$OUT/old_commit.txt"; git -C "$OLD_REPO" status --short >> "$OUT/old_commit.txt"
git -C "$NEW_REPO" rev-parse HEAD > "$OUT/new_commit.txt"; git -C "$NEW_REPO" status --short >> "$OUT/new_commit.txt"

IDX=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/wildlife-reid-10k/SalamanderID2025/indices
CK=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025
VIEW=/shared/sets/datasets/vision/czechlynx/wildlife_processed/SalamanderID2025/legacy
RDDW=/home/kargin/Projects/repositories/rdd/weights
LAUNCH=(accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no)

loma_args() {  # $1 output dir
  echo --trained_model loma --loma_train_component matcher \
    --train_index $IDX/loma/strong-matches_train_combined.json --val_index $IDX/loma/strong-matches_test_combined.json \
    --data_root $VIEW --loma_weights /shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt \
    --loma_cache $CK/loma-cache --output_dir "$1" --project spike --run_name spike --split_protocol legacy \
    --wandb_mode disabled --loma_variant loma-b --epochs 1 --batch_size 16 --lr 1e-5 --weight_decay 1e-4 \
    --margin 0.5 --random_negative_prob 0.3 --num_workers 8 --eval_every_epochs 1 --max_val_entries 64 \
    --seed 0 --resize 512 --num_keypoints 512 --descriptor_microbatch_size 1
}
rdd_args() {  # $1 output dir
  echo --train_index $IDX/rdd/strong-matches_train_combined.json --val_index $IDX/rdd/strong-matches_test_combined.json \
    --data_root $VIEW --rdd_weights $RDDW/RDD-v2.pth --lg_weights $RDDW/RDD_lg-v2.pth --output_dir "$1" \
    --project spike --run_name spike --split_protocol legacy --trained_model lg --rdd_train_component all \
    --epochs 1 --batch_size 8 --grad_accum_steps 1 --keep_every 50 --lr 1e-5 --weight_decay 1e-4 \
    --num_workers 8 --lg_margin 0.5 --random_negative_prob 0.3 --resize 512 --top_k 512 \
    --eval_every_epochs 1 --seed 0 --keypoint_cache $CK/rdd-cache
}
run_old() {  # $1 env, $2 module, $3 name, $4 args function
  ( source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh; conda activate "$1"; cd "$OLD_REPO"
    python -c "import torch; print('torch', torch.__version__)"
    "${LAUNCH[@]}" -m "contrastive_finetuning.$2" $($4 "$OUT/$3") ) > "$OUT/$3.log" 2>&1 || echo "FAILED $3" | tee -a "$OUT/status.txt"
}
run_new() {  # $1 module, $2 name, $3 args function
  ( source "$NEW_ENV/bin/activate"; cd "$NEW_REPO"
    python -c "import torch; print('torch', torch.__version__)"
    "${LAUNCH[@]}" -m "wildmatch.matcher_finetune.$1" $($3 "$OUT/$2") ) > "$OUT/$2.log" 2>&1 || echo "FAILED $2" | tee -a "$OUT/status.txt"
}
for step in "old loma train_loma_matches loma_old1 loma_args" "old loma train_loma_matches loma_old2 loma_args" \
            "new - train_loma_matches loma_new loma_args" \
            "old rdd train_by_lg_matches rdd_old1 rdd_args" "old rdd train_by_lg_matches rdd_old2 rdd_args" \
            "new - train_by_lg_matches rdd_new rdd_args"; do
  set -- $step; start=$(date +%s)
  if [[ $1 == old ]]; then run_old "$2" "$3" "$4" "$5"; else run_new "$3" "$4" "$5"; fi
  echo "$4 $(( $(date +%s) - start ))s" | tee -a "$OUT/status.txt"
done
"$NEW_ENV/bin/python" "$SPIKE/compare.py" "$OUT" | tee "$OUT/compare.txt"
