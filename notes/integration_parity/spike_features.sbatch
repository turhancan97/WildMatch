#!/usr/bin/env bash
#SBATCH --job-name=wm-spike-features
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%j.out
#SBATCH --error=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%j.err
# Integration parity L3: SalamanderID2025 feature caches (mining's LoMa cache builder, fine-tuning's
# RDD keypoint cache builder) from the old code in the old envs (twice) and the merged code in the
# wildmatch-integration env, on one GPU; compared with each other and with the paper's caches.
set -euo pipefail
L3=/shared/results/common/kargin/projects/wildmatch-integration/spike/l3
OUT=$L3/runs/${SLURM_JOB_ID}; mkdir -p "$OUT"
MINING=/home/kargin/Projects/repositories/rdd-parallel-benchmark
FT=/home/kargin/Projects/repositories/lynx-finetuning
NEW_REPO=/shared/results/common/kargin/projects/wildmatch-integration/wt-cli
NEW_ENV=/shared/results/common/kargin/projects/uv-environment/wildmatch-integration
VIEW=/shared/sets/datasets/vision/czechlynx/wildlife_processed/SalamanderID2025/legacy
LOMA_W=/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt
RDD_W=/home/kargin/Projects/repositories/rdd/weights/RDD-v2.pth
export HF_HOME=/shared/results/common/kargin/.cache/huggingface PYTHONHASHSEED=0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
git -C "$NEW_REPO" rev-parse HEAD > "$OUT/new_commit.txt"
loma_args() { echo --dataset_root $VIEW --cache_dir "$1" --weights $LOMA_W --variant loma-b --all_frames --splits train test \
  --resize_max 512 --num_keypoints 512 --batch_size 4 --num_workers 8 --num_shards 20 --shard 0; }
rdd_args() { echo --data_root $VIEW --cache_root "$1" --rdd_weights $RDD_W --splits train test --resize 512 --top_k 512 \
  --num_workers 8 --limit 64; }
old() {  # env repo module name argsfn
  ( source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh; conda activate "$1"; cd "$2"
    python -m "$3" $($5 "$OUT/$4") ) > "$OUT/$4.log" 2>&1 || echo "FAILED $4" | tee -a "$OUT/status.txt"
  echo "done $4" >> "$OUT/status.txt"
}
new() {  # module name argsfn
  ( source "$NEW_ENV/bin/activate"; cd "$NEW_REPO"; export PYTHONPATH="$NEW_REPO/src"; python -c "import wildmatch; print(wildmatch.__file__)"; python -m "$1" $($3 "$OUT/$2") ) > "$OUT/$2.log" 2>&1 || echo "FAILED $2" | tee -a "$OUT/status.txt"
  echo "done $2" >> "$OUT/status.txt"
}
old loma "$MINING" scripts.lynx_build_loma_cache loma_old1 loma_args
old loma "$MINING" scripts.lynx_build_loma_cache loma_old2 loma_args
new wildmatch.mining.lynx_build_loma_cache loma_new loma_args
old rdd "$FT" contrastive_finetuning.build_keypoint_cache rdd_old1 rdd_args
old rdd "$FT" contrastive_finetuning.build_keypoint_cache rdd_old2 rdd_args
new wildmatch.matcher_finetune.build_keypoint_cache rdd_new rdd_args
"$NEW_ENV/bin/python" "$L3/compare_features.py" "$OUT" | tee "$OUT/compare.txt"
