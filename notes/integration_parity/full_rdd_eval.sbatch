#!/usr/bin/env bash
#SBATCH --job-name=wm-full-rdd-eval
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=/shared/results/common/kargin/projects/wildmatch-integration/full_rdd/%x-%j.out
# D3 outcome check: the paper's SalamanderID2025 RDD-LightGlue checkpoint and the one retrained with
# wildmatch finetune-matcher under torch 2.8 (job 525382), evaluated the same way on one GPU.
set -euo pipefail
FULL=/shared/results/common/kargin/projects/wildmatch-integration/full_rdd
REPO=/shared/results/common/kargin/projects/wildmatch-integration/wt-cli
source /shared/results/common/kargin/projects/uv-environment/wildmatch-integration/bin/activate
cd "$REPO"; export PYTHONPATH="$REPO/src"
nvidia-smi --query-gpu=name --format=csv,noheader
PAPER=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-finetuned/legacy-rdd-mined/epoch_299/model.safetensors
NEW=$FULL/salamander-rdd-mined-legacy/epoch_299/model.safetensors
sha256sum "$PAPER" "$NEW"
for k in 50 250; do for name in paper new; do
  ckpt=$PAPER; [[ $name == new ]] && ckpt=$NEW
  echo "=== $name k=$k"
  wildmatch evaluate paths=gmum dataset=salamander output.experiment_root=$FULL/experiments/$name \
    benchmark.method=vismatch benchmark.candidate_k=$k benchmark.methods.vismatch.matcher=rdd-lightglue \
    benchmark.methods.vismatch.checkpoint_source=custom benchmark.methods.vismatch.checkpoint_path=$ckpt \
    benchmark.methods.vismatch.checkpoint_components=matcher_only \
    benchmark.methods.vismatch.checkpoint_owner=SalamanderID2025 benchmark.methods.vismatch.evaluation_animal=SalamanderID2025 \
    wandb.enabled=false 2>&1 | tail -40
done; done
