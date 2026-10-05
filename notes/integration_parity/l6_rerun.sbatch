#!/usr/bin/env bash
#SBATCH --job-name=wm-l6
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=125G
#SBATCH --time=06:00:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --array=14,16
#SBATCH --output=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%A_%a.out
# Integration parity L6: each published checkpoint (conf/weights.yaml) evaluated at k=50 by the main
# environment with main's code and by the integration environment with the integration branch, on the
# same GPU, each with its own fresh feature cache; score matrices compared bit for bit.
set -euo pipefail
L6=/shared/results/common/kargin/projects/wildmatch-integration/spike/l6
IFS=$'\t' read -r key matcher ckpt animal < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $L6/tasks.tsv)
OUT=$L6/runs/${SLURM_ARRAY_JOB_ID}/${key}__${matcher}; mkdir -p "$OUT"
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$OUT/gpu.txt"
extra=(); [[ $matcher == loma ]] && extra=(benchmark.methods.vismatch.loma_arch=LoMa-B)
evaluate() {  # name python pythonpath
  ( cd "$OUT"; export PYTHONPATH="$3"; export HF_HOME=/shared/results/common/kargin/.cache/huggingface
    "$2" -m wildmatch evaluate paths=gmum dataset=$key paths.cache_root=$OUT/cache-$1 \
      output.experiment_root=$OUT/experiments-$1 wandb.enabled=false \
      benchmark.method=vismatch benchmark.candidate_k=50 benchmark.methods.vismatch.matcher=$matcher "${extra[@]}" \
      benchmark.methods.vismatch.checkpoint_source=custom benchmark.methods.vismatch.checkpoint_path=$ckpt \
      benchmark.methods.vismatch.checkpoint_components=matcher_only \
      benchmark.methods.vismatch.checkpoint_owner=$animal benchmark.methods.vismatch.evaluation_animal=$animal
  ) > "$OUT/$1.log" 2>&1 || echo "FAILED $1" >> "$OUT/status.txt"
}
evaluate main /shared/results/common/kargin/projects/uv-environment/wildmatch/bin/python /shared/results/common/kargin/projects/wildmatch-integration/wt-premerge/src
git -C /shared/results/common/kargin/projects/wildmatch-integration/wt-premerge rev-parse HEAD > "$OUT/main_commit.txt"
evaluate new /shared/results/common/kargin/projects/uv-environment/wildmatch-integration/bin/python \
  /shared/results/common/kargin/projects/wildmatch-integration/wt-mine/src
git -C /shared/results/common/kargin/projects/wildmatch-integration/wt-mine rev-parse HEAD > "$OUT/new_commit.txt"
/shared/results/common/kargin/projects/uv-environment/wildmatch-integration/bin/python $L6/compare_l6.py "$OUT" | tee "$OUT/compare.txt"
