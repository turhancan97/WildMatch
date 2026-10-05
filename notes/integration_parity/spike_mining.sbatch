#!/usr/bin/env bash
#SBATCH --job-name=wm-spike-mining
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=c11,c15,c22
#SBATCH --output=/shared/results/common/kargin/projects/wildmatch-integration/spike/logs/%x-%j.out
# Integration parity L4: mine 24 seeded-random SalamanderID2025 train queries (LoMa and RDD-LightGlue)
# with the old code in the old envs and the merged code in the new env, both reading the paper's
# feature caches; compared per query with each other and with the paper's per-query files.
set -euo pipefail
L4=/shared/results/common/kargin/projects/wildmatch-integration/spike/l4
OUT=$L4/runs/${SLURM_JOB_ID}; mkdir -p "$OUT"
MINING=/home/kargin/Projects/repositories/rdd-parallel-benchmark
NEW_REPO=/shared/results/common/kargin/projects/wildmatch-integration/wt-cli
NEW_ENV=/shared/results/common/kargin/projects/uv-environment/wildmatch-integration
VIEW=/shared/sets/datasets/vision/czechlynx/wildlife_processed/SalamanderID2025/legacy
CK=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025
export HF_HOME=/shared/results/common/kargin/.cache/huggingface PYTHONHASHSEED=0
nvidia-smi --query-gpu=name --format=csv,noheader | tee "$OUT/gpu.txt"
git -C "$NEW_REPO" rev-parse HEAD > "$OUT/new_commit.txt"
args() {  # backend query_id prefix
  local w=/home/kargin/Projects/repositories/rdd/weights/RDD_lg-v2.pth
  [[ $1 == loma ]] && w=/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt
  echo --dataset_id SalamanderID2025 --dataset_root $VIEW --cache_dir $CK/$1-cache --lg_weights $w --backend $1 \
    --variant loma-b --split train --query_id $2 --frames_per_collection 20 --top_k_frames 5 --top_m 10 --dump_report $3
}
for backend in loma rdd; do
  env_name=rdd; [[ $backend == loma ]] && env_name=loma
  for q in $(cat $L4/query_ids.txt); do
    ( source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh; conda activate $env_name; cd $MINING
      python -m scripts.wildlife_mine $(args $backend $q "$OUT/old/$backend/strong-matches") ) >> "$OUT/old_$backend.log" 2>&1 || echo "FAILED old $backend $q" >> "$OUT/status.txt"
    ( source "$NEW_ENV/bin/activate"; cd "$NEW_REPO"; export PYTHONPATH="$NEW_REPO/src"
      python -m wildmatch.mining.wildlife_mine $(args $backend $q "$OUT/new/$backend/strong-matches") ) >> "$OUT/new_$backend.log" 2>&1 || echo "FAILED new $backend $q" >> "$OUT/status.txt"
  done
  echo "done $backend" >> "$OUT/status.txt"
done
"$NEW_ENV/bin/python" "$L4/compare_mining.py" "$OUT" | tee "$OUT/compare.txt"
