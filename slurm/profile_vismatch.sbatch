#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --exclude=c22,c11,c15
#SBATCH --job-name=profile_vismatch
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log
# Vismatch cost profile (2026-10-07): SalamanderID2025, k=50, default LoMa and RDD-LightGlue,
# each run twice on a fresh feature cache (cold, then warm). Everything goes under
# experiments/profile/job-$SLURM_JOB_ID, never experiments/probe/ or reports/runs.csv.
set -euo pipefail
nvidia-smi -L
WILDMATCH_ENV="${WILDMATCH_ENV:-${UV_ENV_ROOT:-/shared/results/common/kargin/projects/uv-environment}/wildmatch}"
source "${WILDMATCH_ENV}/bin/activate"
OUT="experiments/profile/job-${SLURM_JOB_ID}"
mkdir -p "$OUT"
for matcher in loma rdd-lightglue; do
  for pass in cold warm; do
    echo "=== ${matcher} ${pass} ==="
    extra=()
    [ "$matcher" = loma ] && extra+=(benchmark.methods.vismatch.loma_arch=LoMa-B)
    wildmatch evaluate paths=gmum dataset=salamander \
      benchmark.method=vismatch \
      benchmark.methods.vismatch.matcher=${matcher} \
      benchmark.methods.vismatch.checkpoint_source=default \
      benchmark.candidate_k=50 \
      benchmark.cache.dir=${OUT}/cache/features \
      benchmark.methods.vismatch.cache_dir=${OUT}/cache/vismatch-${matcher} \
      output.experiment_root=${OUT}/experiments \
      output.run_dir=${OUT}/benchmark_runs \
      output.csv_path=${OUT}/benchmark_results.csv \
      reporting.index_path=${OUT}/runs.csv \
      visualization.enabled=false \
      wandb.enabled=false \
      "${extra[@]}"
  done
done
