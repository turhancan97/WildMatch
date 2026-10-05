#!/usr/bin/env bash
# Queue the complete few-shot pipeline (pipeline_dataset.sh) for a list of datasets.
#
# Default order: datasets on which cosine retrieval is below 90% top-1 in the reference
# results (lynx-main.pdf, k=50) come first, then the saturated ones; inside each group the
# cheapest dataset first (mining pairs + fine-tuning time). Jobs beyond the per-user limits
# of the partition simply wait in the queue, so the order is also the execution order.
#
#   dataset            cosine  frames  train ids  mining Mpairs  train min (4x4090)
#   NyalaData           27.8    1942     237         2.8            37
#   SeaStarReID2023     78.0    2187      95         3.1            36
#   HyenaID2022         70.8    3129     256         5.9            55
#   LeopardID2022       62.4    6806     430         8.9            53
#   BelugaID            52.5    5902     788        20.0            85
#   WhaleSharkID        49.3    7693     543        27.7           101
#   GiraffeZebraID      75.2    6898    2051        36.9           117
#   ZindiTurtleRecall   72.1   12803    2265       107.8           243
#   -- cosine >= 90% --
#   CowDataset          96.6    1485      13         0.1             7
#   StripeSpotter       98.2     820      45         0.4            15
#   Giraffes            97.0    1368     178         1.5            33
#   ATRW                96.8    5415     182        10.1            55
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/spawn_datasets.sh                       # the default list below
#   bash slurm_scripts/fewshot/spawn_datasets.sh NyalaData HyenaID2022  # a subset, in this order
#   FEWSHOT_DATASETS="SeaStarReID2023 LeopardID2022" bash slurm_scripts/fewshot/spawn_datasets.sh
# Everything pipeline_dataset.sh understands (FEWSHOT_FRACTIONS, FEWSHOT_QOS, FEWSHOT_TIME,
# PROBE_SBATCH_ARGS, ...) is passed through the environment.
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"

HARD_DATASETS="NyalaData SeaStarReID2023 HyenaID2022 LeopardID2022 BelugaID WhaleSharkID GiraffeZebraID ZindiTurtleRecall"
EASY_DATASETS="CowDataset StripeSpotter Giraffes ATRW"
if (( $# > 0 )); then
  datasets="$*"
else
  datasets=${FEWSHOT_DATASETS:-${HARD_DATASETS} ${EASY_DATASETS}}
fi

for dataset in ${datasets}; do
  config=configs/wildlife/${dataset}.json
  [[ -f "${config}" ]] || { echo "no configuration for ${dataset}: ${config}" >&2; exit 1; }
done
echo "datasets (in queue order): ${datasets}"
for dataset in ${datasets}; do
  echo
  echo "################ ${dataset}"
  bash slurm_scripts/fewshot/pipeline_dataset.sh "configs/wildlife/${dataset}.json" "${FEWSHOT_SEED:-0}"
done
