#!/usr/bin/env bash
# pipeline_czechlynx.sh with the mining arrays on rtx4090_batch (QOS batch) instead of the
# backend default (rdd: rtx4090_batch already; loma: dgxh100 quick). Use it when the H100
# quick queue is busy or when its MaxSubmitPU=100 cap (array elements count one by one)
# rejects the 2 x 16 workers of a fraction — what killed the LoMa 1.0 chain on 22.09.
#
# Only the mining resources change; views, aggregation, training (rdd: dgxh100, loma:
# rtx4090 4 GPUs), the probe chain and the skip-what-exists logic are pipeline_czechlynx.sh's,
# so this can be run after a partial pipeline_czechlynx.sh run and only fills in the gaps.
# Workers: 1 GPU, 8 CPUs, 64G (the 1.0 train gallery is ~15k LoMa frames x 0.5 MB in RAM),
# 24 h, c11 excluded. RDD on 4090 mined the 1.0 view in <= 73 min (train) / 122 min (test).
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/pipeline_czechlynx_rtx4090.sh [seed]                  # RDD-LightGlue
#   FEWSHOT_BACKEND=loma bash slurm_scripts/fewshot/pipeline_czechlynx_rtx4090.sh   # LoMa-B
# environment: everything pipeline_czechlynx.sh accepts; FEWSHOT_CZ_MINE_SBATCH is set here
#   unless already given, FEWSHOT_CZ_MINE_EXCLUDE (c11) lists the nodes to avoid.
set -euo pipefail
export FEWSHOT_CZ_MINE_SBATCH=${FEWSHOT_CZ_MINE_SBATCH:--p rtx4090_batch --qos=batch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --exclude=${FEWSHOT_CZ_MINE_EXCLUDE:-c11}}
echo "== mining arrays on: ${FEWSHOT_CZ_MINE_SBATCH}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/pipeline_czechlynx.sh" "$@"
