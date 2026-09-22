#!/usr/bin/env bash
# Emergency single-GPU copy of the LoMa matcher training for one CzechLynx few-shot view
# (default: fraction 1.0, seed 0), for when the 4-GPU job of pipeline_czechlynx.sh cannot get
# a whole rtx4090 node and would wait a day in the queue. One GPU is scheduled within minutes,
# so the run starts now and takes roughly four times as long (~6 h instead of ~1.5 h).
#
# The global batch stays the reference 32 (the trainer has no gradient accumulation, so it is
# one process x batch 32 instead of 4 x 8): the optimisation matches the other fractions, but
# a 4090 (24 GB) may run out of memory where 4 x 8 did not. FEWSHOT_GLOBAL_BATCH=8 falls back
# to a smaller batch -- a different optimisation, so the point stops being comparable with the
# rest of the curve; prefer waiting for the 4-GPU job over that.
#
# It writes to its own checkpoint directory (tag gpu1, so nothing collides with the 4-GPU run):
#   $FEWSHOT_ROOT/checkpoints/CzechLynx/legacy/<view>/loma-finetuned-gpu1/
# and its probe runs get the label custom-gpu1. The reporting treats that as a stand-in for the
# missing 4-GPU point only when it is named:
#   python scripts/fewshot_results.py --animal CzechLynx --k-sweep \
#     --fallback-checkpoint $FEWSHOT_ROOT/checkpoints/CzechLynx/legacy/frac1.0-seed0/loma-finetuned-gpu1
# Drop that argument once the 4-GPU checkpoint exists and the curve uses the regular
# loma-finetuned run again (the gpu1 runs stay in the CSV as their own series).
#
# usage (from the rdd-parallel-benchmark root):
#   bash slurm_scripts/fewshot/train_czechlynx_loma_1gpu.sh [fraction] [seed]
# environment: everything pipeline_czechlynx_components.sh accepts; this wrapper sets
#   FEWSHOT_TAG (gpu1), FEWSHOT_GLOBAL_BATCH (32) and FEWSHOT_CZ_COMPONENT_SBATCH
#   (1 GPU, 10 CPUs, 64G, 24 h on rtx4090_batch) unless they are already set.
set -euo pipefail
fraction=${1:-1.0}
seed=${2:-${FEWSHOT_SEED:-0}}
export FEWSHOT_TAG=${FEWSHOT_TAG-gpu1}
export FEWSHOT_GLOBAL_BATCH=${FEWSHOT_GLOBAL_BATCH:-32}
export FEWSHOT_CZ_COMPONENT_SBATCH=${FEWSHOT_CZ_COMPONENT_SBATCH:--p rtx4090_batch --qos=batch --gres=gpu:1 --cpus-per-task=10 --mem=64G --time=24:00:00 --exclude=${FEWSHOT_CZ_TRAIN_EXCLUDE:-c11}}
echo "== single-GPU LoMa training (tag ${FEWSHOT_TAG}, global batch ${FEWSHOT_GLOBAL_BATCH}): ${FEWSHOT_CZ_COMPONENT_SBATCH}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/pipeline_czechlynx_components.sh" loma matcher "${fraction}" "${seed}"
