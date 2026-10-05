#!/usr/bin/env bash
#SBATCH --job-name=fs-cz-smoke
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=c11
#SBATCH --output=logs/fewshot/%x-%j.out
#SBATCH --error=logs/fewshot/%x-%j.err
#
# One-GPU smoke test of a training component before the real jobs are queued: a few dozen
# index entries, one epoch, evaluation on, W&B off, checkpoints into a scratch directory
# that is deleted afterwards. It exercises exactly the paths that broke the first descriptor
# attempt — the pre-training pseudo-accuracy eval, the training step and the checkpoint
# write — in a couple of minutes instead of failing 40 s into a 4-GPU job.
#
#   sbatch slurm_scripts/fewshot/smoke_train_component.sh <rdd|loma> <component> [fraction] [entries]
#   bash   slurm_scripts/fewshot/smoke_train_component.sh rdd descriptor        # on a node with a GPU
#
# components: rdd -> lg, descriptor, lg+descriptor, rdd, lg+rdd
#             loma -> matcher, descriptor, descriptor+matcher
set -euo pipefail
source "${RDD_BENCHMARK_ROOT:-$PWD}/env.sh"
cd "${RDD_BENCHMARK_ROOT}"
backend=${1:?usage: $0 <rdd|loma> <component> [fraction] [entries]}
component=${2:?usage: $0 <rdd|loma> <component> [fraction] [entries]}
fraction=${3:-1.0}
entries=${4:-24}
seed=${FEWSHOT_SEED:-0}
dataset=CzechLynx
protocol=legacy
case "${backend}" in
  rdd) activate_conda_env "${CONDA_ENV_RDD}"; wrapper=train_czechlynx_rdd.sh; epoch_dir=epoch_00 ;;
  loma) activate_conda_env "${CONDA_ENV_LOMA}"; wrapper=train_czechlynx_loma.sh; epoch_dir=epoch_000 ;;
  *) echo "backend must be rdd or loma" >&2; exit 2 ;;
esac

view=$(python -c "from scripts.wildlife_fewshot import view_name; print(view_name(${fraction}, ${seed}))")
view_root=${FEWSHOT_ROOT}/views/${dataset}/${protocol}/${view}
index_root=${FEWSHOT_ROOT}/indices/${dataset}/${protocol}/${view}/${backend}
[[ -f "${index_root}/strong-matches_train_combined.json" ]] || { echo "no ${backend} indices under ${index_root}" >&2; exit 1; }

work=${TMPDIR:-/tmp}/fs-smoke-${backend}-${component//+/-}-${SLURM_JOB_ID:-$$}
mkdir -p "${work}/indices" "${work}/out"
trap 'rm -rf "${work}"' EXIT
python - "${index_root}" "${work}/indices" "${entries}" <<'PY'
import json, sys
from pathlib import Path
source, target, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for split in ("train", "test"):
    name = f"strong-matches_{split}_combined.json"
    entries = json.loads((source / name).read_text())
    (target / name).write_text(json.dumps(entries[:count]))
    print(f"{name}: {min(count, len(entries))} of {len(entries)} entries")
PY

echo "== smoke: ${backend} / ${component} on ${view} (${entries} entries, 1 epoch, 1 GPU)"
started=$(date +%s)
(export CZECHLYNX_SPLIT_PROTOCOL=legacy CZECHLYNX_ROOT=${view_root} CZECHLYNX_INDEX_ROOT=${work}/indices
 export CZECHLYNX_RDD_TRAIN_COMPONENT=${component} CZECHLYNX_LOMA_TRAIN_COMPONENT=${component}
 export CZECHLYNX_RDD_OUTPUT=${work}/out CZECHLYNX_LOMA_OUTPUT=${work}/out
 export CZECHLYNX_RDD_RUN_NAME=smoke CZECHLYNX_LOMA_RUN_NAME=smoke
 export CZECHLYNX_WANDB_PROJECT= CZECHLYNX_WANDB_MODE=disabled   # no run in the real projects
 export CZECHLYNX_EPOCHS=1 CZECHLYNX_EVAL_EVERY=1 CZECHLYNX_NUM_PROCESSES=1
 export CZECHLYNX_BATCH_SIZE=${FEWSHOT_SMOKE_BATCH:-2} CZECHLYNX_NUM_WORKERS=4
 cd "${LYNX_FINETUNING_ROOT}" && mkdir -p logs/czechlynx-rdd-ft logs/czechlynx-loma-ft \
   && bash "${LYNX_FINETUNING_ROOT}/slurm_scripts/${wrapper}")
[[ -f "${work}/out/${epoch_dir}/model.safetensors" ]] || { echo "SMOKE FAILED: no checkpoint in ${work}/out/${epoch_dir}" >&2; exit 1; }
echo "== smoke OK in $(( $(date +%s) - started ))s: ${backend}/${component} trains, evaluates and checkpoints"
