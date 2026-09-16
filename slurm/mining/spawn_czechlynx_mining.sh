#!/usr/bin/env bash
set -euo pipefail

backend=${CZECHLYNX_MINING_BACKEND:-rdd}
case "${backend}" in
  rdd|loma) ;;
  *) echo "CZECHLYNX_MINING_BACKEND must be rdd or loma (got ${backend})" >&2; exit 1 ;;
esac
protocol=${CZECHLYNX_SPLIT_PROTOCOL:-legacy}
case "${protocol}" in
  legacy|strict) ;;
  *) echo "CZECHLYNX_SPLIT_PROTOCOL must be legacy or strict (got ${protocol})" >&2; exit 1 ;;
esac
split_column=${CZECHLYNX_SPLIT_COLUMN:-split-time_closed}
case "${split_column}" in
  split-time_closed|split-time_open) ;;
  *) echo "CZECHLYNX_SPLIT_COLUMN must be split-time_closed or split-time_open (got ${split_column})" >&2; exit 1 ;;
esac
split_suffix=${split_column#split-}
experiment_slug="czechlynx-${split_suffix//_/-}"

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
if [[ "${backend}" == "loma" ]]; then
  conda activate loma
  cache_dir=${CZECHLYNX_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/${experiment_slug}/loma-b-cache}
  weights=${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}
  rdd_weights=${weights}
  variant=${CZECHLYNX_LOMA_VARIANT:-loma-b}
  default_report="outputs/${experiment_slug}/${protocol}/loma/strong-matches"
  log_prefix="czechlynx-mine-loma"
else
  conda activate rdd
  cache_dir=${CZECHLYNX_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/${experiment_slug}/rdd-cache}
  weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
  rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
  variant=loma-b
  default_report="outputs/${experiment_slug}/${protocol}/rdd/strong-matches"
  log_prefix="czechlynx-mine"
fi

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_${split_suffix}}
dump_report=${CZECHLYNX_MINING_REPORT:-${default_report}}
max_concurrent=${CZECHLYNX_MAX_CONCURRENT:-20}
train_log_dir=${CZECHLYNX_TRAIN_LOG_DIR:-logs/${log_prefix}-train}
val_log_dir=${CZECHLYNX_VAL_LOG_DIR:-logs/${log_prefix}-val}
test_log_dir=${CZECHLYNX_TEST_LOG_DIR:-logs/${log_prefix}-test}

mkdir -p "$(dirname "${dump_report}")" "${train_log_dir}" "${val_log_dir}" "${test_log_dir}"
[[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]] || { echo "CZECHLYNX_MAX_CONCURRENT must be a positive integer" >&2; exit 1; }

printf 'backend=%s\nsplit_column=%s\nprotocol=%s\ndataset_root=%s\ncache=%s\nweights=%s\nreport=%s\n' \
  "${backend}" "${split_column}" "${protocol}" "${dataset_root}" "${cache_dir}" "${weights}" "${dump_report}"
echo "Validating CzechLynx ${backend} cache before submitting mining arrays..."
python - "${backend}" "${dataset_root}" "${cache_dir}" "${weights}" "${variant}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

backend, root_text, cache_text, weights_text, variant = sys.argv[1:]
root, cache, weights = map(Path, (root_text, cache_text, weights_text))
if not root.is_dir():
    raise SystemExit(f"missing canonical dataset root: {root}")
if not weights.exists():
    raise SystemExit(f"missing {backend} checkpoint: {weights}")
frames = []
for split in ("train", "val", "test"):
    split_root = root / split
    if not split_root.is_dir():
        raise SystemExit(f"missing dataset split: {split_root}")
    frames.extend(sorted(split_root.rglob("frame_*.jpg")))
if not frames:
    raise SystemExit(f"no frame files found under {root}")
if backend == "loma":
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing LoMa cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for key, expected in (("backend", "loma"), ("variant", variant), ("resize", 512), ("num_keypoints", 512), ("patch_size", 14)):
        if manifest.get(key) != expected:
            raise SystemExit(f"incompatible LoMa cache metadata: {key}={manifest.get(key)!r}, expected {expected!r}")
valid = {"keypoints", "descriptors", "scores", "image_size"}
compact = {"k", "d", "hw"}
missing, invalid = [], []
for frame in frames:
    path = cache / frame.relative_to(root).with_suffix(".npz")
    if not path.is_file():
        missing.append(str(path)); continue
    try:
        with np.load(path, allow_pickle=False) as data:
            fields = set(data.files)
        if not valid <= fields and not (backend == "rdd" and compact <= fields):
            invalid.append(f"{path}: {sorted(fields)}")
    except Exception as exc:
        invalid.append(f"{path}: {exc}")
if missing or invalid:
    raise SystemExit(f"cache invalid: {len(missing)} missing, {len(invalid)} malformed")
print(f"cache valid for {len(frames):,} frames")
PY

n_train=$(find "${dataset_root}/train" -mindepth 2 -maxdepth 2 -type d | wc -l)
n_val=$(find "${dataset_root}/val" -mindepth 2 -maxdepth 2 -type d | wc -l)
n_test=$(find "${dataset_root}/test" -mindepth 2 -maxdepth 2 -type d | wc -l)
if [[ "${n_train}" -lt 1 || "${n_val}" -lt 1 || "${n_test}" -lt 1 ]]; then
  echo "canonical view is missing train/, val/, or test/ collections under ${dataset_root}" >&2
  exit 1
fi

submit_array() {
  local split=$1 log_dir=$2 count=$3
  sbatch --parsable \
    --array="0-$((count - 1))%${max_concurrent}" \
    --output="${log_dir}/czechlynx-mine-%A_%a.out" \
    --error="${log_dir}/czechlynx-mine-%A_%a.err" \
    slurm_scripts/czechlynx_mine_task.sh \
    "${dataset_root}" "${cache_dir}" "${rdd_weights}" "${weights}" "${dump_report}" "${split}" "${backend}" "${variant}"
}

train_job=$(submit_array train "${train_log_dir}" "${n_train}")
val_job=$(submit_array val "${val_log_dir}" "${n_val}")
test_job=$(submit_array test "${test_log_dir}" "${n_test}")
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}:${val_job}:${test_job}" \
  slurm_scripts/czechlynx_aggregate.sh "${dump_report}" "${dataset_root}")

echo "train mining: ${train_job} (${n_train} collections)"
echo "validation mining: ${val_job} (${n_val} collections)"
echo "test mining: ${test_job} (${n_test} collections)"
echo "aggregation: ${aggregate_job}"
echo "array concurrency: ${max_concurrent} per split"
