#!/bin/bash
set -euo pipefail

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

dataset_root=${CZECHLYNX_ROOT:-/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed}
cache_dir=${CZECHLYNX_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache}
rdd_weights=${RDD_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD-v2.pth}
lg_weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
dump_report=${CZECHLYNX_MINING_REPORT:-outputs/czechlynx-time-closed/legacy/strong-matches}
max_concurrent=${CZECHLYNX_MAX_CONCURRENT:-20}
train_log_dir=${CZECHLYNX_TRAIN_LOG_DIR:-logs/czechlynx-mine-train}
val_log_dir=${CZECHLYNX_VAL_LOG_DIR:-logs/czechlynx-mine-val}
test_log_dir=${CZECHLYNX_TEST_LOG_DIR:-logs/czechlynx-mine-test}

mkdir -p "$(dirname "${dump_report}")" "${train_log_dir}" "${val_log_dir}" "${test_log_dir}"
if ! [[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CZECHLYNX_MAX_CONCURRENT must be a positive integer, got: ${max_concurrent}" >&2
  exit 1
fi

echo "Validating CzechLynx RDD cache before submitting mining arrays..."
python - "${dataset_root}" "${cache_dir}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

dataset_root = Path(sys.argv[1])
cache_root = Path(sys.argv[2])
manifest_path = cache_root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing RDD cache manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text())
frames = []
for split in ("train", "val", "test"):
    split_root = dataset_root / split
    if not split_root.is_dir():
        raise SystemExit(f"missing dataset split: {split_root}")
    frames.extend(sorted(split_root.rglob("frame_*.jpg")))
if not frames:
    raise SystemExit(f"no frame files found under {dataset_root}")

manifest_count = manifest.get("n_frames_enumerated")
if manifest_count is not None and int(manifest_count) != len(frames):
    raise SystemExit(
        f"cache manifest covers {manifest_count} frames, but the dataset has {len(frames)}"
    )

descriptive = {"keypoints", "descriptors", "scores", "image_size"}
compact = {"k", "d", "hw"}
missing = []
invalid = []
for frame in frames:
    cache_path = cache_root / frame.relative_to(dataset_root).with_suffix(".npz")
    if not cache_path.is_file():
        missing.append(str(cache_path))
        continue
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            fields = set(archive.files)
    except Exception as exc:
        invalid.append(f"{cache_path}: {exc}")
        continue
    if not (descriptive.issubset(fields) or compact.issubset(fields)):
        invalid.append(f"{cache_path}: fields={sorted(fields)}")

if missing or invalid:
    print(f"missing cache files: {len(missing)}")
    print(f"invalid cache files: {len(invalid)}")
    for item in (missing + invalid)[:20]:
        print(f"  {item}")
    raise SystemExit(1)

print(
    f"RDD cache valid: {len(frames):,} frames; supports benchmark and compact k/d/hw schemas"
)
PY

n_train=$(find "${dataset_root}/train" -mindepth 2 -maxdepth 2 -type d | wc -l)
n_val=$(find "${dataset_root}/val" -mindepth 2 -maxdepth 2 -type d | wc -l)
n_test=$(find "${dataset_root}/test" -mindepth 2 -maxdepth 2 -type d | wc -l)
if [[ "${n_train}" -lt 1 || "${n_val}" -lt 1 || "${n_test}" -lt 1 ]]; then
  echo "canonical view is missing train/, val/, or test/ collections under ${dataset_root}" >&2
  exit 1
fi

train_job=$(sbatch --parsable \
  --array="0-$((n_train - 1))%${max_concurrent}" \
  --output="${train_log_dir}/czechlynx-mine-%A_%a.out" \
  --error="${train_log_dir}/czechlynx-mine-%A_%a.err" \
  slurm_scripts/czechlynx_mine_task.sh \
  "${dataset_root}" "${cache_dir}" "${rdd_weights}" "${lg_weights}" "${dump_report}" train)
val_job=$(sbatch --parsable \
  --array="0-$((n_val - 1))%${max_concurrent}" \
  --output="${val_log_dir}/czechlynx-mine-%A_%a.out" \
  --error="${val_log_dir}/czechlynx-mine-%A_%a.err" \
  slurm_scripts/czechlynx_mine_task.sh \
  "${dataset_root}" "${cache_dir}" "${rdd_weights}" "${lg_weights}" "${dump_report}" val)

test_job=$(sbatch --parsable \
  --array="0-$((n_test - 1))%${max_concurrent}" \
  --output="${test_log_dir}/czechlynx-mine-%A_%a.out" \
  --error="${test_log_dir}/czechlynx-mine-%A_%a.err" \
  slurm_scripts/czechlynx_mine_task.sh \
  "${dataset_root}" "${cache_dir}" "${rdd_weights}" "${lg_weights}" "${dump_report}" test)

aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}:${val_job}:${test_job}" \
  slurm_scripts/czechlynx_aggregate.sh "${dump_report}" "${dataset_root}")
echo "train mining: ${train_job}"
echo "validation mining: ${val_job}"
echo "test mining: ${test_job}"
echo "aggregation: ${aggregate_job}"
echo "array concurrency: ${max_concurrent} per split"
echo "train logs: ${train_log_dir}"
echo "validation logs: ${val_log_dir}"
echo "test logs: ${test_log_dir}"
