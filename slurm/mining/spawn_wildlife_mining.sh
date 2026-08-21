#!/usr/bin/env bash
set -euo pipefail

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
conda activate rdd

config=${WILDLIFE_CONFIG:-configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
cache_dir=${WILDLIFE_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-cache}
lg_weights=${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}
report=${WILDLIFE_MINING_REPORT:-outputs/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/indices/strong-matches}
max_concurrent=${WILDLIFE_MAX_CONCURRENT:-30}

if ! [[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WILDLIFE_MAX_CONCURRENT must be a positive integer" >&2
  exit 1
fi

echo "Checking canonical view and RDD cache before submitting mining arrays..."
python - "${dataset_root}" "${cache_dir}" <<'PY'
import sys
from pathlib import Path
import numpy as np

root, cache = map(Path, sys.argv[1:])
if not root.is_dir():
    raise SystemExit(f"missing canonical dataset root: {root}")
frames = sorted(root.rglob("frame_*.jpg"))
if not frames:
    raise SystemExit(f"no canonical frames found under {root}")
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
        if not (valid <= fields or compact <= fields):
            invalid.append(f"{path}: {sorted(fields)}")
    except Exception as exc:
        invalid.append(f"{path}: {exc}")
if missing or invalid:
    raise SystemExit(f"cache invalid: {len(missing)} missing, {len(invalid)} malformed")
print(f"cache valid for {len(frames):,} frames")
PY

mkdir -p "$(dirname "${report}")" "logs/wildlife-mine-train" "logs/wildlife-mine-val" "logs/wildlife-mine-test"
job_ids=()
splits=(train test)
if [[ "${protocol}" == strict ]]; then
  splits=(train val test)
fi
for split in "${splits[@]}"; do
  count=$(python - "${dataset_root}" "${split}" <<'PY'
import sys
from pathlib import Path
from scripts.wildlife_dataset import list_collections
print(len(list_collections(Path(sys.argv[1]), sys.argv[2])))
PY
)
  if [[ "${count}" -lt 1 ]]; then
    echo "no ${split} collections found under ${dataset_root}" >&2
    exit 1
  fi
  job_ids+=("$(sbatch --parsable \
    --array="0-$((count - 1))%${max_concurrent}" \
    --output="logs/wildlife-mine-${split}/%A_%a.out" \
    --error="logs/wildlife-mine-${split}/%A_%a.err" \
    slurm_scripts/wildlife_mine_task.sh "${WILDLIFE_DATASET_ID}" "${dataset_root}" "${cache_dir}" "${lg_weights}" "${report}" "${split}")")
  echo "${split} mining: ${job_ids[-1]} (${count} collections)"
done

dependency=$(IFS=:; echo "${job_ids[*]}")
aggregate=$(sbatch --parsable --dependency="afterok:${dependency}" \
  slurm_scripts/wildlife_aggregate.sh "${report}" "${dataset_root}" "${WILDLIFE_DATASET_ID}" "${protocol}")
echo "aggregation: ${aggregate}"
echo "array concurrency: ${max_concurrent}"
