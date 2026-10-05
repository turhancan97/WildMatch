#!/usr/bin/env bash
set -euo pipefail

backend=${WILDLIFE_MINING_BACKEND:-rdd}
case "${backend}" in
  rdd|loma) ;;
  *) echo "WILDLIFE_MINING_BACKEND must be rdd or loma (got ${backend})" >&2; exit 1 ;;
esac

source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
if [[ "${backend}" == loma ]]; then conda activate loma; else conda activate rdd; fi

config=${WILDLIFE_CONFIG:-configs/wildlife/BelugaID.json}
protocol=${WILDLIFE_PROTOCOL:-strict}
eval "$(python -m scripts.wildlife_config --config "${config}" --shell)"
dataset_root=${WILDLIFE_VIEW_ROOT:-/shared/sets/datasets/vision/czechlynx/wildlife_processed/${WILDLIFE_DATASET_ID}/${protocol}}
index_root=${WILDLIFE_INDEX_ROOT:-outputs/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/indices}
if [[ -n "${WILDLIFE_MINING_REPORT:-}" ]]; then
  report=${WILDLIFE_MINING_REPORT}
elif [[ "${backend}" == loma ]]; then
  report=${index_root}/loma/strong-matches
elif [[ -n "${WILDLIFE_MINING_BACKEND+x}" ]]; then
  report=${index_root}/rdd/strong-matches
else
  # Preserve the original RDD default when the selector is not supplied.
  report=${index_root}/strong-matches
fi

if [[ "${backend}" == loma ]]; then
  cache_dir=${WILDLIFE_LOMA_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/loma-cache}
  weights=${WILDLIFE_MINING_WEIGHTS:-${LOMA_WEIGHTS:-/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt}}
  variant=${WILDLIFE_LOMA_VARIANT:-loma-b}
else
  cache_dir=${WILDLIFE_RDD_CACHE:-/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/${WILDLIFE_DATASET_ID}/rdd-cache}
  weights=${WILDLIFE_MINING_WEIGHTS:-${LG_WEIGHTS:-/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth}}
  variant=loma-b
fi
max_concurrent=${WILDLIFE_MAX_CONCURRENT:-30}
[[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]] || { echo "WILDLIFE_MAX_CONCURRENT must be a positive integer" >&2; exit 1; }

printf 'backend=%s\ndataset=%s\nprotocol=%s\ncache=%s\nweights=%s\nreport=%s\n' \
  "${backend}" "${WILDLIFE_DATASET_ID}" "${protocol}" "${cache_dir}" "${weights}" "${report}"
echo "Checking canonical view and ${backend} cache before submitting mining arrays..."
python - "${backend}" "${dataset_root}" "${cache_dir}" "${weights}" "${variant}" <<'PYCHECK'
import hashlib
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
        if backend == "loma":
            if not valid <= fields:
                invalid.append(f"{path}: {sorted(fields)}")
        elif not (valid <= fields or compact <= fields):
            invalid.append(f"{path}: {sorted(fields)}")
    except Exception as exc:
        invalid.append(f"{path}: {exc}")
if missing or invalid:
    raise SystemExit(f"cache invalid: {len(missing)} missing, {len(invalid)} malformed")
if backend == "loma":
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"LoMa cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {"backend": "loma", "variant": variant, "resize": 512, "num_keypoints": 512, "patch_size": 14}
    mismatch = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
    def fingerprint(path: Path) -> str:
        if path.is_dir():
            meta = path / "metadata.json"
            data = json.loads(meta.read_text()) if meta.is_file() else {}
            base = data.get("base_weights")
            if base:
                base_path = Path(base).expanduser()
                if not base_path.is_absolute(): base_path = path / base_path
                if base_path.is_file(): path = base_path
            else:
                candidates = [path / n for n in ("model.safetensors", "matcher.safetensors", "weights.pth")]
                path = next((item for item in candidates if item.is_file()), path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
        return digest.hexdigest()
    if manifest.get("weights_sha256"):
        actual = fingerprint(weights)
        if manifest["weights_sha256"] != actual:
            mismatch["weights_sha256"] = (manifest["weights_sha256"], actual)
    if mismatch:
        raise SystemExit("incompatible LoMa cache metadata: " + ", ".join(f"{k}={a!r} (wanted {b!r})" for k, (a, b) in mismatch.items()))
print(f"cache valid for {len(frames):,} frames")
PYCHECK

mkdir -p "$(dirname "${report}")" \
  "logs/wildlife-mine-${backend}-train" "logs/wildlife-mine-${backend}-val" "logs/wildlife-mine-${backend}-test"
job_ids=()
splits=(train test)
if [[ "${protocol}" == strict ]]; then splits=(train val test); fi
for split in "${splits[@]}"; do
  count=$(python - "${dataset_root}" "${split}" <<'PYCOUNT'
import sys
from pathlib import Path
from scripts.wildlife_dataset import list_collections
print(len(list_collections(Path(sys.argv[1]), sys.argv[2])))
PYCOUNT
)
  [[ "${count}" -gt 0 ]] || { echo "no ${split} collections found under ${dataset_root}" >&2; exit 1; }
  job_ids+=("$(sbatch --parsable \
    --array="0-$((count - 1))%${max_concurrent}" \
    --output="logs/wildlife-mine-${backend}-${split}/%A_%a.out" \
    --error="logs/wildlife-mine-${backend}-${split}/%A_%a.err" \
    slurm_scripts/wildlife_mine_task.sh "${WILDLIFE_DATASET_ID}" "${dataset_root}" "${cache_dir}" "${weights}" "${report}" "${split}" "${backend}" "${variant}")")
  echo "${backend} ${split} mining: ${job_ids[-1]} (${count} collections)"
done

dependency=$(IFS=:; echo "${job_ids[*]}")
aggregate=$(sbatch --parsable --dependency="afterok:${dependency}" \
  slurm_scripts/wildlife_aggregate.sh "${report}" "${dataset_root}" "${WILDLIFE_DATASET_ID}" "${protocol}" "${backend}" "${variant}" "${weights}" "${cache_dir}")
echo "aggregation: ${aggregate}"
echo "array concurrency: ${max_concurrent}"
