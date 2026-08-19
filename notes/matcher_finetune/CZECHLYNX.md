# CzechLynx time-closed workflow

The integration uses masked real CzechLynx images and the metadata
`split-time_closed` protocol. The converter creates a symlink view with
`train`, `val`, and `test` directories. The strict protocol uses a
deterministic 20% encounter holdout for validation, while the legacy protocol
uses the metadata test split for training-time validation to match the
previous Lynx workflow.

## Run order

From the benchmark repository:

```bash
sbatch slurm_scripts/prepare_czechlynx.sh
```

Then build the RDD cache and mine the shared train/validation indices:

```bash
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/build_czechlynx_rdd_cache.sh
bash slurm_scripts/spawn_czechlynx_mining.sh
```

After aggregation, run RDD first and LoMa second:

```bash
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_rdd.sh
```

The RDD SLURM entry point requests two GPUs and uses synchronized
shape-bucketed global batches. `--batch_size 8` remains per GPU, so the
effective batch size is 16. Triplets are grouped by their cached `(H, W)`
signature before Accelerate splits each global batch identically across ranks;
this handles mixed dimensions such as `480x512` and `512x512` safely.

```bash
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/build_czechlynx_loma_cache.sh
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_loma.sh
```

The benchmark evaluator is launched with `CZECHLYNX_BACKEND=rdd` or
`CZECHLYNX_BACKEND=loma`, and writes separate full-gallery or top-15 JSON
reports. Existing original-Lynx scripts and artifact directories are not
changed.

## Split protocols

CzechLynx supports two fine-tuning protocols through
`CZECHLYNX_SPLIT_PROTOCOL`:

```text
legacy (default): train → fine-tuning; test → validation and final reporting
strict:           train → fine-tuning; val → validation; test → final reporting
```

The legacy mode is intentionally compatible with the previous Lynx workflow.
Because the test-derived index is used for checkpoint selection during
training, its final metrics are not an untouched test estimate. The strict
mode preserves the encounter-level 20% validation holdout for clean model
selection.

The mining command creates all three indices and keeps separate logs for each
query split:

```bash
bash /home/kargin/Projects/repositories/rdd-parallel-benchmark/slurm_scripts/spawn_czechlynx_mining.sh
```

The default mining report directory is
`outputs/czechlynx-time-closed/legacy/strong-matches`, and the training entry
points default to the corresponding legacy index root. The default protocol is
also `legacy`, so no environment variables are required for the standard
legacy-compatible run.

It produces:

```text
outputs/czechlynx-time-closed/strong-matches_train_combined.json
outputs/czechlynx-time-closed/strong-matches_val_combined.json
outputs/czechlynx-time-closed/strong-matches_test_combined.json
```

Mining uses 20 frames per collection, `top_k_frames=5`, `top_m=10`, and the
pretrained RDD/LightGlue weights. Test queries are scored against the train
gallery, matching the previous Lynx protocol.

### Optional combined train+validation index

For a larger training set, the existing train and validation indices can be
combined into a separate file. This does not modify the original indices:

```bash
cd /home/kargin/Projects/repositories/rdd-parallel-benchmark

python - <<PY
import json
from pathlib import Path

root = Path("outputs/czechlynx-time-closed/legacy")
train = json.loads((root / "strong-matches_train_combined.json").read_text())
val = json.loads((root / "strong-matches_val_combined.json").read_text())

train_queries = {item["query_frame"] for item in train}
val_queries = {item["query_frame"] for item in val}
assert not train_queries & val_queries, "Train/validation query overlap detected"

output = root / "strong-matches_trainval_combined.json"
output.write_text(json.dumps(train + val, indent=2))
print(f"Wrote {output} with {len(train) + len(val)} entries")
PY
```

The resulting index contains approximately 4,826 entries (3,009 train and
1,817 validation entries). Use it explicitly for RDD or LoMa fine-tuning:

```bash
CZECHLYNX_TRAIN_INDEX=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/czechlynx-time-closed/legacy/strong-matches_trainval_combined.json \
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_rdd.sh
```

For LoMa, use the same `CZECHLYNX_TRAIN_INDEX` override with
`train_czechlynx_loma.sh`. The legacy validation index remains
`strong-matches_test_combined.json`; the combined file is used only as the
training index.

This is a direct concatenation of already mined entries. The validation
entries were originally mined against the train gallery, so this does not
create a newly mined train+validation gallery. A true combined-gallery
experiment requires re-mining with both the train and validation collections
available as gallery candidates.

### Legacy RDD training

Legacy mode is the default, but it can be set explicitly:

```bash
CZECHLYNX_SPLIT_PROTOCOL=legacy \
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_rdd.sh
```

This uses `strong-matches_test_combined.json` as `val_index` and writes to the
legacy-specific checkpoint directory:

```text
/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned-legacy
```

### Strict RDD training

```bash
CZECHLYNX_SPLIT_PROTOCOL=strict \
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_rdd.sh
```

This uses `strong-matches_val_combined.json` as `val_index` and writes to
`rdd-finetuned-strict`.

### Legacy and strict LoMa training

Use the same setting with the LoMa entry point:

```bash
CZECHLYNX_SPLIT_PROTOCOL=legacy \
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_loma.sh

CZECHLYNX_SPLIT_PROTOCOL=strict \
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_loma.sh
```

RDD and LoMa use the same aggregated train/test/validation indices by default.
Each run prints its selected protocol and index paths and writes
`czechlynx_protocol.json` in its output directory.

## RDD test-set evaluation

The following commands evaluate the CzechLynx metadata test split using the
RDD cache. Run them from `rdd-parallel-benchmark`.

The pretrained LightGlue checkpoint is:

```bash
/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth
```

The completed CzechLynx run stores fine-tuned checkpoints under:

```bash
/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/
```

For example, `epoch_299/model.safetensors` is the final checkpoint. Select a
different epoch if validation identifies a better checkpoint.

### Pretrained full-gallery evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth \
CZECHLYNX_MODE=full \
CZECHLYNX_OUTPUT=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/czechlynx-time-closed/pretrained-full.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Fine-tuned full-gallery evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/epoch_299/model.safetensors \
CZECHLYNX_MODE=full \
CZECHLYNX_OUTPUT=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/czechlynx-time-closed/epoch299-full.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Pretrained top-15 evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth \
CZECHLYNX_PRESELECT_WEIGHTS=/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth \
CZECHLYNX_MODE=top15 \
CZECHLYNX_OUTPUT=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/czechlynx-time-closed/pretrained-top15.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

### Fine-tuned top-15 evaluation

```bash
CZECHLYNX_BACKEND=rdd \
CZECHLYNX_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-cache \
CZECHLYNX_WEIGHTS=/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-closed/rdd-finetuned/epoch_299/model.safetensors \
CZECHLYNX_PRESELECT_WEIGHTS=/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth \
CZECHLYNX_MODE=top15 \
CZECHLYNX_OUTPUT=/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/czechlynx-time-closed/epoch299-top15.json \
sbatch slurm_scripts/czechlynx_evaluate.sh
```

In top-15 mode, the pretrained checkpoint selects the strongest query frame
and 15 diverse gallery frames. The target checkpoint then reranks those 15
frames. This makes the pretrained and fine-tuned top-15 results directly
comparable.
