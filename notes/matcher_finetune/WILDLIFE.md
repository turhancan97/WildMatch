# Generic WildlifeReID-10k workflow

The generic pipeline trains and evaluates one animal dataset at a time. The
supported configurations are in
`rdd-parallel-benchmark/configs/wildlife/`:

```text
BelugaID+
NyalaData+
WhaleSharkID+
ZindiTurtleRecall+ (labeled rows with a resolved masked image)
AmvrakikosTurtles
ATRW+
CowDataset+
Giraffes+
GiraffeZebraID+
HyenaID2022+
LeopardID2022+
ReunionTurtles
SeaStarReID2023+
StripeSpotter+
ZakynthosTurtles
SalamanderID2025 (not part of WildlifeReID-10k; see below)
```

NDD20 is intentionally not configured: its masked paths and identity labels
are not usable yet.

The 11 newly added datasets use metadata from
`metadata_mdsplit_no_background/metadata_<dataset>.csv`, the shared
`masked_images` tree, and the `identity`, `path`, and `split` columns. The
preparation step never falls back to unmasked images. Rows with an empty or
`unknown` identity, invalid split, or missing masked target are excluded and
recorded in the generated manifest; metadata/schema errors or a dataset with
no usable records stop preparation.

No encounter or session field is currently available in these metadata files,
so identity is used as the collection fallback. This limitation is recorded
by each configuration's `collection_rule: identity` setting and in the
generated experiment metadata. It should be considered when interpreting
collection-level metrics.

### SalamanderID2025

SalamanderID2025 (1,384 images, 584 salamanders, a time-closed database/query
split) is a separate dataset that runs through the same workflow via
`configs/wildlife/SalamanderID2025.json`. It differs from the WildlifeReID-10k
configurations in three ways:

- Root and metadata: `/shared/sets/datasets/vision/czechlynx/SalamanderID2025`
  with `split_time_closed_no_background.csv`. Its images were background-removed
  with SAM3 (`explainable_individual_reidentification/scripts/segment_with_sam3.py`,
  prompt "Salamander", all instances merged) into `masked_images/`.
- Split: the original `split` column holds `database`/`query`. The preparer
  writes the split value as the view folder name and mining/training expect
  `train/` and `test/`, so the config reads the added `split_train_test` column
  (database -> train, query -> test). Use `WILDLIFE_PROTOCOL=legacy`.
- Pairs: both matchers are mined for, and each matcher is fine-tuned on its own
  pairs (LoMa on LoMa-mined, RDD on RDD-mined with `WILDLIFE_MINING_BACKEND=rdd`),
  unlike the other datasets, whose RDD runs use LoMa-mined pairs. Outputs are
  written to `rdd-finetuned/legacy-rdd-mined` and `loma-finetuned/legacy-loma-mined`
  (set `WILDLIFE_RDD_OUTPUT` / `WILDLIFE_LOMA_OUTPUT`), which the probe launcher's
  Salamander profile expects.

Outputs and checkpoints still live under the `wildlife-reid-10k/SalamanderID2025`
folders because that prefix is fixed in the scripts; the name is only a path.
372 of the 584 database identities have a single image and contribute no
positive pairs, so the effective training set is small.

## 1. Select a dataset and prepare its canonical view

Run these commands from `rdd-parallel-benchmark`:

```bash
export WILDLIFE_CONFIG=/home/kargin/Projects/repositories/rdd-parallel-benchmark/configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/prepare_wildlife.sh
```

The view is symlink-based and is written to:

```text
/shared/sets/datasets/vision/czechlynx/wildlife_processed/<dataset>/<protocol>/
```

`strict` uses official train images with a deterministic 20% validation holdout
per identity, while official test remains untouched. `legacy` uses official
train for fine-tuning and official test for both validation and final reporting.

## 2. Build caches and mine shared indices

After preparation, build the backend caches from `lynx-finetuning`:

```bash
export WILDLIFE_CONFIG=/home/kargin/Projects/repositories/rdd-parallel-benchmark/configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/build_wildlife_rdd_cache.sh
sbatch slurm_scripts/build_wildlife_loma_cache.sh
```

Mine the training pairs with the backend you want to fine-tune. RDD remains
the default if `WILDLIFE_MINING_BACKEND` is unset:

```bash
cd /home/kargin/Projects/repositories/rdd-parallel-benchmark
export WILDLIFE_CONFIG=/home/kargin/Projects/repositories/rdd-parallel-benchmark/configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
export WILDLIFE_MINING_BACKEND=rdd
bash slurm_scripts/spawn_wildlife_mining.sh
```

For LoMa, use its compatible cache and pretrained checkpoint:

```bash
export WILDLIFE_MINING_BACKEND=loma
export WILDLIFE_LOMA_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/BelugaID/loma-cache
export LOMA_WEIGHTS=/shared/sets/datasets/confidential/lynx/checkpoints/loma/loma_B.pt
bash slurm_scripts/spawn_wildlife_mining.sh
```

The defaults are 20 frames per collection, `top_k_frames=5`, `top_m=10`, and
array concurrency 30. Explicit backend selection stores indices under
`outputs/wildlife-reid-10k/<dataset>/indices/rdd/` or `loma/`; the historical
RDD path remains available when no selector is supplied. Each combined index
keeps the trainer-compatible JSON list and has a `.metadata.json` sidecar with
the backend, checkpoint, cache, protocol, and mining settings. RDD and LoMa
indices are intentionally separate. In legacy mode, the validation combined
file is an explicit alias of the test combined file; no second test mining pass
is performed.

## 3. Fine-tune

```bash
cd /home/kargin/Projects/repositories/lynx-finetuning
export WILDLIFE_CONFIG=/home/kargin/Projects/repositories/rdd-parallel-benchmark/configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/train_wildlife_rdd.sh
sbatch slurm_scripts/train_wildlife_loma.sh
```

The RDD wrapper reads the index of the pair source named by
`WILDLIFE_MINING_BACKEND` (default `loma`, i.e. `indices/loma/`) and fails if
it is missing. Until 2026-09-29 it silently fell back to RDD-mined pairs when
no LoMa index existed, so NyalaData and WhaleSharkID were trained on a
different pair source than the other datasets; mine LoMa pairs for a dataset
before training RDD on it. An output directory that already holds `epoch_*`
checkpoints is refused unless `WILDLIFE_RDD_RESUME=auto` (or an epoch
directory) continues it. Validation is
`strong-matches_val_combined.json` in strict mode and
`strong-matches_test_combined.json` in legacy mode. Checkpoints and W&B runs
are isolated by dataset, backend, and protocol. Unless overridden with
`WILDLIFE_WANDB_PROJECT`, the default projects are
`wildlife-reid-rdd-<dataset>-<protocol>` and
`wildlife-reid-loma-<dataset>-<protocol>`.

To run another dataset, replace `BelugaID.json` in the commands above with
one of the configuration files listed at the beginning of this document. The
cache, index, checkpoint, and evaluation paths are derived from that dataset
identifier.

### Optional RDD descriptor experiment

The default RDD script behavior remains LightGlue-only with cached RDD
features. To train only RDD's descriptor, keeping its detector and LightGlue
fixed, run:

```bash
export WILDLIFE_RDD_TRAIN_COMPONENT=descriptor
sbatch slurm_scripts/train_wildlife_rdd.sh
```

This mode computes RDD features online (the fixed feature cache is not valid
while the descriptor is changing) and saves separately under
`rdd-descriptor-finetuned/<protocol>/`. Other accepted values are `lg` (the
existing default), `rdd` (all RDD detector and descriptor weights), and
`lg+rdd` (LightGlue plus all RDD weights). Descriptor mode loads one sample
per GPU and accumulates 8 per optimizer step (effective batch 32, as in the
other modes); override with `WILDLIFE_BATCH_SIZE` / `WILDLIFE_GRAD_ACCUM_STEPS`.

### Optional LoMa descriptor experiment

Matcher-only remains the default. To train the complete DeDoDe descriptor
stack for the selected species, keep DaD and the LoMa matcher frozen and run:

```bash
export WILDLIFE_LOMA_TRAIN_COMPONENT=descriptor
sbatch slurm_scripts/train_wildlife_loma.sh
```

This uses LoMa-mined indices by default (or `WILDLIFE_TRAIN_INDEX` and
`WILDLIFE_VAL_INDEX` overrides), preserves the configured strict/legacy split,
and writes to `loma-descriptor-finetuned/<protocol>/`, separate from matcher
checkpoints. The script automatically builds a dataset-specific keypoint-only
cache for all train/validation/test frames; descriptors are recomputed during
training and evaluation. Override its path with
`WILDLIFE_LOMA_KEYPOINT_CACHE`. The descriptor microbatch size defaults to
`1` to reduce activation memory while gradients accumulate to the configured
training batch size. Use
`WILDLIFE_LOMA_TRAIN_COMPONENT=matcher` (the default) for the existing cached
matcher-only workflow.

## 4. Evaluate

Set `WILDLIFE_BACKEND`, `WILDLIFE_CACHE`, and `WILDLIFE_WEIGHTS` and submit the
generic evaluator. The test split is used for final reporting:

```bash
cd /home/kargin/Projects/repositories/rdd-parallel-benchmark
export WILDLIFE_CONFIG=/home/kargin/Projects/repositories/rdd-parallel-benchmark/configs/wildlife/BelugaID.json
export WILDLIFE_PROTOCOL=legacy
export WILDLIFE_BACKEND=rdd
export WILDLIFE_CACHE=/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/BelugaID/rdd-cache
export WILDLIFE_WEIGHTS=/home/kargin/Projects/repositories/lynx-finetuning/rdd/weights/RDD_lg-v2.pth
export WILDLIFE_MODE=full
sbatch slurm_scripts/wildlife_evaluate.sh
```

For the existing top-15 protocol, set `WILDLIFE_MODE=top15` and provide
`WILDLIFE_PRESELECT_WEIGHTS`. Change `WILDLIFE_BACKEND` and the cache/checkpoint
paths to run LoMa.
