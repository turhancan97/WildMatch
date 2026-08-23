# Generic WildlifeReID-10k workflow

The generic pipeline trains and evaluates one animal dataset at a time. The
supported configurations are in
`rdd-parallel-benchmark/configs/wildlife/`:

```text
BelugaID
NyalaData
WhaleSharkID
ZindiTurtleRecall (labeled rows with a resolved masked image)
```

NDD20 is intentionally not configured: its masked paths and identity labels
are not usable yet.

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

The training wrappers select the backend-specific combined index when it is
present (`indices/rdd/` for RDD and `indices/loma/` for LoMa), while retaining
the historical shared RDD path as a fallback. Validation is
`strong-matches_val_combined.json` in strict mode and
`strong-matches_test_combined.json` in legacy mode. Checkpoints and W&B runs
are isolated by dataset, backend, and protocol.

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
