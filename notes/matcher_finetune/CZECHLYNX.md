# CzechLynx time-closed workflow

The integration uses masked real CzechLynx images and the metadata
`split-time_closed` protocol. The converter creates a symlink view with
`train`, `val`, and `test` directories; validation is a deterministic 20%
encounter holdout from metadata training encounters, while metadata test
frames are reserved for final evaluation.

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
