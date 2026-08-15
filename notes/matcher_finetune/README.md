# lynx-finetuning

Finetuning feature matching modules for better performance on lynx-reidentification.

## Installation

Clone RDD and change commit
```bash
git clone --recursive https://github.com/xtcpete/rdd
cd rdd
git checkout 539508b270095969f9934c574cf7026bf37c434c
cd .. # root directory
```
Download `RDD-v2.pth` and `RDD_lg-v2.pth` checkpoints to rdd/weights

Install packages
```bash
conda create -n lynx-finetuning python=3.12
conda activate lynx-finetuning

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r rdd/requirements.txt
pip install -r contrastive_finetuning/requirements.txt
```

## Training

Training is driven by `contrastive_finetuning/train_by_lg_matches.py`, which fine-tunes RDD and/or its
LightGlue-based matcher (`--trained_model lg|rdd|lg+rdd`) on triplets (anchor / positive / negative) of
lynx crops.

Besides your image dataset, you need **train/val index files** — JSON files listing, for each anchor
image, its positive and candidate negative matches (mined with a given `top_k`/`top_m`). These indices
are **not built by this repo**; build them with
[rdd-parallel-benchmark](https://github.com/PiotrKubaty/rdd-parallel-benchmark) against your own dataset,
then point `--train_index` / `--val_index` at the resulting files.

```bash
python -m contrastive_finetuning.train_by_lg_matches \
    --train_index /path/to/index_train.json \
    --val_index /path/to/index_val.json \
    --data_root /path/to/dataset \
    --rdd_weights rdd/weights/RDD-v2.pth \
    --lg_weights rdd/weights/RDD_lg-v2.pth \
    --output_dir /path/to/output \
    --trained_model lg \
    --epochs 300 \
    --batch_size 32 \
    --lr 1e-5 \
    --weight_decay 1e-4 \
    --num_workers 16 \
    --lg_margin 0.5 \
    --random_negative_prob 0.3 \
    --seed 0
```

- `--train_index` / `--val_index`: index JSON files built with
  [rdd-parallel-benchmark](https://github.com/PiotrKubaty/rdd-parallel-benchmark).
- `--data_root`: root directory prepended to the (relative) image paths stored in the index files.
- `--output_dir`: where checkpoints (and, with `--project`, W&B logs) are written.
- `--trained_model`: which model(s) receive gradient — `lg`, `rdd`, or `lg+rdd`.

Run `python -m contrastive_finetuning.train_by_lg_matches --help` for the full list of options
(augmentation, LoRA, EMA, frozen confidence head, etc.).

If you're on the Helios cluster, `helios_scripts/train_lg.sh` is a ready-to-submit SLURM job wrapping
this same command with cluster-specific paths.

## `contrastive_finetuning/` layout

- **`train_by_lg_matches.py`** — main training entrypoint (see above). Trains RDD and/or LightGlue on a
  margin loss over LightGlue's own match-confidence scores; also wires up the optional augmentation,
  LoRA, EMA, and frozen-confidence-head mechanisms.
- **`train_common.py`** — shared building blocks used by the training script: the common CLI arguments
  (`add_common_args`), RDD feature extraction (`extract_train`), LightGlue matching (`run_lg_matching_grad`,
  `batch_features`), and the validation routines (`eval_epoch` — mean match counts; `eval_pseudo_accuracy` —
  frame/video pseudo-accuracy).
- **`loading.py`** — dataset/dataloader code: `IndexAssignedTripletDataset` reads a JSON index (as built by
  [rdd-parallel-benchmark](https://github.com/PiotrKubaty/rdd-parallel-benchmark)) and samples
  query/positive/negative triplets from it; `PseudoAccuracyDataset` returns a query with its *full*
  candidate pool for accuracy evaluation; `get_loader` builds a `DataLoader` from either.
- **`models.py`** — factory functions `build_rdd` and `build_masked_lg` that construct the RDD backbone and
  the (masked) LightGlue matcher from config + checkpoint weights.
- **`process.py`** — small tensor utilities (`pad_to_length`, `align_tensors_to_max_length`) for stacking
  variable-length keypoint/descriptor sets into a padded batch.
- **`eval_video_accuracy.py`** — standalone script to evaluate frame- and video-level pseudo-accuracy of a
  single checkpoint (no training); prints every misclassified video with its winning score and matched
  candidate frame.
- **`requirements.txt`** — Python dependencies for this subpackage.
## LoMa matcher fine-tuning

LoMa is an additive backend. The first experiment freezes LoMa's DaD detector
and DeDoDe descriptor and trains only the LoMa-B matcher on the existing lynx
positive/negative index. Install it in a separate environment because it has a
different dependency stack from the RDD pipeline:

```bash
conda create -n loma python=3.12 -y
conda activate loma
pip install -r requirements-loma.txt
```

Run a small local smoke experiment with W&B offline logging:

```bash
WANDB_MODE=offline python -m contrastive_finetuning.train_loma_matches \
  --trained_model loma \
  --train_index /path/to/train_index.json \
  --val_index /path/to/val_index.json \
  --data_root /shared/sets/datasets/confidential/lynx/processed_frames/segmented/lynx-ds-Jul-20 \
  --loma_variant loma-b \
  --output_dir checkpoints/loma-b-smoke \
  --project lynx-loma-finetuning \
  --wandb_mode offline \
  --epochs 1 \
  --max_train_batches 2 \
  --max_val_entries 2
```

For the cluster run, set `LOMA_WEIGHTS` if using a local pretrained LoMa-B
checkpoint and submit:

```bash
sbatch slurm_scripts/train_loma.sh
```

The output is a LoMa bundle (`model.safetensors`, `metadata.json`, optimizer
state, and RNG state). The benchmark consumes that bundle through its LoMa
backend; it must not be passed to the LightGlue loader.
