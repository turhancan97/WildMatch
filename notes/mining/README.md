# RDD parallel benchmark — lynx re-identification on SLURM

Fork of [RDD: Robust Feature Detector and Descriptor using Deformable Transformer (CVPR 2025)](https://xtcpete.github.io/rdd/)
stripped down to what we actually use: sequence-level **lynx re-identification** with
RDD keypoints + LightGlue matching, run as SLURM array jobs.

The upstream MegaDepth / ScanNet / Air-to-Ground benchmarks have been removed —
only the `lynx_*` pipeline in [scripts/](scripts/) and its SLURM wrappers in
[slurm_scripts/](slurm_scripts/) are maintained here.

## Table of contents

- [Installation](#installation)
- [Dataset layout and the feature cache](#dataset-layout-and-the-feature-cache)
- [1. Retrieval evaluation — `spawn_evaluation.sh`](#1-retrieval-evaluation--spawn_evaluationsh)
- [2. Strong-match mining — `spawn_find_strong_matches.sh`](#2-strong-match-mining--spawn_find_strong_matchessh)
- [3. Two-stage evaluation — `spawn_two_stage.sh`](#3-two-stage-evaluation--spawn_two_stagesh)
- [Hyperparameters at a glance](#hyperparameters-at-a-glance)
- [Why `_load_train_approach` in `lynx_build_cache.py`](#why-_load_train_approach-in-lynx_build_cachepy)
- [Known inconsistencies / gotchas](#known-inconsistencies--gotchas)
- [Citation, license, acknowledgements](#citation-license-acknowledgements)

## Installation

On Helios, skip straight to [Cluster environment](#cluster-environment-helios_scripts) — the
`helios_scripts/` scripts do all of this for you. The conda recipe below is for a local
workstation.

```bash
git clone --recursive <this-repo>
cd rdd-parallel-benchmark

# Create conda env
conda create -n rdd python=3.10 pip
conda activate rdd

# CUDA + torch
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install -r requirements.txt

# safetensors is needed only for the two-stage pipeline (finetuned checkpoints
# come out of accelerate as .safetensors)
pip install safetensors

# Custom deformable-attention ops. Not required to run RDD, but recommended.
cd ./RDD/models/ops
pip install -e .
```

### Weights

Put both checkpoints in `weights/` (the directory is gitignored):

| File | Used by | Notes |
| --- | --- | --- |
| `weights/RDD-v2.pth` | `lynx_build_cache.py` (feature extraction) | default of `--weights`, never overridden by the SLURM scripts |
| `weights/RDD_lg-v2.pth` | LightGlue matcher in every scoring script | pretrained LightGlue-for-RDD |

Finetuned LightGlue checkpoints (e.g. `model-serene-firefly-40.safetensors`, produced by the
`contrastive_finetuning` repo) are passed explicitly — see
[spawn_two_stage.sh](slurm_scripts/spawn_two_stage.sh).

### Cluster environment (`helios_scripts/`)

On Helios the environment is a plain `venv` under `$SCRATCH`, not conda. Three scripts in
[helios_scripts/](helios_scripts/) cover it:

| Script | What it does |
| --- | --- |
| [`_activate.sh`](helios_scripts/_activate.sh) | `ml ML-bundle/25.10`, `GCCcore/14.3.0`, `Python/3.13.5`, then `source $SCRATCH/<env>/bin/activate`. Returns to the original working directory, so **it is sourced, never sbatched** — the `#SBATCH` header it carries is inert. |
| [`create_env.sh`](helios_scripts/create_env.sh) | one-off `python -m venv $SCRATCH/<env>`. Run with `sbatch`. |
| [`install_packages.sh`](helios_scripts/install_packages.sh) | sources `_activate.sh`, then installs torch/torchvision (cu132 wheels) + `requirements.txt` and prints `pip show` for each package. Run with `sbatch`. |

So a first-time setup on Helios is:

```bash
sbatch helios_scripts/create_env.sh
sbatch helios_scripts/install_packages.sh   # after the first one finishes
```

**Every** worker script in [slurm_scripts/](slurm_scripts/) starts with
`source helios_scripts/_activate.sh`. If you run elsewhere, keep the file name and replace its
contents with your own site's activation (e.g. `conda activate rdd`); nothing else needs to
change.

All SLURM headers currently target Helios (`--account=plgittossl2-gpu-gh200`,
`--partition=plgrid-gpu-gh200`); change them if you run elsewhere.

All commands below are run **from the repository root** (the scripts use
`python -m scripts.<module>` and relative `slurm_scripts/...` paths), and the spawn scripts
submit jobs themselves, so run them with `bash`, not `sbatch`:

```bash
mkdir -p logs
bash slurm_scripts/spawn_evaluation.sh
```

## Dataset layout and the feature cache

All scripts assume:

```
<dataset_root>/
  train/lynx_<id>/<site>/<sequence_id>/frame_XXXX.jpg
  test/ lynx_<id>/<site>/<sequence_id>/frame_XXXX.jpg
```

A "video" (sequence) is the third level below the split — that is exactly what the spawn
scripts count with `find <root>/<split> -mindepth 3 -maxdepth 3 -type d` to size the SLURM
array. The **gallery is always `train`**; the query split varies per pipeline.

Every pipeline starts with the same step: [build_cache.sh](slurm_scripts/build_cache.sh) →
[scripts/lynx_build_cache.py](scripts/lynx_build_cache.py), which samples `sample` frames per
sequence (uniform, via `sample_frames`), runs RDD, and writes one
`.npz` per frame (keypoints, descriptors, scores, image size) mirroring the dataset tree under
`cache_dir`. Frames already cached are skipped with a warning, so re-running is cheap —
**but the cache is only valid for one (`top_k`, `resize_max`, `sample`) triple and one
preprocessing function**; nothing in the filename records those, which is why the cache
directory names encode them.

Scoring scripts never touch images — they read the cache. Consequently `top_k`/`resize_max`
cannot be changed after cache build, and passing them to a scoring job has no effect.

## 1. Retrieval evaluation — `spawn_evaluation.sh`

**What it does.** The standard sequence-level retrieval benchmark: every `test` video is
scored against every `train` video with LightGlue, frame scores are pooled into a video
score, and top-1 / top-5 / frame accuracy are reported.

**Job chain** ([spawn_evaluation.sh](slurm_scripts/spawn_evaluation.sh)):

```
build_cache.sh                        (1 GPU job, features for train+test)
  └─ query_in_parallel.sh             (array 0..n_test-1, %32 concurrent)
       └─ aggregate.sh                (1 job, merges the per-query JSONs)
```

`query_in_parallel.sh` → [lynx_query_in_parallel.py](scripts/lynx_query_in_parallel.py) handles
**one query video per array task** (`--query_id $SLURM_ARRAY_TASK_ID`), scores it against all
gallery frames in batches of 32, pools with `top_m_pool`, and dumps
`<dump_report>_parallel_<query_id>.json`.
`aggregate.sh` → [lynx_aggregate.py](scripts/lynx_aggregate.py) globs those files and prints the
final metrics.

Its positional arguments are
`dataset_root cache_dir dump_report top_m_pool sample lg_weights` — note there is no `top_k`
among them: the keypoint count is baked into the cache and a scoring job cannot change it.

**Hyperparameters** (top of the spawn script):

| Name | Value | Meaning |
| --- | --- | --- |
| `top_k` | `512` | RDD keypoints per frame (cache build only) |
| `resize_max` | `512` | long side before extraction, floored to a multiple of 32 |
| `sample` | `20` | frames sampled uniformly per video (`--frames_per_seq`) |
| `top_m_pool` | `1` | frame scores pooled into a video score. `1` ⇒ the video score is the single strongest (query frame, gallery frame) pair |
| `lg_weights` | `weights/RDD_lg-v2.pth` | LightGlue checkpoint used for scoring |
| `ds_version` | `big` | label only — goes into the output paths |

**Run:**

```bash
bash slurm_scripts/spawn_evaluation.sh
```

**Outputs:** cache in `outputs/eval-Jul29/<lg_weights>/lynx_cache-<ds_version>-<top_k>-<resize_max>-<sample>/`,
reports in `outputs/eval-Jul29/<lg_weights>/reports/<ds_version>-<top_k>-<top_m_pool>-<resize_max>-<sample>/`.

To evaluate a different LightGlue checkpoint, change `lg_weights` — it is threaded through to
`--lg_weights` and is part of the output path, so runs do not overwrite each other.

## 2. Strong-match mining — `spawn_find_strong_matches.sh`

**What it does.** Builds the **triplet index consumed by the finetuning repo**
(`IndexAssignedTripletDataset` in `contrastive_finetuning`). For each query frame it keeps the
`top_k_frames` best-scoring positives (same `lynx_id`) and `top_k_frames` best-scoring negatives
(different `lynx_id`), preferring one frame per distinct video (`select_diverse_topk`), then
keeps the `top_m` best query frames per video. The query's own video is always excluded from
the gallery.

Both splits are mined: `test` queries and `train` queries (the latter against the rest of
`train`).

**Job chain** ([spawn_find_strong_matches.sh](slurm_scripts/spawn_find_strong_matches.sh)):

```
build_cache.sh
  ├─ find_strong_matches_in_parallel.sh   (array 0..n_test-1,  split=test)
  ├─ find_strong_matches_in_parallel.sh   (array 0..n_train-1, split=train)
  └─ find_strong_matches_aggregate.sh     (after both, one combined JSON per split)
```

[lynx_find_strong_matches_aggregate.py](scripts/lynx_find_strong_matches_aggregate.py) flattens
the per-query reports into `<dump_report>_<split>_combined.json`, with frame paths made
**relative to `dataset_root`** — that is the file the finetuning repo reads.

**Hyperparameters:**

| Name | Value | Meaning |
| --- | --- | --- |
| `top_k` | `512` | RDD keypoints per frame (cache build only) |
| `resize_max` | `512` | long side before extraction |
| `sample` | `20` | frames per video |
| `top_k_frames` | `5` | positives **and** negatives kept per query frame |
| `top_m` | `10` | query frames kept per video, ranked by their best score |

LightGlue weights are **not** configurable here: `build_masked_lg(device)` is called without a
`weights=` argument, so the index is always mined with the pretrained `./weights/RDD_lg-v2.pth`.
That is intentional — the index defines the candidate pool the finetuning is trained and
evaluated on, so it must stay tied to one fixed checkpoint.

**Run:**

```bash
bash slurm_scripts/spawn_find_strong_matches.sh
```

**Outputs:** `outputs/reports/strong_matches-<ds_version>-<top_k>-<top_k_frames>-<top_m>-<resize_max>-<sample>/top_k=<top_k_frames>_top_m=<top_m>_{train,test}_combined.json`.

⚠️ The cache directory here (`outputs/lynx_cache_Jul20-...`) is the one the **existing index was
mined with**, built with the old `parse_input` preprocessing. Re-running this script now
rebuilds missing frames with `_load_train_approach` (see below), which would make the cache
internally inconsistent. Mine a fresh index into a **new** cache directory instead of topping
up the old one.

## 3. Two-stage evaluation — `spawn_two_stage.sh`

**What it does.** Explains the gap between "pseudo accuracy went up during finetuning" and
"retrieval accuracy went down". The index from section 2 only stores the candidates the
**pretrained** checkpoint considered strong; a finetuned checkpoint's true global argmax may be
a negative that was never written to the index. So the two roles are split:

- **stage 1 (preselect)** — the pretrained checkpoint picks a narrow candidate pool,
- **stage 2 (score)** — the finetuned checkpoint scores only that pool; the prediction is the
  `lynx_id` of the strongest surviving pair.

Four preselection rules, selected with `mode` (comma-separated subset or `all`) — full
description in the docstring of
[lynx_query_two_stage_in_parallel.py](scripts/lynx_query_two_stage_in_parallel.py):

| Mode | Pool | Purpose |
| --- | --- | --- |
| A | per-identity query frames, 1 frame per identity | tightest, perfectly symmetric |
| B | `top_m` global query frames × one diverse top-k over the whole gallery | closest copy of how the index is built |
| C | like B, but `top_k_frames` per identity | superset of the index's pool per query frame |
| D | no preselection — finetuned checkpoint scores the whole gallery | ground truth; the only expensive mode |

A/B/C are subsets of the stage-1 score matrix and are essentially free to combine; D adds a
second full pass.

**Job chain** ([spawn_two_stage.sh](slurm_scripts/spawn_two_stage.sh)):

```
build_cache.sh (score cache only)
  └─ query_two_stage_in_parallel.sh   (array 0..n_query-1, %32)
       └─ two_stage_summary.sh        (one accuracy per mode)
```

**Two caches, deliberately:**

| Argument | Directory | Built with | Read by |
| --- | --- | --- | --- |
| `preselect_cache_dir` (`--cache_dir`) | `outputs/_eval-Aug03/...` | `parse_input` (cv2 decode + antialiased kornia resize) | stage 1, `preselect_weights` |
| `score_cache_dir` | `outputs/eval-Aug03/...` | `_load_train_approach` | stage 2, `score_weights` |

Stage 1 must reproduce the *real* index file, so it has to read the features the index was
mined with — the preselect cache is **never** passed to `build_cache.sh` and must not be
rebuilt (verified with [scripts/debug/lynx_debug_preselect_pool.py](scripts/debug/lynx_debug_preselect_pool.py)).
Stage 2 must show the finetuned checkpoint the preprocessing it was trained on. The two stages
are independent read-only passes, so the caches never need to agree with each other — only
each with the checkpoint reading it.

**Hyperparameters:**

| Name | Value | Meaning |
| --- | --- | --- |
| `top_k` | `512` | RDD keypoints per frame (cache build only) |
| `resize_max` | `512` | long side before extraction |
| `sample` | `20` | frames per video |
| `top_m_pool` | `1` | pooling, same as section 1 — video score = strongest surviving pair |
| `top_m` | `10` | query frames kept — **must match section 2** |
| `top_k_frames` | `5` | candidates per identity per query frame — **must match section 2** |
| `top_k_global` | `0` (default) | mode B only; `0` ⇒ `top_k_frames`, mirroring the index's negative pool |
| `mode` | `all` | A,B,C,D |
| `split` | `test` | query split (gallery is always `train`) |
| `preselect_weights` | `weights/RDD_lg-v2.pth` | pretrained, the checkpoint the index was mined with |
| `score_weights` | `model-serene-firefly-40.safetensors` | finetuned checkpoint under evaluation |

`top_m` and `top_k_frames` are not free knobs: the index reconstruction is only faithful when
they equal the values section 2 mined with (`10` / `5`).

**Run:**

```bash
bash slurm_scripts/spawn_two_stage.sh
```

**Outputs:** `outputs/two-stage-Aug03/<score_weights>/reports/.../top_m=1-<mode>_parallel_<query_id>.json`
per mode, plus two references — `-index_parallel_*` (the index's own pool, rescored with the
score checkpoint, i.e. exactly `eval_pseudo_accuracy`'s criterion) and `-pre-full_parallel_*`
(preselect checkpoint over the full gallery, identical to section 1).
[lynx_two_stage_summary.py](scripts/lynx_two_stage_summary.py) prints one accuracy per mode.

Only `top1_acc` and `frame_acc` are comparable across modes — gallery videos with no
preselected candidate score 0, so top5/mAP of the restricted pools are meaningless.

## Hyperparameters at a glance

Everything that reaches the cache is **identical across all three pipelines**, so a cache built
by one is feature-compatible with the others (preprocessing permitting, see below):

| | evaluation | strong matches | two-stage |
| --- | --- | --- | --- |
| `top_k` (RDD keypoints) | 512 | 512 | 512 |
| `resize_max` (long side) | 512 | 512 | 512 |
| `sample` (frames/video) | 20 | 20 | 20 |
| `top_m_pool` | 1 | — | 1 |
| `top_m` | — | 10 | 10 |
| `top_k_frames` | — | 5 | 5 |
| query split | test | test + train | test |
| gallery | train | train | train |
| array throttle | `%32` | `%32` | `%32` |

The Python defaults **do not** match these values (`lynx_build_cache.py`: `top_k=2048`,
`resize_max=1024`, `frames_per_seq=10`; `lynx_query_in_parallel.py`: `top_m_pool=5`). Always go
through the spawn scripts, or pass every flag explicitly.

## Why `_load_train_approach` in `lynx_build_cache.py`

[lynx_build_cache.py](scripts/lynx_build_cache.py) contains three image loaders and uses
[`_load_train_approach`](scripts/lynx_build_cache.py#L121); `parse_input` and
`_load_image_exreid` are kept only for reference/debugging.

| Loader | Decode | Resize |
| --- | --- | --- |
| `parse_input` (upstream RDD demo) | `cv2.imread` + BGR→RGB | `kornia.geometry.transform.resize`, `antialias=True`, bilinear, dims floored to ×32 |
| `_load_image_exreid` | PIL | PIL bilinear, downscale only |
| **`_load_train_approach` (in use)** | **PIL → float [0,1]** | **`F.interpolate` bilinear, `align_corners=False`, no antialiasing, dims floored to ×32** |

**The reason is consistency with the finetuning repository.** `contrastive_finetuning`
(`train_common`) loads images with PIL + ToTensor and its own `resize_long_side`, which is
plain `F.interpolate` bilinear without antialiasing. `_load_train_approach` reproduces that
function exactly. This matters in two places:

1. **The index this repo produces is consumed there.** The strong-match index (section 2) is a
   list of frame paths whose "strong"-ness was decided by scores computed on *these* features.
   If the finetuning run re-extracts the same frames through a different preprocessing path, the
   keypoints and descriptors it trains on are not the ones the index was selected with, and the
   supervision drifts away from what was mined.
2. **A finetuned checkpoint scored on foreign preprocessing loses accuracy.** The finetuned
   LightGlue only ever saw images through `resize_long_side`. Feeding it `parse_input` features
   at eval time is a train/test mismatch: per
   [scripts/debug/lynx_debug_preprocessing.py](scripts/debug/lynx_debug_preprocessing.py) it is
   roughly **4–4.7× more sensitive** to this exact preprocessing gap than the pretrained
   checkpoint is (that script scores the same pairs under all three loaders with both
   checkpoints and reports the spread per checkpoint).
   The features themselves really do differ:
   [scripts/debug/lynx_compare_caches.py](scripts/debug/lynx_compare_caches.py) compares a
   `parse_input` cache against a `_load_train_approach` one and lands around
   `cos_sim ≈ 0.995` with a few keypoints gained/lost per frame — small, but not nothing.

Antialiasing is the dominant difference: it low-pass filters the image before downsampling, so
RDD's soft-detector fires on a measurably different set of locations, and with only
`top_k = 512` keypoints kept per frame the surviving set differs enough to move LightGlue
scores.

The one place that must **not** use it is stage 1 of the two-stage pipeline: it reconstructs an
index that was mined before under `parse_input`, so its cache is frozen. That is exactly
why `spawn_two_stage.sh` keeps two cache directories and only ever passes the stage-2 one to
`build_cache.sh`.

## Known inconsistencies / gotchas
- **Each spawn script uses its own cache directory** (`eval-Jul29/...`, `lynx_cache_Jul20-...`,
  `eval-Aug03/...`) even though the cache hyperparameters are identical, so the same features
  can be extracted several times. Intentional for the two-stage preselect cache (different
  preprocessing), incidental for the rest.
- **`spawn_two_stage.sh` names the preselect cache after `${score_weights}`** (with a leading
  `_` to distinguish it) although it belongs to the preselect checkpoint. Path naming only.
- **`sample=20` is a maximum.** `sample_frames` returns all frames of shorter sequences, so
  videos contribute unequal numbers of frames.

## Citation, license, acknowledgements

```
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Gonglin and Fu, Tianwen and Chen, Haiwei and Teng, Wenbin and Xiao, Hanyuan and Zhao, Yajie},
    title     = {RDD: Robust Feature Detector and Descriptor using Deformable Transformer},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6394-6403}
}
```

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Upstream RDD builds on [ALIKE](https://github.com/Shiaoming/ALIKE),
[LoFTR](https://github.com/zju3dv/LoFTR), [DeDoDe](https://github.com/Parskatt/DeDoDe),
[XFeat](https://github.com/verlab/accelerated_features),
[LightGlue](https://github.com/cvg/LightGlue), [Kornia](https://github.com/kornia/kornia) and
[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
