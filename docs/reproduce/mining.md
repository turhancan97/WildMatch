# Pair mining

Repository: `rdd-parallel-benchmark`. Pair mining runs once per dataset and matcher with
the **pretrained** matcher on the training (database) split only. It produces the
strong-match index that fine-tuning samples its triplets from. The index is tied to one
fixed pretrained checkpoint by design: it defines the candidate pool the adaptation is
trained and validated on.

!!! info "Commands are examples"
    Paths below are placeholders for your own data and checkpoint locations. The Slurm
    headers in the repository target one specific cluster; adjust partitions, accounts and
    the activation script for yours.

## Environment

The RDD pipeline runs in its own environment. On a workstation:

```bash
git clone --recursive <rdd-parallel-benchmark>
cd rdd-parallel-benchmark
conda create -n rdd python=3.10 pip && conda activate rdd
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt safetensors
```

Place the pretrained RDD weights in `weights/`: `RDD-v2.pth` (detector and descriptor,
used for feature extraction) and `RDD_lg-v2.pth` (the released LightGlue matcher for RDD).
LoMa mining uses the pretrained LoMa-B checkpoint in a separate `loma` environment.

Every worker script starts by sourcing one activation script; replace its contents with
your site's activation and nothing else changes. Spawn scripts submit their own Slurm jobs,
so they are run with `bash`, not `sbatch`, from the repository root.

## Canonical dataset view

Mining expects a symlink view with `train/` and `test/` directories and one folder per
individual and collection. The preparation step builds it from the dataset metadata; for
WildlifeReID-10k datasets it reads the official pre-masked images and never falls back to
unmasked ones.

```bash
# CzechLynx (time-closed split)
sbatch slurm_scripts/prepare_czechlynx.sh

# One WildlifeReID-10k dataset (or SalamanderID2025)
export WILDLIFE_CONFIG=configs/wildlife/NyalaData.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/prepare_wildlife.sh
```

`legacy` is the protocol used for the paper: the official training (database) split is
used for fine-tuning and the official test (query) split for training-time validation and
final reporting. A `strict` protocol with a held-out validation part exists but is not
used in the paper.

## Feature cache

All scoring reads a feature cache, never the images. Each image is resized to a long side
of 512 pixels (dimensions floored to a multiple of 32 for RDD, 14 for LoMa), run through
the detector and descriptor, and stored with up to 512 keypoints. The cache is only valid
for one combination of keypoint budget, resolution and preprocessing, which is why the
cache directory names encode them.

```bash
# from lynx-finetuning
sbatch slurm_scripts/build_czechlynx_rdd_cache.sh     # or build_czechlynx_loma_cache.sh
sbatch slurm_scripts/build_wildlife_rdd_cache.sh      # or build_wildlife_loma_cache.sh
```

Images are decoded with PIL to float tensors in [0, 1] and resized with plain bilinear
interpolation without antialiasing, the same function the fine-tuning code uses, so the
features mined here are the features trained on.

## Mining the index

```bash
# CzechLynx
bash slurm_scripts/spawn_czechlynx_mining.sh

# WildlifeReID-10k dataset, LoMa pairs (the pair source used for every paper dataset
# except Salamander's RDD run)
export WILDLIFE_MINING_BACKEND=loma
export WILDLIFE_LOMA_CACHE=<checkpoints>/wildlife-reid-10k/NyalaData/loma-cache
export LOMA_WEIGHTS=<pretrained>/loma_B.pt
bash slurm_scripts/spawn_wildlife_mining.sh
```

The job chain builds the cache, mines every query collection as a Slurm array task, and
aggregates the per-query reports into one combined JSON per split with image paths relative
to the dataset root.

| Setting | Value | Meaning |
|---|---:|---|
| keypoints per image | 512 | detector budget, fixed at cache build |
| long side | 512 px | resize before extraction |
| images per collection | 20 | uniformly sampled; shorter collections contribute all images |
| positives and negatives kept per anchor | 5 each | highest-scoring same-identity and different-identity images, spread over distinct collections |
| anchors kept per collection | 10 | ranked by their best pair score |

Anchors without a positive are dropped. The query's own collection is always excluded from
the candidate gallery, so a positive never comes from the same encounter.

Output: `outputs/<dataset>/<protocol>/<backend>/strong-matches_{train,test}_combined.json`
(the CzechLynx layout) or `outputs/wildlife-reid-10k/<dataset>/indices/<backend>/` with a
metadata sidecar recording backend, checkpoint, cache and settings. In the legacy protocol
the validation index is an alias of the test index.

## Pair sources used in the paper

Every dataset fine-tunes both matchers on LoMa-mined pairs, with one documented exception:
SalamanderID2025 fine-tunes RDD-LightGlue on RDD-mined pairs. Indices of the two backends
are kept in separate directories and never mixed.
