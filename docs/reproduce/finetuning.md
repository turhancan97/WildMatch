# Matcher fine-tuning

Repository: `lynx-finetuning`. One recipe is shared by LoMa and RDD-LightGlue; only the
network being adapted differs.

## Recipe

| Element | Setting |
|---|---|
| Trained part | matching module only (LoMa matcher, or LightGlue for RDD); detector, descriptor and global encoder frozen |
| Triplets | one positive sampled uniformly from the anchor's positive pool; one negative from the hard-negative pool or, with probability 0.3, a random image of another individual |
| Training score | relaxed score: mean over each image's real keypoints of the best assignment probability, averaged over both directions, computed before mutual selection and thresholding |
| Loss | triplet margin, \((0.5 - \tilde{s}(a,p) + \tilde{s}(a,n))_+\), over every triplet |
| Optimiser | AdamW, learning rate \(10^{-5}\), weight decay \(10^{-4}\), cosine schedule, gradient clipping at 1 |
| Batch | effective batch 32 (8 per GPU on 4 GPUs; descriptor modes load 1 per GPU and accumulate 8) |
| Schedule | 300 epochs; the final checkpoint (epoch 299) is reported, no epoch selection |
| Validation score | the inference score (mutual matches above the confidence threshold), on the test index in the legacy protocol |
| Padding | RDD batches pad keypoints to the longest image; padded keypoints are excluded from the assignment, so they cannot be matched |

On CzechLynx the training steps of the LoMa run take 5.1 GPU-hours on RTX 4090 GPUs; the
RDD-LightGlue run takes 5.0.

## Environment

```bash
git clone --recursive https://github.com/xtcpete/rdd && (cd rdd && git checkout 539508b)
conda create -n lynx-finetuning python=3.12 && conda activate lynx-finetuning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r rdd/requirements.txt -r contrastive_finetuning/requirements.txt
```

Download `RDD-v2.pth` and `RDD_lg-v2.pth` into `rdd/weights/`. LoMa has a different
dependency stack and uses a separate environment installed from `requirements-loma.txt`.

## Training

The generic entry point takes the mined index files and the dataset root:

```bash
python -m contrastive_finetuning.train_by_lg_matches \
    --train_index <index>/strong-matches_train_combined.json \
    --val_index   <index>/strong-matches_test_combined.json \
    --data_root <dataset view> \
    --rdd_weights rdd/weights/RDD-v2.pth --lg_weights rdd/weights/RDD_lg-v2.pth \
    --output_dir <checkpoints>/rdd-finetuned-loma-mined-legacy \
    --trained_model lg --epochs 300 --batch_size 32 --lr 1e-5 --weight_decay 1e-4 \
    --lg_margin 0.5 --random_negative_prob 0.3 --seed 0
```

The Slurm wrappers used for the paper set these values and derive the index, cache and
output paths from the dataset:

```bash
# CzechLynx, time-closed split, legacy protocol (defaults)
sbatch slurm_scripts/train_czechlynx_rdd.sh
sbatch slurm_scripts/train_czechlynx_loma.sh

# One WildlifeReID-10k dataset or SalamanderID2025
export WILDLIFE_CONFIG=<rdd-parallel-benchmark>/configs/wildlife/NyalaData.json
export WILDLIFE_PROTOCOL=legacy
sbatch slurm_scripts/train_wildlife_rdd.sh
sbatch slurm_scripts/train_wildlife_loma.sh
```

The RDD wrapper reads the LoMa-mined index by default and fails if it is missing; the
Salamander RDD run sets `WILDLIFE_MINING_BACKEND=rdd`. A wrapper refuses to start into an
output directory that already holds epochs; long runs resume from a saved epoch directory
and reject a checkpoint whose objective, optimiser or batch configuration differs. Every
50th epoch is kept, plus the first and the last.

## Checkpoint layout

```text
<checkpoints>/<dataset>/<matcher>-finetuned-<pairs>-mined-legacy/
    czechlynx_protocol.json | wildlife_protocol.json   # split, backend, indices, component, objective
    epoch_000/ epoch_050/ ... epoch_299/
        model.safetensors                              # LightGlue (RDD) or the LoMa bundle
        optimizer / scheduler / RNG state              # never loaded for evaluation
```

The protocol file records the trained component (`lg` or `matcher` for the paper's main
runs), the pair source, the split protocol and the training objective, and the evaluation
code validates it before loading a checkpoint.

## Ablation variants

The "what to adapt" comparison trains the other parts with the same pairs and objective:

```bash
# descriptor branch only (detector and matcher frozen)
CZECHLYNX_RDD_TRAIN_COMPONENT=descriptor  sbatch slurm_scripts/train_czechlynx_rdd.sh
CZECHLYNX_LOMA_TRAIN_COMPONENT=descriptor sbatch slurm_scripts/train_czechlynx_loma.sh

# descriptor and matcher together (detector frozen)
CZECHLYNX_RDD_TRAIN_COMPONENT=joint  sbatch slurm_scripts/train_czechlynx_rdd.sh
CZECHLYNX_LOMA_TRAIN_COMPONENT=joint sbatch slurm_scripts/train_czechlynx_loma.sh
```

A trainable descriptor rules out the fixed feature cache, so features are recomputed every
step and these runs cost 16 to 21 times more than the matcher-only run. They are written to
separate `*-descriptor-finetuned-*` and `*-joint-finetuned-*` directories and are reported
separately from matcher-only fine-tuning.

## Unseen-identity protocol

The matcher for the unseen-identity experiment is trained the same way on the training
part of the CzechLynx time-open split (275 individuals), with pairs mined on that split.
