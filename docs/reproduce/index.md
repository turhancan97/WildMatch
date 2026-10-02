# Reproduce

The pipeline spans three repositories. Each step reads the previous step's outputs
from the shared storage layout described on its page.

| Step | Repository | What it does |
|---|---|---|
| 1. [Pair mining](mining.md) | `rdd-parallel-benchmark` | Builds feature caches, matches every training image against the others with the pretrained matcher, and writes the positive and negative pools (strong-match indices). |
| 2. [Matcher fine-tuning](finetuning.md) | `lynx-finetuning` | Samples triplets from the pools and fine-tunes the matching module (LoMa or RDD-LightGlue) with the triplet margin loss on the relaxed score; saves per-epoch checkpoints. |
| 3. [Evaluation](probing.md) | `explainable_individual_reidentification` (this repository) | Runs the probes: cosine retrieval, WildFusion, default and fine-tuned matchers over the MegaDescriptor-L candidate list, and the classifier baselines; exports tables and figures. |

!!! note
    The commands shown are the ones used for the manuscript, with cluster-specific paths
    replaced by `<placeholders>`. Each repository's own README holds the complete option
    lists.

## Environment

All evaluation code runs in one conda environment with the pinned dependencies listed
in the repository's `requirements.txt`; Vismatch is pinned to a fixed commit and
downloads matcher weights on first use. Fine-tuning and mining have their own
environments, described on their pages.

## Data layout

Each dataset is a root directory with an `images/` or `masked_images/` folder and a
metadata CSV holding the image path, identity, split and, where applicable, a COCO-RLE
mask. Mining and fine-tuning read a symlink view with one folder per individual and
collection built from that metadata; evaluation reads the metadata directly. Feature
caches and run artifacts are keyed by content hashes of images, metadata and checkpoints,
so a changed input can never reuse stale features.

## Shared settings

| Setting | Value |
|---|---|
| Image long side | 512 px |
| Keypoints per image | up to 512 |
| Pairs per anchor | 5 positives, 5 hard negatives |
| Training | 300 epochs, AdamW \(10^{-5}\), effective batch 32, margin 0.5 |
| Reported checkpoint | epoch 299, no selection |
| Candidate budgets | 10, 50, 100, 250 (default), 500, 1000 |
| Metrics | Top-5 accuracy, balanced Top-1 accuracy |
