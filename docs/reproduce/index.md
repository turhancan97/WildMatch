# Reproduce

The whole pipeline is one installable package, `wildmatch`, with one command per step. Each step
reads the previous step's outputs from the storage layout set by the path profile.

| Step | Command | What it does |
|---|---|---|
| 1. [Pair mining](mining.md) | `wildmatch mine` | Builds feature caches, matches every training image against the others with the pretrained matcher, and writes the positive and negative pools (strong-match indices). |
| 2. [Matcher fine-tuning](finetuning.md) | `wildmatch finetune-matcher` | Samples triplets from the pools and fine-tunes the matching module (LoMa or RDD-LightGlue) with the triplet margin loss on the relaxed score; saves per-epoch checkpoints. |
| 3. [Evaluation](probing.md) | `wildmatch evaluate` | Runs the probes: cosine retrieval, WildFusion, default and fine-tuned matchers over the MegaDescriptor-L candidate list, and the classifier baselines; exports tables and figures. |

!!! note
    The commands shown are the ones used for the manuscript, with machine-specific paths
    replaced by `<placeholders>`. `wildmatch <command> --help` lists every option. The Slurm
    scripts in `slurm/` target one specific cluster; adjust partitions and the environment
    location for yours.

## Environment

One environment with pinned dependencies (Python 3.12) runs all three steps:

```bash
uv sync --extra cu126 --extra matchers --extra train --group dev   # or --extra cpu
source .venv/bin/activate
```

`matchers` adds the local matchers through Vismatch, pinned to a fixed commit, which downloads
pretrained matcher weights on first use; `train` adds what mining and fine-tuning need
(accelerate and LoMa, pinned). The RDD code used for mining and fine-tuning ships inside the
package. Evaluation alone does not need `train`.

## Data layout

Every location comes from a path profile: `data_root` for the datasets, `checkpoint_root` for
feature caches and fine-tuned checkpoints, and `external.mining_outputs` for the mined indices.
The `default` profile reads them from environment variables (`WILDMATCH_DATA_ROOT`,
`WILDMATCH_CHECKPOINT_ROOT`, `WILDMATCH_MINING_OUTPUTS`), and the pretrained weights the same way
(`WILDMATCH_LOMA_WEIGHTS` for `loma_B.pt`, `WILDMATCH_RDD_WEIGHTS_DIR` for the folder holding
`RDD-v2.pth` and `RDD_lg-v2.pth`). Datasets are entries of the dataset registry, selected by key
(`nyala`, `salamander`, `czechlynx_closed`, ...).

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
