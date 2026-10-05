# Pair mining

Command: `wildmatch mine`. Pair mining runs once per dataset and matcher with the **pretrained**
matcher on the training (database) split only. It produces the strong-match index that
fine-tuning samples its triplets from. The index is tied to one fixed pretrained checkpoint by
design: it defines the candidate pool the adaptation is trained and validated on.

!!! info "Commands are examples"
    Locations come from the path profile (see [Reproduce](index.md#data-layout)). The Slurm
    script `slurm/mine.sbatch` targets one specific cluster; adjust its partition and the
    environment location for yours. Run the commands from the repository root.

## Steps

`wildmatch mine <step> --dataset <registry key> --backend loma|rdd` runs one step; `--protocol`
defaults to `legacy`. Start with `plan`, which writes nothing and prints the view, cache, index
locations and the exact command of every later step:

```bash
wildmatch mine plan --dataset nyala --backend loma
```

| Step | What it does |
|---|---|
| `view` | builds the symlink view from the registry entry's metadata |
| `cache` | extracts the per-image feature cache (GPU) |
| `check` | validates the view, the weights and the cache before mining |
| `submit` | freezes the run and submits the per-split mining arrays and the aggregation |

## Canonical dataset view

Mining expects a symlink view with `train/` and `test/` directories (and `val/` for CzechLynx)
and one folder per individual and collection. The `view` step builds it from the dataset's
registry entry; for WildlifeReID-10k datasets it reads the pre-masked images the paper's runs
used and never falls back to unmasked ones.

```bash
wildmatch mine view --dataset czechlynx_closed --backend rdd   # CzechLynx, time-closed split
wildmatch mine view --dataset nyala --backend loma             # a WildlifeReID-10k dataset
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
sbatch slurm/mine.sbatch cache --dataset nyala --backend loma   # one GPU
wildmatch mine check --dataset nyala --backend loma
```

Images are decoded with PIL to float tensors in [0, 1] and resized with plain bilinear
interpolation without antialiasing, the same function fine-tuning and evaluation use, so the
features mined here are the features trained on.

## Mining the index

```bash
wildmatch mine submit --dataset nyala --backend loma --dry-run   # print the Slurm commands
wildmatch mine submit --dataset nyala --backend loma
```

`submit` checks the inputs and writes the resolved run, with the code identity and the
checksums of the weights and the cache, to a submission file that every job then executes; a
job stops if the code or an input changed since submission. Mined files and an existing
index are never overwritten without `--overwrite`.

The submission mines every query collection as one Slurm array task per split and then
aggregates the per-query reports into one combined JSON per split, with image paths relative
to the view root.

| Setting | Value | Meaning |
|---|---:|---|
| keypoints per image | 512 | detector budget, fixed at cache build |
| long side | 512 px | resize before extraction |
| images per collection | 20 | uniformly sampled; shorter collections contribute all images |
| positives and negatives kept per anchor | 5 each | highest-scoring same-identity and different-identity images, spread over distinct collections |
| anchors kept per collection | 10 | ranked by their best pair score |

Anchors without a positive are dropped. The query's own collection is always excluded from
the candidate gallery, so a positive never comes from the same encounter.

Output, under `external.mining_outputs`:
`czechlynx-time-closed/<protocol>/<backend>/strong-matches_{train,val,test}_combined.json` for
CzechLynx, or `wildlife-reid-10k/<dataset>/indices/<backend>/strong-matches_{train,test}_combined.json`
for the other datasets, each with a metadata sidecar recording backend, checkpoint, cache and
settings. In the legacy protocol the validation index is an alias of the test index.

## Pair sources used in the paper

Each published checkpoint records the index it was trained on. Indices of the two backends are
kept in separate directories and never mixed within one run.

| Dataset | LoMa trained on | RDD-LightGlue trained on |
|---|---|---|
| CzechLynx (time-closed) | LoMa-mined pairs | RDD-mined pairs |
| CzechLynx (time-open, unseen-identity protocol) | LoMa-mined pairs | LoMa-mined pairs |
| Hyena, Leopard, Sea star, Turtle | LoMa-mined pairs | LoMa-mined pairs |
| Nyala, Whale shark | RDD-mined pairs | RDD-mined pairs |
| Salamander | LoMa-mined pairs | RDD-mined pairs |

To train on the other source, mine with that backend and pass
`matcher_finetune.mined_by=loma|rdd` (see [Matcher fine-tuning](finetuning.md)).
