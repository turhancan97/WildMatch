# Few-shot views

!!! note "Extension, not part of the paper"
    This page describes tooling for studying how fine-tuning behaves with less labelled data. The
    paper reports no few-shot experiments, and this page shows no results.

A few-shot view is a copy of a dataset's training split that keeps only a fraction of the
training images; the test split stays unchanged. Pair mining, fine-tuning and evaluation then run
on that view exactly as on the full one ([mining](mining.md), [fine-tuning](finetuning.md),
[evaluation](probing.md)), with one extra option each.

## How a view is built

| Rule | Effect |
|---|---|
| Exact budget | the training split keeps `round(fraction × N_train)` images in total |
| Proportional | the budget is shared across individuals in proportion to their image counts |
| Minimum of two | an individual never drops below two training images, so every remaining image still has a positive; the other individuals share what is left of the budget |
| Nested | the images kept for an individual are a prefix of one seeded permutation, so for one seed the 1/8 view lies inside the 1/4 view, which lies inside the 1/2 view |
| Same frame names | images keep their names from the full view, so the full view's feature caches serve every few-shot view |

Individuals with a single image keep it: they never yield a training pair, but they stay in the
gallery as in the full view. When the two-image minimum alone exceeds the budget, the view keeps
more than the requested fraction. Datasets with many small individuals reach this limit early;
SalamanderID2025, for example, keeps 70 % of its training images for any fraction up to 1/2. The
view's `fewshot.json` records the effective fraction and `budget_feasible`.

CzechLynx few-shot views exist for the time-closed split only.

## Mining

The full view and its feature cache must exist first (see [Pair mining](mining.md)). `--fraction`
and `--seed` then select the few-shot view in every `wildmatch mine` step:

```bash
wildmatch mine plan   --dataset hyenaid2022 --backend loma --fraction 0.25 --seed 0
wildmatch mine view   --dataset hyenaid2022 --backend loma --fraction 0.25 --seed 0
wildmatch mine submit --dataset hyenaid2022 --backend loma --fraction 0.25 --seed 0
```

Views, indices, checkpoints and metadata go to their own folders under the path profile's
`fewshot_root` (by default `<data_root>/fewshot`, or `WILDMATCH_FEWSHOT_ROOT`):
`{views,indices,checkpoints,metadata}/<dataset>/<protocol>/frac<F>-seed<S>/`.

## Fine-tuning

```bash
sbatch slurm/finetune_matcher.sbatch dataset=hyenaid2022 matcher_finetune=loma \
    matcher_finetune.fewshot.fraction=0.25 matcher_finetune.fewshot.seed=0
```

The run reads the few-shot view and its mined index and writes to
`<fewshot_root>/checkpoints/<dataset>/<protocol>/frac<F>-seed<S>/<matcher>-finetuned/`. The recipe
is the paper's (see [Matcher fine-tuning](finetuning.md)).

## Evaluation

Building a view also writes a copy of the dataset's metadata with one extra column,
`split_frac<F>_seed<S>`: `train` for the kept training images, `unused` for the dropped ones, and
`test` unchanged. Evaluating with that column puts only the kept images in the gallery, which
models a smaller labelled collection:

```bash
wildmatch evaluate dataset=hyenaid2022 \
  dataset.metadata_file=<fewshot_root>/metadata/HyenaID2022/legacy/frac0.25-seed0/metadata.csv \
  dataset.split_col=split_frac0.25_seed0 \
  dataset.database_split_value=train dataset.query_split_value=test \
  benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma \
  benchmark.methods.vismatch.checkpoint_source=custom \
  benchmark.methods.vismatch.checkpoint_path=<fewshot_root>/checkpoints/HyenaID2022/legacy/frac0.25-seed0/loma-finetuned/epoch_299/model.safetensors \
  benchmark.methods.vismatch.checkpoint_components=matcher_only \
  benchmark.candidate_k=250
```

The metadata path makes the run's cache and run identity separate from the full dataset's.
