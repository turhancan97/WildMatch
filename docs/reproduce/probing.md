# Evaluation

Repository: `explainable_individual_reidentification` (this one). Evaluation is a
Hydra-configured *probe*: for every query it ranks the gallery with one method, stores the
score matrix, metrics and timings in a self-contained run directory, and appends one row
to a central run index. Tables and figures are built from those run directories.

## Environment

```bash
conda create -n ex-reid python=3.11 && conda activate ex-reid
pip install -r requirements.txt
```

The matchers are wrapped through Vismatch, pinned to a fixed commit, which downloads the
pretrained matcher weights on first use. DINOv3 weights are gated on Hugging Face and need
an accepted licence.

## Data contract

Each dataset is a root directory plus a metadata CSV with an image path (relative to the
root), an identity column and a split column. For masked inputs the CSV points at the
pre-masked images, or carries a COCO run-length mask applied at load time (CzechLynx). The
eight paper datasets use:

| Dataset | Metadata | Split values | Masking |
|---|---|---|---|
| CzechLynx | dataset metadata CSV, `split-time_closed` column | `train` / `test` | RLE mask applied at load time |
| Hyena, Leopard, Nyala, Sea star | WildlifeReID-10k `metadata_mdsplit_no_background/metadata_<dataset>.csv` | `train` / `test` | official pre-masked images |
| Whale shark, Turtle | WildlifeReID-10k `metadata_no_background/metadata_<dataset>.csv` | `train` / `test` | official pre-masked images |
| Salamander | `split_time_closed_no_background.csv` written by `scripts/segment_with_sam3.py` | `database` / `query` | SAM 3 pre-masked images |

## One run

```bash
# Cosine retrieval, full gallery, MegaDescriptor-L (default backbone)
python train/probe.py benchmark.method=cosine

# Default LoMa over the MegaDescriptor-L candidate list, k = 250
python train/probe.py benchmark.method=vismatch \
  benchmark.methods.vismatch.matcher=loma benchmark.candidate_k=250

# Fine-tuned LoMa (matching module only)
python train/probe.py benchmark.method=vismatch \
  benchmark.methods.vismatch.matcher=loma \
  benchmark.methods.vismatch.checkpoint_source=custom \
  benchmark.methods.vismatch.checkpoint_path=<checkpoints>/loma-b-finetuned-loma-mined-legacy/epoch_299/model.safetensors \
  benchmark.methods.vismatch.checkpoint_components=matcher_only \
  benchmark.candidate_k=250

# WildFusion over the same candidates
python train/probe.py benchmark.method=wildfusion benchmark.candidate_k=250

# Class-weighted identity classifier, backbone fully fine-tuned
python train/probe.py benchmark.method=linear_probe \
  benchmark.methods.linear_probe.train_mode=all \
  benchmark.methods.linear_probe.class_weighting=inverse_frequency
```

`benchmark.candidate_k` is the single budget setting: it sets how many MegaDescriptor-L
candidates the matcher scores and the cutoff of the shortlist-aware metrics. Descriptor and
joint checkpoints are loaded with `checkpoint_components=descriptor_only` and `full`; the
loader validates the checkpoint's protocol file and refuses a mismatched component mode.
Unscored gallery positions are set to minus infinity, so a matcher can never rank an image
it did not score.

## Benchmark grids

Two Slurm launchers build the full method-by-budget grid for a dataset and submit it as an
immutable array job: the launcher copies the configuration, hashes every checkpoint, and
each task reads that frozen manifest instead of the editable files.

```bash
bash probe-parallel-czechlynx.sh --list-tasks   # closed split, unseen protocol
bash probe-parallel-wildlife.sh --list-tasks    # one WildlifeReID-10k dataset or Salamander
bash probe-parallel-wildlife.sh                 # submit
```

The active rows of each launcher's `VARIANTS` table and its candidate list are the grid.
Budgets are \(k \in \{10, 50, 100, 250, 500, 1000\}\) (and \(\{10, 50, 100, 160\}\) for the
unseen protocol); classifier probes ignore the budget and run once.

## Unseen-identity protocol

```bash
python scripts/build_unseen_eval_metadata.py   # writes metadata_unseen_eval.csv + manifest
```

The generator selects the identities absent from the time-open training part, groups their
images by encounter, assigns the earliest encounter to the gallery and later ones to the
queries, hashes every file, and fails on path overlap or duplicate content between the two
sides. The CzechLynx launcher has an opt-in profile that evaluates it.

## Metrics

Top-k accuracy at the identity level and balanced Top-1 (Top-1 averaged over individuals).
Ranking uses descending score with the original gallery index as tie-breaker everywhere.
Full-gallery methods also report mAP; shortlist methods report mAP@k, which gives no credit
to unscored positions. Every run stores its finite scores so metrics can be recomputed.

## Tables and figures

```bash
python scripts/export_paper_tables.py              # per-dataset main and ablation tables
python scripts/plot_paper_figures.py               # accuracy-versus-k figures
python scripts/plot_training_cost.py --candidate-k 250 \
  --metrics balanced_top_1,top_5 --output-stem training_cost_k250_balanced_top5
python scripts/plot_match_examples.py render       # qualitative matches
python scripts/plot_data_quality_examples.py       # challenging images
python scripts/export_class_balance.py             # dataset statistics
python scripts/export_project_page_data.py         # this page's data and figures
```

The exporters select the newest completed run for each dataset, method, checkpoint,
backbone and budget, keep descriptor-only and joint runs in their own tables, and never read
the aggregate CSV.
