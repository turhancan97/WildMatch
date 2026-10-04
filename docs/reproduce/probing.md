# Evaluation

Repository: `explainable_individual_reidentification` (this one), installed as the `wildmatch`
package with one `wildmatch` command. Evaluation is a Hydra-configured *probe*: for every query it ranks the gallery with one method, stores the
score matrix, metrics and timings in a self-contained run directory, and appends one row
to a central run index. Tables and figures are built from those run directories.

## Environment

```bash
uv sync --extra cu126 --extra matchers --group dev   # or --extra cpu; Python 3.12
source .venv/bin/activate
```

A conda route installs the same pinned versions (`environment.yml`, then
`pip install --no-deps -r requirements/cu126.txt`).

The matchers are wrapped through Vismatch, pinned to a fixed commit, which downloads the
pretrained matcher weights on first use. DINOv3 weights are gated on Hugging Face and need
an accepted licence.

## Data contract

Each dataset is an entry of the dataset registry (`src/wildmatch/conf/dataset/<key>.yaml`,
selected with `dataset=<key>`): a root directory plus a metadata CSV with an image path
(relative to the root), an identity column and a split column. `wildmatch prepare status`
shows what each entry needs on disk. For masked inputs the CSV points at the
pre-masked images, or carries a COCO run-length mask applied at load time (CzechLynx). The
eight paper datasets use:

| Dataset | Metadata | Split values | Masking |
|---|---|---|---|
| CzechLynx | dataset metadata CSV, `split-time_closed` column | `train` / `test` | RLE mask applied at load time |
| Hyena, Leopard, Nyala, Sea star, Whale shark, Turtle | WildlifeReID-10k tables with the closed-set split of WildFusion (every individual on both sides) | `train` / `test` | pre-masked images (see below) |
| Salamander | `split_time_closed_no_background.csv`: the latest capture date of each individual is the query | `database` / `query` | SAM 3 pre-masked images |

The WildlifeReID-10k masked images used for the paper's runs were made by the team, not shipped
with the dataset. The released pipeline rebuilds the same split tables exactly and masks the
images with SAM 3 (`wildmatch prepare build`, then `finish`); the new masks agree closely with
the old ones (median IoU 0.98 to 0.999 per dataset), so results can differ slightly from the
paper's.

## One run

```bash
# Cosine retrieval, full gallery, MegaDescriptor-L (default backbone); add dataset=<key>
wildmatch evaluate benchmark.method=cosine

# Default LoMa over the MegaDescriptor-L candidate list, k = 250
wildmatch evaluate benchmark.method=vismatch \
  benchmark.methods.vismatch.matcher=loma benchmark.candidate_k=250

# Fine-tuned LoMa (matching module only)
wildmatch evaluate benchmark.method=vismatch \
  benchmark.methods.vismatch.matcher=loma \
  benchmark.methods.vismatch.checkpoint_source=custom \
  benchmark.methods.vismatch.checkpoint_path=<checkpoints>/<dataset>/loma/model.safetensors \
  benchmark.methods.vismatch.checkpoint_components=matcher_only \
  benchmark.candidate_k=250

# WildFusion over the same candidates
wildmatch evaluate benchmark.method=wildfusion benchmark.candidate_k=250

# Class-weighted identity classifier, backbone fully fine-tuned
wildmatch evaluate benchmark.method=linear_probe \
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

`wildmatch sweep` builds a method-by-budget grid from a YAML spec (registry datasets,
candidate budgets, method rows) and runs it locally or as an immutable Slurm array job: the
submission copies the configuration, hashes every checkpoint, and each task reads that frozen
manifest instead of the editable files. Fine-tuned rows take their checkpoint from the dataset
registry, filled by `wildmatch weights download`.

```bash
wildmatch sweep example --list-tasks       # annotated example spec (src/wildmatch/conf/sweep/)
wildmatch sweep my_grid.yaml --local       # run every task here
wildmatch sweep my_grid.yaml --submit      # Slurm array (slurm/sweep_task.sbatch)
```

Budgets are \(k \in \{10, 50, 100, 250, 500, 1000\}\) (and \(\{10, 50, 100, 160\}\) for the
unseen protocol); classifier probes ignore the budget and run once.

## Unseen-identity protocol

```bash
wildmatch prepare unseen-split   # writes metadata_unseen_eval.csv + manifest
```

The generator selects the identities absent from the time-open training part, groups their
images by encounter, assigns the earliest encounter to the gallery and later ones to the
queries, hashes every file, and fails on path overlap or duplicate content between the two
sides. The registry entry `czechlynx_unseen_eval` evaluates it.

## Metrics

Top-k accuracy at the identity level and balanced Top-1 (Top-1 averaged over individuals).
Ranking uses descending score with the original gallery index as tie-breaker everywhere.
Full-gallery methods also report mAP; shortlist methods report mAP@k, which gives no credit
to unscored positions. Every run stores its finite scores so metrics can be recomputed.

## Tables and figures

```bash
wildmatch tables                                   # per-dataset main and ablation tables
wildmatch figures                                  # accuracy-versus-k figures
python paper/figures/plot_training_cost.py --candidate-k 250 \
  --metrics balanced_top_1,top_5 --output-stem training_cost_k250_balanced_top5
python paper/figures/plot_match_examples.py render         # qualitative matches
python paper/figures/plot_data_quality_examples.py         # challenging images
wildmatch class-balance                            # dataset statistics
python paper/page/export_project_page_data.py      # this page's data and figures
```

The exporters select the newest completed run for each dataset, method, checkpoint,
backbone and budget, keep descriptor-only and joint runs in their own tables, and never read
the aggregate CSV.
