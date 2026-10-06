# Experiments

How to run mining, fine-tuning, single evaluations and sweeps, where the outputs go, how tables
and figures are made from them, and the reporting rules behind the numbers. Back to the
[README](../README.md); every setting is described in [CONFIGURATION.md](CONFIGURATION.md).

- [Mining and matcher fine-tuning](#mining-and-matcher-fine-tuning)
- [Single runs](#single-runs)
- [Sweeps](#sweeps)
- [Outputs and reporting](#outputs-and-reporting)
- [Research validity and parallel submissions](#research-validity-and-parallel-submissions)
- [Known constraints](#known-constraints)
- [Other tools](#other-tools)

## Mining and matcher fine-tuning

Pair mining and matcher fine-tuning need the `train` extra (see
[INSTALLATION.md](INSTALLATION.md#uv-recommended)). The project page's Reproduce pages walk through
both steps with the commands used for the paper: [mining](../docs/reproduce/mining.md),
[fine-tuning](../docs/reproduce/finetuning.md), [evaluation](../docs/reproduce/probing.md) and the
[few-shot views](../docs/reproduce/fewshot.md).

```bash
wildmatch mine plan --dataset salamander --backend loma        # view, cache, task and aggregate commands
wildmatch mine submit --dataset salamander --backend loma --dry-run
wildmatch finetune-matcher dataset=salamander matcher_finetune=rdd matcher_finetune.dry_run=true  # plan only
```

## Single runs

### Backbone fine-tuning

```bash
wildmatch finetune-backbone
```

Override configuration values with Hydra dotlist syntax:

```bash
wildmatch finetune-backbone train.epochs=10 train.batch_size=32
```

### Evaluation

```bash
wildmatch evaluate
```

Select methods and override nested settings with Hydra:

```bash
wildmatch evaluate benchmark.method=linear_probe
wildmatch evaluate benchmark.method=efficient_probe
wildmatch evaluate benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
```

### JaguarReID

Paused: the competition rules allow competition use only, and research use needs the sponsors'
written authorization (requested 2026-10-06). Do not run JaguarReID until it is granted; see
[DATASET.md](DATASET.md#55-jaguarreid-paused) and `THIRD_PARTY_LICENSES.md`.

## Sweeps

For grids of runs (method x checkpoint x candidate budget over registry datasets), write a
sweep spec and run it with `wildmatch sweep`. Packaged specs live in
`src/wildmatch/conf/sweep/`; `example.yaml` there documents every option.

```bash
wildmatch sweep parity --list-tasks              # print the task table; writes nothing
wildmatch sweep my_sweep.yaml --dry-run          # freeze the submission, print the sbatch command
wildmatch sweep my_sweep.yaml --submit           # Slurm array (slurm/sweep_task.sbatch), from the repo root
wildmatch sweep my_sweep.yaml --local            # run every task here, one after another
wildmatch sweep my_sweep.yaml --submit --max-concurrent 4 --sbatch-arg=--partition=gpu
```

A spec names registry `datasets`, `candidate_k` budgets and `variants` rows:

```yaml
datasets: [salamander]
candidate_k: [50, 250]
variants:
  - {method: cosine}
  - {method: vismatch, matcher: loma}                       # default weights
  - {method: vismatch, matcher: loma, checkpoint: custom}   # fine-tuned matcher (registry path)
  - {method: linear_probe, train_mode: classifier, class_weighting: weighted}
```

`inputs: paper` in a spec (or per dataset in `dataset_overrides`) runs on the tables the paper's
runs read instead of the current ones (WildlifeReID-10k: the team's masks instead of SAM 3).
Classifier probes run once, at the first budget. Fine-tuned rows take their checkpoint from the
dataset's registry entry (`custom` = matcher only, `descriptor-fine-tuned`, `joint-fine-tuned`)
unless the row or `dataset_overrides` names a path; a missing checkpoint fails before
submission. Every non-listing mode freezes an immutable submission under
`logs/parallel_run/submissions/<id>/` (config snapshot, spec copy, task table, manifest with
checkpoint SHA-256s); each task re-validates it before loading a model and mirrors its output
into `logs/parallel_run/<dataset>/<animal>/<split>/job-<id>/task-...{.out,.err,.combined.log,.json}`.
`logs/index.csv` is rebuilt after every task. Weighted and unweighted classifier rows map to
`inverse_frequency` and `none` and are shown in paper tables as, for example,
`frozen (weighted)`. Descriptor rows may be cross-species: set `checkpoint_owner` and
`evaluation_animal` in `dataset_overrides`; the owner must match an owner-identifiable path.

To inspect the task logs:

```bash
wildmatch summarize-logs --format markdown
wildmatch summarize-logs --dataset WildlifeReID-10k --status failed
wildmatch summarize-logs --method vismatch --matcher loma --format csv
```

Historical log files are not moved or rewritten.

## Outputs and reporting

New probe and finetune runs are stored under `experiments/` using dataset, split,
model, method, matcher, timestamp, and configuration-hash components. For example:

```text
experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/
  megadescriptor-l/vismatch/loma/20260813T142530Z_a1b2c3d4/
    config.snapshot.yaml
    run_manifest.json
    metrics.json
    timings.json
    visualizations/index.csv
    visualizations/contact_sheet_top1.png
```

Finetune run directories also contain canonical checkpoints and
`training_metrics.csv`. Each manifest records status, resolved configuration, git
commit, environment information, dataset sizes, metrics, timings, and artifact paths.
Failed runs are retained with status and error information.

Existing aggregate files remain active for compatibility:

- `results/.../train_metrics.csv`
- `benchmark_runs/benchmark_results.csv`
- `reports/runs.csv` (one row per modern run)

New visualizations are stored with their run under `visualizations/`. The local
`index.csv` connects query/database indices, identities, ranks, scores, correctness,
and image paths. Top-1 and failure contact sheets are generated when images are
available. Historical `visualizations/<run_id>/` folders are not migrated.

Summarize runs with:

```bash
wildmatch summarize-runs --dataset CzechLynx_v2
wildmatch summarize-runs --method vismatch --matcher loma
wildmatch summarize-runs --sort-by top_1 --format markdown

# Generate per-animal CVPR-ready LaTeX and audit CSV tables
wildmatch tables
wildmatch tables --animal BelugaID
wildmatch tables --animal CzechLynx --split-protocol split-time_open
wildmatch tables --detailed-comments  # opt in to provenance comments

# Generate CVPR-style accuracy-versus-candidate-budget figures
wildmatch figures
wildmatch figures --metric top_1
wildmatch figures --animal BelugaID --metric top_5 --formats png pdf
wildmatch figures --animal CzechLynx --split-protocol split-time_closed
```

For the standalone unseen-identity split, plotting automatically uses only
`k=10, 50, 100, 160`, matching its 160-image gallery. An explicit `--budgets`
list is restricted to those valid values for `unseen_eval_split`.

When completed descriptor runs exist, the plotting script also writes separate
`descriptor_rdd_*` and `descriptor_loma_*` figures. Use
`--descriptor-family rdd` or `--descriptor-family loma` to restrict them. These
figures never add descriptor results to the existing matcher series.

The paper-table exporter reads completed `experiments/` manifests directly. For
split-aware artifacts it creates separate files such as
`reports/paper_tables/CzechLynx_split-time_closed_main.tex` and
`CzechLynx_split-time_open_main.tex`; no combined closed/open table is generated.
For legacy artifacts without split provenance it retains the animal-only names.
Use `--split-protocol` to export one protocol explicitly. The main table uses
`candidate_k=50` and presents default
and fine-tuned rows together with same-budget gain arrows; the ablation table
uses `10, 50, 100, 250, 500, 1000`. Failed or incomplete runs are excluded,
and missing configurations are shown as `--`. LaTeX values are percentage
points, while companion CSV files retain the source fractional values. Paper tables
display the checkpoint source `custom` as `fine-tuned`; run-selection identities
remain unchanged. The backbone/model is also part of the run-selection identity and is shown
in a dedicated `Backbone` column, so cosine baselines such as MegaDescriptor-T/L and
DINOv2/L or DINOv3/L are never collapsed into one row. Older manifests without a model
field are labeled `unknown` rather than guessed. Full-gallery methods use `mAP`; shortlist-constrained WildFusion and Vismatch use
`mAP@k`. The generated tabular is wrapped in
`\resizebox{\linewidth}{!}{...}` so the wide ablation table fits a CVPR
column; the template must provide `graphicx` (the standard CVPR template does).
The main and ablation LaTeX fragments use a compact CVPR-style layout with method
sections, gray default rows, green fine-tuned rows, same-`k` delta arrows,
Top-1/5/10, balanced Top-1, and primary compute runtime. They intentionally omit
mAP, mAP@k, and total runtime from the typeset fragments to keep them readable;
the companion CSV files retain all metrics and timing fields for auditability.
For `unseen_eval_split`, the exporter uses only `k=10, 50, 100, 160`, matching
the 160-image evaluation gallery; the regular six-budget grid remains unchanged
for other splits.
The colored rows and arrows require the usual `xcolor` support in the manuscript
template.
By default, generated LaTeX omits timestamp, run-ID, and manifest comments; pass
`--detailed-comments` when those provenance comments are needed.
Include a generated table with `\input{reports/paper_tables/BelugaID_main.tex}`.

`wildmatch figures` reads completed probe artifacts directly from
`experiments/` and writes one multi-panel figure per requested metric under
`reports/figures/` (PNG and PDF by default). The default `paper` style uses a
color-blind-safe palette, redundant line/marker encodings, publication
typography, and the actual candidate budgets `10, 50, 100, 250, 500, 1000`
on a logarithmic x-axis with shared y-limits across panels. The default series
are WildFusion, LoMa default/fine-tuned, and RDD-LightGlue default/fine-tuned.
Use `--style presentation` for larger slide-friendly typography, or
`--style diagnostic --x-scale categorical --independent-y` for the previous
equally spaced, independently zoomed view. `--label-endpoints` optionally adds
direct labels to the final available point of each series. Missing runs are left
as gaps; failed or incomplete runs are ignored. Use `--metric balanced_top_1`
when a balanced-accuracy-only figure is needed, or `--metric all` for every
supported metric. Balanced Top-1 is included in the default metric set, so a
normal invocation also writes `balanced_top_1_vs_k` figures. Split-aware artifacts
are plotted separately: for example,
`split-time_closed` and `split-time_open` produce
`top_1_vs_k_split-time_closed.png` and `top_1_vs_k_split-time_open.png` rather
than being combined in one panel. Use `--split-protocol` to restrict the output
to one split; repeat it to request selected splits. Legacy artifacts without
split provenance retain the unsuffixed filenames.

Use `--exclude-method wildfusion`, `--exclude-method rdd`, or
`--exclude-method loma` to remove a method family from the plot. The filter
removes both default and fine-tuned matcher series; repeat the option to omit
multiple families. RDD and LoMa descriptor figures are filtered consistently.

Paper-table runtime uses the primary compute phase: pairwise matcher time for
Vismatch, WildFusion, and Local LightGlue, and method-computation time for
cosine and classifier probes. Total wall-clock runtime is shown separately.
Matcher timing excludes Stage-A selection, feature extraction, model setup,
calibration, cache I/O, and visualization. Historical runs without the new
timing fields show `--` for primary compute runtime and must be rerun before
making matcher-speed claims.

For compatibility, tagged model-only checkpoints remain readable. Automatic probe
discovery searches the new `experiments/finetune/` root first and then historical
`results/`; explicit checkpoint paths always take precedence.

## Research validity and parallel submissions

[AGENTS.md](../AGENTS.md) is the single source for these rules; read its "Research-validity reporting
policy" and "Immutable parallel probe submissions" sections. In short:

- Ties are broken by original database index everywhere (metrics, shortlists, visualizations).
- Shortlist methods (Vismatch, WildFusion) report `mAP_at_k`; full-matrix `mAP` is emitted only
  when every gallery position is scored, and is `nan` otherwise.
- `recall_at_k`, `rerank_mAP_at_k` and `mAP_at_k` separate shortlist reach from matcher ordering.
- Every probe run writes `scores.npz`, so metrics can be recomputed without rerunning a matcher.
- Feature caches are keyed on image content, metadata, preprocessing, image variant and model
  or checkpoint weights, plus the mask of each image when masks are applied at load time
  (CzechLynx); Vismatch caches also record the GPU type, because its features depend on it.
- Sweeps snapshot each submission (config copy, spec, task table, manifest) under
  `logs/parallel_run/submissions/`; tasks read only that snapshot, and custom checkpoints are
  checked for owner and SHA-256 before loading.

## Known constraints

- Model weights are downloaded from Hugging Face on first use.
- Vismatch requires the pinned package, model-weight downloads, and usually CUDA for practical runtimes.
- Machine-specific locations come from a path profile; the `default` profile uses relative
  `data/`, `cache/` and `checkpoints/` (see [INSTALLATION.md](INSTALLATION.md#data-locations)).
- Masking and Vismatch matcher settings are dataset-dependent and should be validated rather than
  assumed to improve every dataset. Matcher ablations must keep Stage-A candidates, preprocessing,
  keypoint budgets, scoring, and evaluation metrics fixed.
- The test suite intentionally avoids CUDA, downloaded models, private datasets, and external
  Vismatch integration; optional environment-gated smoke and Lynx parity checks are required
  before changing the pinned Vismatch commit.
- Future experiment priorities are tracked in [AGENTS.md](../AGENTS.md).

## Other tools

Paper tables and figures (all read completed runs under `experiments/`):

- `wildmatch tables`: per-animal LaTeX and audit CSV tables.
- `wildmatch figures`: accuracy against candidate budget k.
- `paper/figures/plot_training_cost.py`: test accuracy against training GPU-hours (CzechLynx closed).
- `slurm/eval_loma_epoch_curve.sh`: Slurm array that evaluates intermediate LoMa checkpoints for that plot.
- `paper/figures/plot_match_examples.py`: qualitative figure, one fine-tuned LoMa match per dataset.
- `paper/figures/plot_data_quality_examples.py`: confirmed low-quality examples as raw photos.
- `wildmatch class-balance`: per-identity image counts and imbalance statistics.
- `wildmatch audit`: heuristic low-quality image flags for review by eye.

Data preparation (step by step in [DATASET.md](DATASET.md)):

- `wildmatch prepare status|download|build|retry-empty|finish|compare-masks|unseen-split|jaguar`: the dataset recipes.
- `wildmatch build-unseen-split`: unseen-identity gallery/query split.
- `src/wildmatch/data/prepare/sam3_masks.py` (via `slurm/sam3_masks.sbatch`): SAM 3 background removal and pre-masked metadata (its own SAM 3 environment, see DATASET.md; any GPU but V100).
- `python -m wildmatch.data.prepare.jaguar`: brings the Kaggle Jaguar training data into the shared format (`JaguarReID`: `prepare` writes masked images and RLE masks, `embed` DINOv2 embeddings, `split` the burst-aware `split_v2` database/query split).

Run and log inspection:

- `wildmatch summarize-runs`: filter and sort `reports/runs.csv`.
- `wildmatch summarize-logs`: list sweep task logs; `--write-index` rebuilds `logs/index.csv`.

Project page exporters in `paper/page/` (write committed files under `docs/`; GPU where noted;
run as `python paper/page/<name>.py`):

- `export_project_page_data.py`: results, curves and page figures from the paper's results snapshot.
- `export_score_separation.py`, `export_frequency_bins.py`, `export_budget_tradeoff.py`: the "why it works" views.
- `export_before_after_demo.py` (GPU), `export_rank_change_demo.py`, `export_mined_pairs_demo.py` (GPU), `export_masking_demo.py`, `export_synthetic_demo.py` (GPU): the demos.
- `export_data_challenges.py`: the data-challenges galleries.
- `build_demo_cards.py`: Demo hub thumbnails.
- `paper/figures/analyze_background_matches.py` (GPU): analysis only, writes to `reports/`.

The project page builds with `mkdocs build --strict` (dependencies in
`requirements-docs.txt`). The explainer video is built from `video/explainer/`; see
`video/explainer/ENVIRONMENT.md`. AGENTS.md ("Project page") holds the page's rules and
publication checklist.
