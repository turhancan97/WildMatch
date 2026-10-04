# Explainable Individual Re-Identification

Modular deep learning codebase for wildlife individual re-identification (ReID), with:
- backbone finetuning (`train/finetune.py`)
- retrieval probing / benchmarking (`train/probe.py`)
- optional mask-based background removal
- optional Weights & Biases logging

## Overview

The repository supports two workflows:
- `finetune`: train a backbone with ArcFace loss on a train split and evaluate retrieval on a validation split.
- `probe`: benchmark retrieval methods (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `vismatch`) with pretrained or finetuned backbones.

The code is the installable `wildmatch` package under `src/wildmatch/`; `train/probe.py` and
`train/finetune.py` are thin wrappers around its entry points. (The package refactor is in
progress on this branch; see AGENTS.md, "Package refactor".)

## Repository Structure

```text
.
├── pyproject.toml, uv.lock      package metadata and the locked environment (uv)
├── requirements/          pinned pip/conda requirements exported from uv.lock
├── environment.yml        conda route (installs requirements/cu126.txt)
├── src/wildmatch/         the package
│   ├── conf/             Hydra configs: probe.yaml, finetune.yaml
│   ├── entrypoints.py    Hydra entry points for evaluation (probe) and backbone fine-tuning
│   ├── data/             dataset views, COCO-RLE masking, split safety checks
│   ├── evaluate/         probe runner, metrics, stable ranking, candidate scoring
│   ├── features/         feature containers
│   ├── matchers/         Vismatch (profiles, batching, checkpoints), WildFusion calibration
│   ├── models/           backbone factory, ViT CLS adapter, training objectives
│   ├── reporting/        manifests, run index, paper tables and figures, W&B names
│   ├── train/            backbone fine-tuning runner, checkpointing, accumulation, class weights
│   ├── utils/            I/O, fingerprints, cache identities, reproducibility, config defaults
│   └── mining/, matcher_finetune/   slots for pair mining and matcher fine-tuning (merged in later)
├── train/                thin wrappers: finetune.py, probe.py
├── scripts/              analysis, export and plotting tools (see below)
├── paper/tools/          parity check for the refactor
├── tests/                unit tests (pytest)
├── notebooks/            dataset annotation viewer (outputs stripped)
├── probe.sh, finetune.sh      single-job Slurm wrappers
├── probe-parallel-czechlynx.sh, probe-parallel-wildlife.sh   Slurm array launchers
├── mkdocs.yml, docs/, overrides/   project page (MkDocs Material)
├── video/explainer/      explainer video sources (Manim + Kokoro, wm-video env)
├── AGENTS.md             operating guide, decisions and open work
├── notes/                narrative history and the refactor's parity reference
└── CHANGELOG.MD          chronological record of changes
```

Generated and ignored: `experiments/` (one directory per run), `reports/` (run index,
paper tables, figures), `logs/` (Slurm and Hydra logs), `benchmark_runs/`,
`wandb/`, `site/` (page build) and `dataset/` (symlinks to the shared
datasets). `results/`, `cache/` and `visualizations/` appear only when old workflows run.

## Installation

Python 3.12. Pick exactly one torch build: `cu126` (CUDA 12.6) or `cpu`.

### uv (recommended)

```bash
uv sync --extra cu126 --extra matchers --group dev   # or --extra cpu
uv run python train/probe.py --help
```

`--extra matchers` adds Vismatch (RDD-LightGlue, LoMa and the other local matchers);
`--extra wandb` adds Weights & Biases logging. `uv.lock` pins every package, including the
git dependencies (wildlife-tools, wildlife-datasets, Vismatch, glue-factory, LightGlue).
uv creates `.venv/` in the repository unless `UV_PROJECT_ENVIRONMENT` points elsewhere,
for example to keep large environments off a small home directory:

```bash
export UV_PROJECT_ENVIRONMENT=/path/with/space/wildmatch
```

### Conda

```bash
conda env create -f environment.yml                  # Python 3.12 and pip
conda activate wildmatch
pip install --no-deps -r requirements/cu126.txt      # or requirements/cpu.txt
pip install pytest                                   # only to run the tests
```

The requirement files pin every package; `requirements/export.sh` generates them from
`uv.lock`, so both routes install the same versions. `--no-deps` is required: a normal pip
resolve fails because glue-factory asks for an unpinned LightGlue git URL while the lock pins
a commit. `pip check` then reports only `uniception`, which is excluded on purpose.

## Supported Models

Supported model identifiers:

- megadescriptor-t
- megadescriptor-l (the default)
- lynx_megadescriptorV3
- lynx_megadescriptorV4
- miewid
- dinov2 (ViT-S/14 with registers)
- dinov2-l (ViT-L/14 with registers)
- dinov3 (ViT-S+/16)
- dinov3-l (ViT-L/16)

The legacy name megadescriptor is not accepted. Use an explicit supported
identifier in custom configurations.

The WildlifeReID-10k analysis profiles currently cover NyalaData, WhaleSharkID,
BelugaID, ZindiTurtleRecall, ATRW, Giraffes, LeopardID2022, HyenaID2022,
GiraffeZebraID, CowDataset, StripeSpotter, and SeaStarReID2023. The added
profiles use `metadata_mdsplit_no_background/metadata_<animal>.csv`, with
`identity` as the label column and `split` values `train` and `test`. These
metadata paths point to the corresponding pre-masked `masked_images/` tree, so
their profile uses `image_variant: no_background` and `no_background: false`.


## Dataset Requirements

Configs assume a dataset root containing metadata CSV with split/label columns.

Default expected fields:
- `unique_name` (identity label)
- split column:
  - probe: `split-time_closed`
  - finetune: `split-time_closed`
- optional `mask` column for background removal (`dataset.no_background: true`)

Default path in configs:
- `/shared/sets/datasets/vision/czechlynx/CzechLynx_v2`

## Quick Start

### Finetune

```bash
python train/finetune.py
```

Override configuration values with Hydra dotlist syntax:

```bash
python train/finetune.py train.epochs=10 train.batch_size=32
```

### Probe / Benchmark

```bash
python train/probe.py
```

Select methods and override nested settings with Hydra:

```bash
python train/probe.py benchmark.method=linear_probe
python train/probe.py benchmark.method=efficient_probe
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
```

For the reproducible Slurm ablation grid, use the separate launcher:

```bash
# Submit the CzechLynx task table.
bash probe-parallel-czechlynx.sh

# Submit the WildlifeReID-10k task table.
bash probe-parallel-wildlife.sh

# Inspect either task table without submitting jobs.
bash probe-parallel-czechlynx.sh --list-tasks
bash probe-parallel-wildlife.sh --list-tasks

# Print an array submission command without submitting it.
PROBE_PARALLEL_DRY_RUN=1 bash probe-parallel-wildlife.sh
```
The wildlife launcher includes ready-to-activate profiles for ATRW, Giraffes,
LeopardID2022, HyenaID2022, GiraffeZebraID, CowDataset, StripeSpotter, and
SeaStarReID2023 in addition to the existing WildlifeReID-10k animals. Activate
exactly one profile at a time; each new profile expects the corresponding
`legacy/epoch_299/model.safetensors` LoMa and RDD checkpoint paths; several of these
folders have since been renamed (see AGENTS.md, "Known issues"). The launcher also carries
commented SalamanderID2025 and JaguarReID profiles.

The CzechLynx launcher contains separate `split-time_closed` and
`split-time_open` profiles. The closed profile is active by default; uncomment
the open profile to run it as well in the same array. Both profiles use the same
active `VARIANTS` and candidate grid, but each has independent custom LoMa/RDD
checkpoint settings. The open profile defaults to
`/shared/sets/datasets/vision/czechlynx/checkpoints/czechlynx-time-open`, epoch
`299`, with `loma-b-finetuned-legacy/epoch_299/model.safetensors` and
`rdd-finetuned-legacy/epoch_299/model.safetensors`. These paths can be overridden
by the `CZECHLYNX_OPEN_*` variables, but are never inferred from the closed-split
checkpoint root. Logs and task metadata
include the split name, and the resulting experiment paths are split-specific.

`probe-parallel-czechlynx.sh` and `probe-parallel-wildlife.sh` leave `probe.sh`
unchanged and provide separate task tables for CzechLynx and WildlifeReID-10k.
Each launcher crosses its active variants with the candidate budgets listed in
`CANDIDATE_K_VALUES`; its `VARIANTS` table is the source of truth for which methods
run. The CzechLynx launcher may have one or both split profiles active; the
wildlife launcher uses one active animal profile. The concurrency cap is
controlled by `MAX_CONCURRENT_JOBS` near the top of the selected file. Custom
Vismatch checkpoint paths are editable there; matcher-fine-tuned variants use
`checkpoint_components=matcher_only` and descriptor-fine-tuned variants use
`checkpoint_components=descriptor_only`, while default variants use Vismatch-managed
weights. Descriptor rows are opt-in and the selected launcher fails before submission if a custom checkpoint is
missing. Slurm's raw stdout and stderr remain under `logs/parallel_run/`, while each task also creates descriptive copies under
`logs/parallel_run/<dataset>/<animal>/<split_protocol>/job-<array_job>/`. Files are
named with the task index, split, method, matcher, checkpoint, and candidate budget. Each task writes
`.out`, `.err`, `.combined.log`, and a JSON metadata record containing
its command, status, timestamps, error summary, and experiment-run link.
`logs/index.csv` is updated atomically as tasks start and finish. The launcher uses
Slurm's `SLURM_SUBMIT_DIR`, so it remains valid even though Slurm executes a copied
script from its private spool directory. Use
`MAX_CONCURRENT_JOBS=2 bash probe-parallel-wildlife.sh` to change the throttle.
For `linear_probe` and `efficient_probe` comparisons, add launcher rows ending in
`|weighted` or `|unweighted`. These become `inverse_frequency` and `none` Hydra
overrides, respectively, and are included in task names, commands, manifests, and
logs. The generated paper tables also show the policy in the checkpoint cell, for
example `frozen (weighted)` and `frozen (unweighted)`, so paired classifier rows
cannot be confused. The audit CSV retains the separate `class_weighting` column.

Descriptor profiles may be cross-species: set the evaluation animal and checkpoint
owner independently. The launcher requires the checkpoint path to contain the
declared owner in the WildlifeReID-10k layout, so an accidental owner/path mismatch
fails before model loading or cache creation.

To inspect the organized logs:

```bash
python scripts/summarize_logs.py --format markdown
python scripts/summarize_logs.py --dataset WildlifeReID-10k --status failed
python scripts/summarize_logs.py --method vismatch --matcher loma --format csv
```

Historical log files are not moved or rewritten; the descriptive layout applies to
future parallel tasks.

### Jaguar (JaguarReID)

The Kaggle Jaguar Re-ID training photos run through the shared probe pipeline as
`JaguarReID`: prepare them once with `scripts/prepare_jaguar_metadata.py`, then activate the
`jaguar` profile of `probe-parallel-wildlife.sh`. See AGENTS.md, "JaguarReID".

## Configuration Guide

Hydra is the primary configuration interface for probe and finetuning. The shipped
defaults are in `src/wildmatch/conf/probe.yaml` and `src/wildmatch/conf/finetune.yaml`; Hydra resolves their
interpolations before the runner starts. Nested overrides use `key=value` dotlist
syntax and values are type-converted by OmegaConf.

Unknown keys and misspelled paths fail immediately. The old `--config`, `--method`,
`--dataset-root`, and related argparse flags are no longer supported for these two
entrypoints.

Hydra is configured not to change the working directory or replace project-managed
artifact paths. Probe and finetune continue writing their normal run directories and
store a fully resolved `config.snapshot.yaml` in each run. Hydra multirun sweeps are
not part of the supported experiment workflow.

### `src/wildmatch/conf/finetune.yaml`

Key blocks:
- `dataset`: root, metadata file, split values, `no_background`, `mask_col`, and `image_variant` (`background` or `no_background`)
- `model`: backbone type
- `train`: epochs, batch size, AMP, deterministic mode, resume checkpoint
- `loss`: ArcFace parameters
- `scheduler`: cosine settings
- `output`: experiment root, save frequency, best metric, and legacy aggregate CSV path
- `reporting`: run-index path and reporting enablement
- `benchmark`: validation retrieval metrics (`top_k`, `mAP`)
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

### `src/wildmatch/conf/probe.yaml`

Key blocks:
- `dataset`: root/splits + mask options and explicit `image_variant` (`background` or `no_background`)
- `model`: type/mode/checkpoint behavior
- `benchmark`: method (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `vismatch`), metrics, cache
- `benchmark.candidate_k`: single comparison budget (default `100`) used for Vismatch
  candidates, WildFusion refinement, and the `mAP_at_k`, `rerank_mAP_at_k`, and
  `recall_at_k` evaluation cutoff.
- `benchmark.classifier_evaluation.open_set_policy`: `open` (default), `warn`, or
  `closed`, controlling how `linear_probe` and `efficient_probe` handle query
  identities absent from the training/database mapping.
- `benchmark.classifier_evaluation.embedding_retrieval`: optional `false`/`true`
  cosine retrieval diagnostic from the final backbone representations.
- Here, `k` is the number of gallery candidates retained for the expensive second
  stage. A larger `k` can recover identities missed by a smaller shortlist, but
  increases computation. In the paper tables, `k=--` means the method uses the
  full gallery rather than a shortlist, such as cosine.
- WildFusion settings: `local_batch_size` controls pair-processing batches and
  `local_top_k` controls ALIKED keypoints (default `512`). Its refinement `B` is derived
  from `benchmark.candidate_k`.
- Local LightGlue also receives its refinement budget from `benchmark.candidate_k`; its
  old method-specific `B` override is no longer supported.
- `visualization`: optional qualitative retrieval plots
- `output`: experiment root, legacy run folder, and aggregate CSV
- `reporting`: central run-index path
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

### Safety Checks

When `safety_checks.enabled: true`, both `finetune` and `probe` run pre-run validators before model loading:
- overlap check between split files (hard error)
- identity coverage report (seen/unseen identities)
- per-split class count histogram

Artifacts are saved under each run folder:
- `safety_checks/summary.json`
- `safety_checks/class_counts.csv`
- `safety_checks/class_count_histogram.png`

Classifier-based probe methods (`linear_probe`, `efficient_probe`) support open-world
evaluation. The default `benchmark.classifier_evaluation.open_set_policy: open`
maps query identities absent from the training/database identity set to an internal
sentinel: they remain in the predictions, count as incorrect classification results,
and contribute zero recall to balanced Top-1. Their samples are skipped only for
validation loss because there is no classifier target for them. Set the policy to
`warn` to retain the legacy seen-only `classification_top_*` fields while also
emitting all-query `classification_open_*` fields, or to `closed` to fail before
model construction. Safety checks are enabled by default and report the coverage
gap in every case.

The classifier coverage fields include seen/unseen query images and identities plus
`classification_query_seen_coverage`. When
`benchmark.classifier_evaluation.embedding_retrieval: true`, an optional cosine
retrieval diagnostic is reported under `embedding_*`; it is separate from the
classifier-head metrics and is disabled by default. No unknown classifier class is
added.

#### Building an unseen-identity evaluation split

For an evaluation-only open-world test, the standalone
`scripts/build_unseen_eval_metadata.py` utility can derive a new metadata CSV
without changing the probe pipeline or either parallel launcher. It selects
identities that occur in the source query split but not in the source database
split, assigns the earliest encounter to `database`, and assigns later
encounters to `query` using the supplied order column. It never performs a
random image-level split. For example, for CzechLynx:

```bash
python scripts/build_unseen_eval_metadata.py \
  --metadata /shared/sets/datasets/vision/czechlynx/CzechLynx_v2/CzechLynxDataset-Metadata-Real.csv \
  --output-dir /path/to/czechlynx-unseen-eval \
  --label-col unique_name \
  --source-split-col split-time_open \
  --database-value train \
  --query-value test \
  --group-col encounter \
  --order-col date \
  --root /shared/sets/datasets/vision/czechlynx/CzechLynx_v2
```

The output contains `metadata_unseen_eval.csv` and
`unseen_eval_manifest.json`. The manifest records source/output SHA-256 hashes,
selected and excluded identities, grouping parameters, path-overlap checks, and
cross-split duplicate-content checks. Missing files, malformed grouping/order
values, path overlap, or duplicate image content fail closed. Archive both
outputs with the resulting experiment. Consume the generated CSV through the
normal Hydra contract:

```bash
python train/probe.py \
  dataset.name=CzechLynx_v2 dataset.animal=CzechLynx \
  dataset.metadata_file=/path/to/czechlynx-unseen-eval/metadata_unseen_eval.csv \
  dataset.split_col=unseen_eval_split \
  dataset.database_split_value=database dataset.query_split_value=query \
  dataset.no_background=true dataset.image_variant=no_background \
  benchmark.candidate_k=100
```

Use the default and fine-tuned checkpoints as separate runs with identical
generated gallery/query metadata. The generated metadata path and split name
keep the experiment and feature-cache identities separate from
`split-time_open`; existing artifacts are not rewritten.

The CzechLynx parallel launcher also contains a commented
`czechlynx_unseen_eval` profile. It defaults to the repository-local generated
CSV shown below; override `CZECHLYNX_UNSEEN_EVAL_METADATA_FILE` when using a
different output, uncomment the profile, and inspect it with:

```bash
bash probe-parallel-czechlynx.sh --list-tasks
```

The profile fails before submission if the generated metadata file is missing
or if its split settings are not `unseen_eval_split`, `database`, and `query`.
It uses the same active `VARIANTS` table as the other CzechLynx profiles, so
uncomment the desired default or fine-tuned rows before submitting.

#### Linear Probe Settings

`linear_probe` trains a softmax classifier on top of backbone embeddings and can optionally tune backbone weights.

Config path:
- `benchmark.methods.linear_probe`

Core options:
- `train_mode`: `all` | `partial` | `classifier`
- `class_weighting`: `inverse_frequency` (default) | `none`
- `class_weight_normalize`: normalize inverse-frequency weights to mean 1 (default `true`)
- `class_weight_max`: cap after normalization (default `5.0`; no second normalization)
- `epochs`, `batch_size`, `num_workers`, `accumulation_steps`
- `optimizer`: `sgd` | `adam` | `adamw`
- `lr`, `momentum`, `weight_decay`, `eta_min_scale`
- `eval_batch_size`, `eval_num_workers`
- `resume_checkpoint`
- `save_checkpoint` (default `false`), `save_every`, `final_checkpoint_name`
- `partial_rules`: per-model parameter-name patterns for partial unfreezing

Reported metrics for `linear_probe`:
- Retrieval: `top_k`, `mAP` (same benchmark path as other methods)
- Classification: policy-selected `classification_top_1`, `classification_top_5`,
  `classification_top_10`, and `classification_balanced_top_1`; explicit
  `classification_seen_*` and `classification_open_*` diagnostics are also saved.

By default, the training cross-entropy is identity-weighted using only the
database/training split: each identity receives raw weight `1 / n_identity`,
weights are optionally normalized to mean one, and then capped at `5.0`.
Evaluation loss and all reported metrics remain unweighted. Set
`class_weighting: "none"` for a paired unweighted run. This policy applies to
`classifier`, `partial`, and `all` modes for both `linear_probe` and
`efficient_probe`. Singleton identities are mathematically upweighted, but
weighting cannot create additional visual information for them. Report weighted
and unweighted results separately. Validation loss and all reported metrics remain
unweighted for both probe heads.

Example snippet:

```yaml
benchmark:
  method: "linear_probe"
  methods:
    linear_probe:
      train_mode: "classifier"   # all | partial | classifier
      class_weighting: "inverse_frequency"  # inverse_frequency | none
      class_weight_normalize: true
      class_weight_max: 5.0
      epochs: 10
      optimizer: "sgd"
      lr: 0.001
      save_checkpoint: false
      partial_rules:
        default: ["layers.3", "norm"]
```

#### Efficient Probe Settings

`efficient_probe` applies a softmax head on top of ViT patch-token outputs:
- token source: `outputs.last_hidden_state[:, -number_of_patches:, :]`
- supports train modes: `all` | `partial` | `classifier`
- logs train/val loss and top-k metrics with tqdm progress bars
- when `visualization.enabled: true`, also saves a single attention-overlay grid from query images

Config path:
- `benchmark.methods.efficient_probe`

Core options:
- `train_mode`, `epochs`, `log_every`
- `class_weighting`: `inverse_frequency` (default) | `none`
- `class_weight_normalize`: normalize inverse-frequency weights to mean 1 (default `true`)
- `class_weight_max`: cap after normalization (default `5.0`; no second normalization)
- `batch_size`, `num_workers`, `accumulation_steps`
- `optimizer`, `lr`, `momentum`, `weight_decay`, `eta_min_scale`
- `dropout_rate`, `num_queries`, `d_out`
- `eval_batch_size`, `eval_num_workers`
- `resume_checkpoint`, `save_checkpoint`, `save_every`, `final_checkpoint_name`
- `partial_rules`

Visualization options used by efficient probe overlays:
- `visualization.attention_num_examples`
- `visualization.attention_average_queries`

Visualization option used by Vismatch keypoint match images:
- `visualization.vismatch_max_matches`

#### Vismatch Settings

`vismatch` runs a two-stage pipeline:
- Stage A (fast global retrieval) builds top-K candidates per query.
- Stage B reranks only those candidates with a configured local matcher.

The production path extracts each image once and matches cached features. The
optional pairwise Vismatch API is reserved for diagnostics because it would repeat
feature extraction for every candidate pair.

Config path:
- `benchmark.methods.vismatch`

Supported matcher profiles:
- `rdd-lightglue` (the migrated legacy RDD-LightGlue setup)
- `aliked-lightglue`
- `superpoint-lightglue`
- `loma` (Vismatch-managed LoMa-B)

Core options:
- `matcher`: selected Vismatch matcher profile
- `cache_dir`: matcher-specific per-image feature cache directory (`.npz`)
- `device`: `auto` | `cpu` | `cuda`
- `path_col`: metadata image path column
- `resize_max`, `top_k`, `matcher_threshold` (`null` selects the profile default: `0.01` for
  RDD/LightGlue and `0.10` for LoMa). For Vismatch, `resize_max` is the target long-side
  resolution; the shipped parity default is `512`.
- Vismatch preprocessing converts images to RGB float32 tensors in `[0, 1]` and resizes
  directly with bilinear `F.interpolate`. RDD-LightGlue, ALIKED-LightGlue, and
  SuperPoint-LightGlue floor both dimensions to multiples of 32. LoMa floors both
  dimensions to multiples of 14 because its DINOv2-L/14 descriptor requires patch
  divisibility. Source image dimensions are retained separately for provenance and
  visualization. Existing caches from the previous generic `/32` LoMa path are
  incompatible and will not be reused.
- Cosine, WildFusion, local LightGlue, linear probe, and efficient probe retain their
  existing square-resize protocols.
- `feature_matching_mode`: `feature_level` (production) or `pairwise` (diagnostics only)
- `batch_mode`: `batched` (production default) or `serial` (parity/debug reference)
- `match_batch_size`: candidate-pair batch size (default `16`)
- `extract_batch_size`: cached feature-extraction batch size (default `8`)
- `oom_backoff`: halve and retry the active CUDA batch on OOM (default `true`)
- Batched and serial matching show a pair-counted progress bar with throughput and ETA; OOM retries advance it only after successful completion.
- `stage_a_method`: `cosine` | `wildfusion` | `local_lightglue` | `linear_probe` | `efficient_probe`
- `candidate_k`: shared benchmark budget for Vismatch shortlists, WildFusion `B`, Local LightGlue `B`, and evaluation cutoffs

Vismatch probe scoring is shortlist-constrained, matching the WildFusion baseline:
only the `candidate_k` pairs are scored by Vismatch; all unscored matrix positions
are `-inf`. The primary metrics and visualizations use this same matrix. The run
records `score_matrix_policy=shortlist_only_neg_inf`, `num_candidate_pairs`,
`num_unscored_pairs`, and `candidate_fraction`. WildFusion derives its refinement
budget from the same `benchmark.candidate_k`, so comparison runs cannot accidentally
use different candidate/refinement budgets. These are not full-gallery matcher
metrics, so always interpret them together with candidate recall.

Checkpoint selection options:
- `checkpoint_source`: `default` (bundled Vismatch weights) or `custom`.
- `checkpoint_path`: an exact `.safetensors`, `.pth`, `.pt`, or epoch directory; no newest-epoch auto-selection is performed.
- `checkpoint_components`: `auto`, `matcher_only`, `extractor_only`, `descriptor_only`, or `full`.
- `loma_arch`: explicit LoMa variant, default `LoMa-B`.

For the current LightGlue-only RDD checkpoint, use:

```bash
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=rdd-lightglue benchmark.methods.vismatch.checkpoint_source=custom benchmark.methods.vismatch.checkpoint_path=/path/to/epoch_15 benchmark.methods.vismatch.checkpoint_components=matcher_only
```

The resolver detects components from tensor schemas and optionally validates `checkpoint_manifest.json`; it does not trust filenames such as `model.safetensors` or `model_1.safetensors`. RDD-LightGlue may combine custom RDD and LightGlue files, while a LightGlue/RDD checkpoint is never accepted for LoMa. Custom component hashes are included in feature-cache identities and run manifests.

Descriptor-only fine-tuning is selected explicitly with `checkpoint_components=descriptor_only`:

```bash
python train/probe.py benchmark.method=vismatch \
  benchmark.methods.vismatch.matcher=rdd-lightglue \
  benchmark.methods.vismatch.checkpoint_source=custom \
  benchmark.methods.vismatch.checkpoint_path=/path/to/descriptor/epoch_175 \
  benchmark.methods.vismatch.checkpoint_components=descriptor_only
```

RDD descriptor checkpoints apply only `descriptor.*` tensors; detector tensors
present for compatibility are shape-validated and ignored, while the default
detector and LightGlue matcher are retained. LoMa descriptor checkpoints apply
only `_descriptor.*` tensors and retain its default detector and matching layers.
`auto` recognizes descriptor training from tensor schemas and optional
`czechlynx_protocol.json`. Descriptor, matcher-only, full, and default runs have
different cache identities. Their manifests record the effective component mode,
protocol metadata, applied/ignored prefixes, checkpoint hash, and retained default
components. Optimizer, scheduler, RNG, and scaler files are never loaded.

Descriptor variants in the two parallel launchers are commented out by default.
Uncomment a descriptor row only after setting the split/profile-specific descriptor
checkpoint path. Cross-species tests must declare a separate checkpoint owner and
evaluation animal; normal profiles remain fail-closed on ownership mismatches.

To run LoMa instead of RDD-LightGlue, keep `benchmark.method: "vismatch"` and set:

```yaml
benchmark:
  methods:
    vismatch:
      matcher: "loma"
      matcher_threshold: null
```

LoMa follows the Lynx reference protocol: Vismatch's LoMa-B model, target long-side
resize with dimensions floored to multiples of 14, normalized[-1,1] cached keypoints,
mutual matching, and confidence-sum normalization by the smaller keypoint count. Its
weights are managed and downloaded by Vismatch on first use.

Vismatch is pinned to commit
`4a743b75749a3770af59d275483ed341dea51ff0` in `requirements.txt`. Its matcher
weights are downloaded on first use. The wrapper is BSD-3-Clause, but wrapped
models may have separate licenses.
The old public `rdd` method and direct RDD repository settings are unsupported.
Use `vismatch` with `matcher: rdd-lightglue` when reproducing the migrated RDD
experiment. Feature caches are matcher/profile-specific and are regenerated when
the schema, matcher, preprocessing, keypoint budget, threshold, or checkpoint
identity changes.

Batched mode preserves the Stage-A shortlist and score protocol. Feature extraction
uses matcher-native spatial-shape buckets; pair matching groups candidate pairs across
queries by exact keypoint shape, so LoMa is not padded in a way that changes softmax
normalization. Incompatible
or singleton groups use the serial backend, and CUDA OOM retries halve the active batch
size. Timing metadata records configured/effective extraction and matching sizes.
Use `batch_mode: serial` for a direct parity reference before changing matcher profiles,
preprocessing, thresholds, or keypoint budgets.

## Training and Evaluation Outputs

### Experiment artifacts and reporting

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
python scripts/summarize_runs.py --dataset CzechLynx_v2
python scripts/summarize_runs.py --method vismatch --matcher loma
python scripts/summarize_runs.py --sort-by top_1 --format markdown

# Generate per-animal CVPR-ready LaTeX and audit CSV tables
python scripts/export_paper_tables.py
python scripts/export_paper_tables.py --animal BelugaID
python scripts/export_paper_tables.py --animal CzechLynx --split-protocol split-time_open
python scripts/export_paper_tables.py --detailed-comments  # opt in to provenance comments

# Generate CVPR-style accuracy-versus-candidate-budget figures
python scripts/plot_paper_figures.py
python scripts/plot_paper_figures.py --metric top_1
python scripts/plot_paper_figures.py --animal BelugaID --metric top_5 --formats png pdf
python scripts/plot_paper_figures.py --animal CzechLynx --split-protocol split-time_closed
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

`scripts/plot_paper_figures.py` reads completed probe artifacts directly from
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

## Checkpoints and Gradient Accumulation

The standard finetune-to-probe workflow uses checkpoint-final.pth. If automatic
discovery is enabled, the newest run is searched for the configured canonical
filename first and then for compatible tagged model-only checkpoints. Full
checkpoints are never selected for inference.

accumulation_steps controls optimizer updates in all three training loops:
finetune, linear_probe, and efficient_probe. The final partial group at the end
of an epoch is flushed so its gradients are not discarded.

## Weights & Biases (W&B)

Both pipelines support optional W&B logging via config.

Enable:

```yaml
wandb:
  enabled: true
  project: "explainable-reid"
  entity: null
  group: null
  tags: []
  name: null
```

Logged data:
- finetune: train loss, validation metrics, learning rate
- probe: benchmark metrics/timings, metadata, optional visualization images
- linear_probe (within probe): per-epoch train loss, learning rate, classification + retrieval metrics
- efficient_probe (within probe): per-epoch train loss, learning rate, classification + retrieval metrics

When `wandb.name` is `null`, the repository generates an informative name such
as `probe-wildlifereid-10k-nyaladata-split-megadescriptor-l-pretrained-vismatch-loma-finetuned-k100-masked-<id>`.
It includes workflow, dataset/animal, split, model and pretrained/finetuned
backbone mode, method or matcher,
checkpoint variant or linear-probe weighting, candidate budget where relevant,
image variant, and a short run identifier. Finetuning names similarly include
the dataset, animal, split, model, epoch count, and learning rate. Set an
explicit `wandb.name` when a custom name is preferred; it always takes priority.

## Known Constraints and Future Work

- Model weights are downloaded from Hugging Face on first use.
- Vismatch requires the pinned package, model-weight downloads, and usually CUDA for practical runtimes.
- Default configs contain environment-specific shared filesystem paths; update them
  for another machine.
- Masking and Vismatch matcher settings are dataset-dependent and should be validated rather than
  assumed to improve every dataset. Matcher ablations must keep Stage-A candidates, preprocessing,
  keypoint budgets, scoring, and evaluation metrics fixed.
- The test suite intentionally avoids CUDA, downloaded models, private datasets, and external
  Vismatch integration; optional environment-gated smoke and Lynx parity checks are required
  before changing the pinned Vismatch commit.
- Future experiment priorities are tracked in AGENTS.md.

## Reproducibility

- seed control available in both configs
- deterministic mode toggle (`deterministic: true/false`)
- finetune resume from full checkpoints via `train.resume_checkpoint`
- probe feature caching keyed by method/model/checkpoint/dataset signature and `no_background`
- Cache identities also include the resolved dataset root, metadata file, and explicit `dataset.image_variant` (`background` or `no_background`) so normal and pre-masked features cannot be reused interchangeably.

## Testing

Run the tests (pytest; the tests are unittest-style classes):

```bash
uv run pytest                       # or: python -m pytest, inside the conda env
uv run pytest -m "not gpu and not data"
```

## Troubleshooting

- `ModuleNotFoundError: wildmatch`
  - Install the package (`uv sync ...` or the conda route); the code no longer patches `sys.path`.
- mask decoding errors with `no_background: true`
  - Verify metadata has valid `mask` field (JSON string or COCO-RLE dict).
- CUDA mismatch or availability issues
  - Adjust device/AMP settings in config.

## Scripts beyond training and probing

Paper tables and figures (all read completed runs under `experiments/`):

- `scripts/export_paper_tables.py`: per-animal LaTeX and audit CSV tables.
- `scripts/plot_paper_figures.py`: accuracy against candidate budget k.
- `scripts/plot_training_cost.py`: test accuracy against training GPU-hours (CzechLynx closed).
- `scripts/eval_loma_epoch_curve.sh`: Slurm array that evaluates intermediate LoMa checkpoints for that plot.
- `scripts/plot_match_examples.py`: qualitative figure, one fine-tuned LoMa match per dataset.
- `scripts/plot_data_quality_examples.py`: confirmed low-quality examples as raw photos.
- `scripts/export_class_balance.py`: per-identity image counts and imbalance statistics.
- `scripts/audit_image_quality.py`: heuristic low-quality image flags for review by eye.

Data preparation:

- `scripts/build_unseen_eval_metadata.py`: unseen-identity gallery/query split.
- `scripts/segment_with_sam3.py`: SAM 3 background removal and pre-masked metadata (`lynx-app` env, A100/H100).
- `scripts/prepare_jaguar_metadata.py`: brings the Kaggle Jaguar training data into the shared format (`JaguarReID`: `prepare` writes masked images and RLE masks, `embed` DINOv2 embeddings, `split` the burst-aware `split_v2` database/query split).

Run and log inspection:

- `scripts/summarize_runs.py`: filter and sort `reports/runs.csv`.
- `scripts/summarize_logs.py`: index the organized Slurm task logs.
- `scripts/probe_parallel_manifest.py`, `scripts/probe_log_metadata.py`: helpers called by the parallel launchers.

Project page exporters (write committed files under `docs/`; GPU where noted):

- `scripts/export_project_page_data.py`: results, curves and page figures from the paper's results snapshot.
- `scripts/export_score_separation.py`, `scripts/export_frequency_bins.py`, `scripts/export_budget_tradeoff.py`: the "why it works" views.
- `scripts/export_before_after_demo.py` (GPU), `scripts/export_rank_change_demo.py`, `scripts/export_mined_pairs_demo.py` (GPU), `scripts/export_masking_demo.py`, `scripts/export_synthetic_demo.py` (GPU): the demos.
- `scripts/export_data_challenges.py`: the data-challenges galleries.
- `scripts/build_demo_cards.py`: Demo hub thumbnails.
- `scripts/analyze_background_matches.py` (GPU): analysis only, writes to `reports/`.

The project page builds with `mkdocs build --strict` (dependencies in
`requirements-docs.txt`). The explainer video is built from `video/explainer/`; see
`video/explainer/ENVIRONMENT.md`. AGENTS.md ("Project page") holds the page's rules and
publication checklist.

## Research validity and parallel submissions

AGENTS.md is the single source for these rules; read its "Research-validity reporting
policy" and "Immutable parallel probe submissions" sections. In short:

- Ties are broken by original database index everywhere (metrics, shortlists, visualizations).
- Shortlist methods (Vismatch, WildFusion) report `mAP_at_k`; full-matrix `mAP` is emitted only
  when every gallery position is scored, and is `nan` otherwise.
- `recall_at_k`, `rerank_mAP_at_k` and `mAP_at_k` separate shortlist reach from matcher ordering.
- Every probe run writes `scores.npz`, so metrics can be recomputed without rerunning a matcher.
- Feature caches are keyed on image content, metadata, preprocessing, image variant and model
  or checkpoint weights; mask contents are not hashed yet.
- The parallel launchers snapshot each submission (config copy, task table, manifest) under
  `logs/parallel_run/submissions/`; tasks read only that snapshot, and custom checkpoints are
  checked for owner and SHA-256 before loading.
