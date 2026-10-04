# AGENTS.md

## Project purpose

This repository trains and evaluates wildlife individual re-identification systems.
Given query and database images, it produces identity-retrieval similarity scores.
The primary datasets are wildlife datasets such as CzechLynx, WildlifeReID-10k
animals (including LeopardID2022, GiraffeZebraID, CowDataset, StripeSpotter,
and SeaStarReID2023), ReunionTurtles, and Kaggle Jaguar data.

The project is retrieval-oriented. Do not describe or evaluate it as a conventional
single-label animal-species classifier: identity labels represent individual animals.

## Repository map

- models/model.py: pretrained backbone factory and ViT CLS adapter.
- models/objective.py: ArcFace, triplet, softmax, and efficient-probe objectives.
- reid/engine/finetune_runner.py: ArcFace finetuning and validation retrieval.
- reid/engine/probe_runner.py: cosine, WildFusion, local LightGlue, linear probe,
  efficient probe, and Vismatch matcher benchmark dispatch.
- reid/data/: dataset views, COCO-RLE masking, and split safety checks.
- reid/evaluation/metrics.py: top-k, balanced top-1, and mAP calculations.
- reid/training/: checkpoint serialization and accumulation helpers.
- reid/reporting/: run identities, manifests, metrics, visualization indexes, and summaries.
- conf/: Hydra configuration for probe and finetuning.
- train/ and scripts/: command-line entrypoints.
- tests/: dependency-light regression tests.
- experiments/, reports/, logs/, benchmark_runs/, and wandb/: generated
  artifacts (ignored); do not edit them manually. results/, cache/, and visualizations/ are
  legacy locations that only old workflows create; none exists at the root today.
- mkdocs.yml, docs/, overrides/: MkDocs Material project page (see "Project page");
  site/ is its ignored build output.
- video/explainer/: explainer video sources (separate `wm-video` environment).
- notebooks/: dataset annotation viewer; commit it with outputs stripped.
- notes/history.md: narrative history moved out of this guide (2026-10-03); new history goes
  to CHANGELOG.MD.

## Standard commands

Run from the repository root:

~~~bash
python train/finetune.py
python train/finetune.py train.epochs=10
python train/probe.py
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
bash probe-parallel-czechlynx.sh --list-tasks
bash probe-parallel-wildlife.sh --list-tasks
python scripts/summarize_runs.py --format markdown
python scripts/export_paper_tables.py
python scripts/export_class_balance.py
python scripts/audit_image_quality.py --dataset leopard --limit 400  # dev subset
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile models/*.py reid/**/*.py train/*.py scripts/*.py
mkdocs build --strict   # project page; needs requirements-docs.txt installed
~~~

Do not run full GPU training or Vismatch benchmarks as a default validation step.

## Hydra configuration

`train/probe.py` and `train/finetune.py` use Hydra 1.3 as their primary single-run
configuration interface. Defaults live in `conf/probe.yaml` and `conf/finetune.yaml`;
use nested dotlist overrides such as `benchmark.method=vismatch` or
`train.epochs=10`. Hydra/OmegaConf performs type conversion and rejects unknown or
misspelled configuration paths. The legacy argparse flags and `--config` option are
not supported by these entrypoints.

Hydra does not change the working directory (`hydra.job.chdir: false`). Its job log goes
to `logs/hydra/<job name>.log` (`hydra.run.dir` in both configs, pinned by
`test_hydra_job_log_stays_out_of_repository_root`); before 2026-10-03 it was appended to
`probe.log`/`finetune.log` at the repository root. New probe and finetune runs use the
reporting-managed `experiments/` layout while legacy aggregate CSVs remain under
`benchmark_runs/` and `results/`. Hydra multirun sweeps are intentionally outside
the current experiment contract. The Jaguar data runs through the shared pipeline as
the `JaguarReID` profile (see "JaguarReID"); the former standalone Kaggle submission
workflow was deleted on 2026-10-04.

`probe-parallel-czechlynx.sh` and `probe-parallel-wildlife.sh` are separate
self-submitting Slurm launchers and must not modify or replace `probe.sh`. Each
builds tasks from its explicit `VARIANTS` table and crosses them with the
`CANDIDATE_K_VALUES` list. The active tables near the top of the selected launcher
are the source of truth for its comparison grid. The CzechLynx launcher supports
the independent `split-time_closed` and `split-time_open` profiles; either or
both may be uncommented. The wildlife launcher takes one active profile at a time:
a WildlifeReID-10k animal, SalamanderID2025 or JaguarReID (see the sections of those
names below).
`MAX_CONCURRENT_JOBS` becomes the Slurm array `%`
throttle.
The CzechLynx launcher is the only parallel launcher for CzechLynx; the wildlife
launcher serves WildlifeReID-10k profiles plus the SalamanderID2025 profile
(decided 2026-09-29, so Salamander reuses the same manifest, SHA-256 and log
machinery instead of a copied launcher) and must not gain a CzechLynx profile.
The CzechLynx launcher also contains a commented `czechlynx_unseen_eval` profile.
After `scripts/build_unseen_eval_metadata.py` creates an evaluation CSV, the
launcher defaults to the repository-local CzechLynx unseen-evaluation path;
override `CZECHLYNX_UNSEEN_EVAL_METADATA_FILE` when using another output and
uncomment that profile. It uses `unseen_eval_split` with `database/query` values, validates the
generated CSV before task construction, and keeps its logs, manifests, runs, and
caches separate from `split-time_open`. It is opt-in so ordinary closed/open
submissions are unchanged.
Launcher regression tests derive their expected candidate budgets from the active launcher table, so intentional budget edits do not require changing the launcher itself. The same applies to opt-in rows and profiles: tests validate whichever classifier-probe rows and CzechLynx profiles are active (skipping when none are) and never require a specific row to be uncommented. The closed/open profile-multiplication test disables `czechlynx_unseen_eval` and any active `joint-fine-tuned` rows in its temporary copy (joint checkpoints exist for the closed split only, so the launcher would otherwise fail closed on the open profile) and re-enables the split-agnostic cosine row so the table is never empty; the joint-only guard has its own test.
`--list-tasks` and `PROBE_PARALLEL_DRY_RUN=1` are safe non-executing inspection
modes. Custom Vismatch tasks explicitly declare their component mode;
matcher fine-tuning uses `checkpoint_components=matcher_only` and descriptor
fine-tuning uses `checkpoint_components=descriptor_only`;
the selected launcher validates custom paths before submission and prints the
complete Hydra command in each task log. CzechLynx custom checkpoint paths are
stored per split; an open-split custom task fails closed if its path is missing or
does not belong to the declared animal. The supplied open profile defaults to
the `czechlynx-time-open` checkpoint root at epoch 299, using the
`loma-b-finetuned-loma-mined-legacy` and `rdd-finetuned-loma-mined-legacy` subdirectories
(both matchers are fine-tuned on LoMa-mined pairs; the closed profile uses the same names under
`czechlynx-time-closed`). Split names appear in task manifests,
metadata, log directories, and task filenames.

**Fine-tuned matcher objective (decided 2026-09-29).** RDD-LightGlue and LoMa are now
fine-tuned in the sibling `lynx-finetuning` repository with one shared recipe: the relaxed
pair score (mean over real keypoints of each keypoint's best assignment probability before
mutual selection/thresholding, averaged over both directions), a margin-0.5 hinge over every
triplet, AdamW (lr 1e-5, wd 1e-4, cosine over 300 epochs), effective batch 32, LoMa-mined
pairs for every dataset, and the fixed epoch-299 checkpoint. RDD previously trained on the
filtered inference score (dropping triplets whose positive had no surviving match) with
Adam+L2, padded keypoints inside LightGlue's assignment softmax, and an effective descriptor
batch of 4; wildlife RDD runs also mixed pair sources (RDD-mined for NyalaData and
WhaleSharkID). All RDD runs feeding the paper are retrained under the shared recipe into the
same canonical directory names, after the old directories are renamed to
`<name>__filtered-archive`, so probe launcher paths do not change except the CzechLynx closed
RDD descriptor default, which moves from `epoch_175` to `epoch_299` once the retrained
checkpoint exists. RDD probe results produced before the retrain describe the archived
checkpoints. Status checked on 2026-10-03: the retrain has not started under this plan (no
`__filtered-archive` directory exists), and the newest CzechLynx closed RDD descriptor
checkpoint is still `epoch_175`. Two relaxed-score RDD runs from 2026-09-29 exist under
non-canonical names, both on RDD-mined pairs (`czechlynx-time-closed/rdd-finetuned-rdd-mined-legacy-relaxed`,
`NyalaData/rdd-finetuned/legacy-rdd-mined-relaxed`, protocol `training_score: relaxed_v1`);
their purpose is not recorded. The probe runs reference
`<animal>/rdd-finetuned/legacy/epoch_299` for five wildlife animals that was missing on
2026-09-29. For HyenaID2022 and LeopardID2022 it was renamed, not lost:
`legacy-loma-mined/epoch_299/model.safetensors` matches the SHA-256 recorded in their runs
(checked 2026-10-03). ATRW, CowDataset and StripeSpotter have no checkpoint directory at all;
none of them is a paper dataset. Inference
ranking is unchanged: Vismatch still scores pairs with the filtered mutual-confidence score.
Descriptor profiles may explicitly separate `evaluation_animal` from
`checkpoint_owner` for cross-species tests. The immutable manifest validates that
an owner-identifiable WildlifeReID-10k path matches the declared checkpoint owner;
unidentifiable cross-species paths fail closed.
Both CzechLynx profiles use `dataset.no_background=true` and
`dataset.image_variant=no_background`, matching the shipped probe configuration;
the launcher passes these values explicitly rather than inheriting them from a
mutable configuration file.
The legacy `LOMA_CUSTOM_CHECKPOINT_PATH` and `RDD_CUSTOM_CHECKPOINT_PATH`
environment variables remain accepted as closed-profile aliases only.
The wildlife and CzechLynx launchers support three explicit classifier-probe
variants for both `linear_probe` and `efficient_probe`: `classifier`, `partial`,
and `all`; activate or comment the corresponding rows in the selected launcher’s
`VARIANTS` table as needed. They pass the method-specific
`benchmark.methods.<method>.train_mode=<mode>` directly to Hydra and run once per
active mode; classifier probes do not need the full candidate-budget sweep, so
each launcher uses the first configured candidate value only for the shared
evaluation configuration. Each mode can be paired with a `weighted` or
`unweighted` launcher row. These map to `class_weighting=inverse_frequency` or
`class_weighting=none`; the weighting label is recorded in task names, commands,
manifests, metadata, and logs. The active launcher table remains the source of
truth, so uncomment both rows when a paired comparison is wanted.
Slurm's raw array stdout and stderr remain under `logs/parallel_run/`; keep that
directory separate from single-run probe logs. After a task starts, the selected
launcher also mirrors output into
`logs/parallel_run/<dataset>/<animal>/<split_protocol>/job-<array_job>/task-<index>__<split_protocol>__<method>__<checkpoint>__k<candidate>/`.
Each task-local record contains `.out`, `.err`, `.combined.log`, and JSON metadata
with the resolved command, status, timestamps, concise failure summary, and the
experiment run directory when probe finalization prints it. `logs/index.csv` is
rebuilt atomically from these metadata records and can be filtered with
`scripts/summarize_logs.py`. Historical logs are not migrated. The launcher
resolves its repository working directory from `SLURM_SUBMIT_DIR` because Slurm
runs copied scripts from a non-writable spool directory.

## Experiment artifacts

Modern probe and finetune runs use `experiments/` and are self-contained. Paths are
organized as dataset/animal/split/model/method/variant/run-id. Each run must retain
`config.snapshot.yaml`, `run_manifest.json`, `metrics.json`, and `timings.json`;
finetune runs also retain `training_metrics.csv` and canonical checkpoints.

`reports/runs.csv` is the central one-row-per-run index. Legacy
`benchmark_runs/benchmark_results.csv` and `results/.../train_metrics.csv` remain
populated for compatibility. Historical generated artifacts are never migrated or
rewritten automatically.
Accuracy-versus-`k` figures are generated with `scripts/plot_paper_figures.py`
from the same completed run artifacts. It writes publication-quality PNG/PDF
figures under `reports/figures/`, uses a color-blind-safe palette and redundant
line/marker encodings, and renders WildFusion plus default/fine-tuned LoMa and
RDD-LightGlue series. The default `paper` style uses a logarithmic candidate
budget axis and shared metric y-limits across panels; `presentation` and
`diagnostic` styles are available for larger text or zoomed inspection.
`--exclude-method wildfusion|rdd|loma` filters a method family, including both
default and fine-tuned series; repeat the option for multiple exclusions. The
same filter applies to the corresponding descriptor-family figures.
Missing budgets are plotted as gaps; failed and incomplete runs are excluded.
For `split_protocol=unseen_eval_split`, the default plot grid is restricted to
`10, 50, 100, 160` because the generated unseen gallery has 160 images;
explicit `--budgets` values are filtered to that valid set.
The default metric set includes Top-1, Top-5, Top-10, and balanced Top-1;
`--metric` can select a subset.
Split-aware artifacts are grouped by `split_protocol`, so CzechLynx
`split-time_closed` and `split-time_open` are rendered as separate figures with
split-specific filenames. Use `--split-protocol` to select one or more splits;
never combine closed/open panels for a scientific comparison. Legacy artifacts
without split provenance retain the unsuffixed output names.
Paper tables are generated with `scripts/export_paper_tables.py` from completed
run-local manifests under `experiments/`, never from the aggregate benchmark
CSV. The exporter discovers animals, split protocols, and methods automatically,
selects the newest completed run for each split/method/matcher/backbone/train-mode/weighting/checkpoint/budget
identity, and writes ignored generated files under `reports/paper_tables/`. Main tables use
`candidate_k=50` and use the compact paired default/fine-tuned layout with
same-budget gain arrows; ablation tables use `[10, 50, 100, 250, 500, 1000]` and show
missing configurations as `--`. LaTeX displays percentage points and labels
custom checkpoints as `fine-tuned` for paper readability. Run IDs and manifest paths are included
only with the exporter’s `--detailed-comments` option; companion CSV files always
retain source fractions. For `linear_probe` and `efficient_probe`, the resolved
`train_mode` (`all`, `partial`, or `classifier`) and weighted/unweighted policy
are part of the table identity, so these variants are never collapsed into one
row. The paper-facing checkpoint column labels them as
`full fine-tuned`, `partial fine-tuned`, and `frozen`, respectively. Older
artifacts without a readable mode are labeled `unknown` rather than guessed.
The model/backbone is also part of the selection and grouping identity and is
shown in a dedicated `Backbone` column. This keeps separate cosine baselines for
MegaDescriptor-T/L, DINOv2/L, and DINOv3/L; older manifests without a model field
are labeled `unknown` rather than inferred from a path.
The visible checkpoint cell also appends the loss policy for classifier probes,
such as `frozen (weighted)` or `frozen (unweighted)`; the audit CSV retains the
machine-readable `class_weighting` field. This prevents paired weighted and
unweighted rows from appearing identical in manuscript tables.
Full-gallery methods report `mAP`, while WildFusion and Vismatch report
shortlist-aware `mAP@k`. Generated LaTeX wraps the wide tabular in
`\resizebox{\linewidth}{!}{...}` and requires `graphicx` (normally already
loaded by the CVPR template).
Generated main and ablation LaTeX use compact method sections, gray default rows,
green fine-tuned rows, and same-budget delta arrows. Main is fixed at `k=50`.
They show Top-1/5/10, balanced Top-1, and primary compute runtime, while omitting
mAP, mAP@k, and total runtime from the typeset fragments for readability. The CSV
files are the audit records and retain the omitted metrics, total runtime, and
provenance.
For `split_protocol=unseen_eval_split`, the default ablation grid is restricted
to `10, 50, 100, 160`, and the main candidate must be one of those values;
other split protocols retain the standard six-budget ablation grid.
Descriptor-only Vismatch runs are excluded from matcher tables and figures. When
present, the exporter writes separate `<animal>_<split>_descriptor_rdd_*` and
`<animal>_<split>_descriptor_loma_*` LaTeX/CSV files with the corresponding default
matcher plus cosine/WildFusion context. `plot_paper_figures.py` writes separate
`descriptor_rdd_*` and `descriptor_loma_*` figures. Descriptor and matcher
fine-tuning must be reported separately; never merge their cache identities or
same-budget gains.
Joint (descriptor + matcher trained together) runs form a third, equally separate
family (added 2026-09-30). They load with `checkpoint_components=full` (variant
`full-fine-tuned`, shown as "joint fine-tuned"), are excluded from the main and
ablation tables and from the matcher "fine-tuned" plot series, and get their own
`<animal>_<split>_joint_{rdd,loma}_*` tables and `joint_{rdd,loma}_*` figures
(`SEPARATE_FAMILIES` in `reid/reporting/paper_tables.py`, `JOINT_PLOT_SERIES` in
`plot_figures.py`). They come from the lynx-finetuning `joint` presets: RDD descriptor
+ LightGlue (`--trained_model lg+rdd --rdd_train_component descriptor`, detector frozen)
saved as an accelerate epoch directory with `model.safetensors` (RDD) and
`model_1.safetensors` (LightGlue), and LoMa DeDoDe + matcher (DaD frozen) saved as one
complete bundle; both protocol files record `*_train_component=joint`. The Vismatch
loader treats that protocol value as a complete model and refuses any other component
mode; the CzechLynx launcher's `joint-fine-tuned` rows (closed split only, paths
`CZECHLYNX_CLOSED_JOINT_{RDD,LOMA}_CHECKPOINT`) and the manifest enforce
`joint-fine-tuned` <=> `full`. Recipe: pretrained initialization, one AdamW learning
rate (1e-5), relaxed score, effective batch 32, LoMa-mined pairs, legacy protocol,
epoch 299. Verified 2026-09-30 on smoke checkpoints: detectors unchanged, descriptor
and matcher updated, strict loading into the real Vismatch models.
The manuscript template must provide `xcolor` for row colors and delta arrows.

Probe timing uses separate fields for primary compute, matcher/reranking,
method computation, feature extraction, cache lookup, model setup, calibration,
and total wall-clock runtime. Primary compute is pairwise Stage-B matching for
Vismatch, WildFusion, and Local LightGlue; Stage-A selection, feature
extraction, setup, calibration, cache I/O, and visualization are excluded.
Cosine and classifier probes use their method-computation time. Historical runs
without primary timing fields must not be relabeled as matcher timings.


Per-identity class-balance statistics for the paper are generated with
`scripts/export_class_balance.py` into the ignored `experiments/class-balance/`
directory: one CSV per dataset/split (`nyala`, `beluga`, `hyena`, `leopard`,
`sea_star`, `whale_shark`, `turtle`, `salamander`, `lynx_closed`, `lynx_open`, and the
benchmark-only `jaguar`) with exact
database/query image counts per identity, a `summary.csv` (counts, Gini, singleton
fraction, top-decile query share), and a `manifest.json` with source metadata
SHA-256 hashes. Its profiles come from `reid/reporting/paper_datasets.py`
(`ALL_PROFILES` = `PAPER_PROFILES` + `BENCHMARK_ONLY_PROFILES`), which mirrors the
launcher metadata files, identity columns, split values, roots, and mask handling; keep it
in sync when a split changes. `PAPER_PROFILES` is exactly the paper's datasets and is what
the paper figures and the page exporters look up; datasets outside the paper (JaguarReID)
go into `BENCHMARK_ONLY_PROFILES`, and only the dataset-level tools (this export and
`audit_image_quality.py`) iterate `ALL_PROFILES`. Rows whose
split is neither side (490 unlabeled ZindiTurtleRecall rows) are excluded and counted
in `excluded_rows_other_split`. The paper reports CzechLynx closed and open splits
only; the unseen-eval split is not part of the dataset statistics.

Candidate low-quality images are audited with `scripts/audit_image_quality.py`
into the ignored `experiments/image-quality/` directory. It measures the exact model
input (pre-masked WildlifeReID-10k files; CzechLynx RLE masks applied through
`BenchmarkDatasetView`) and writes per-image CSVs (foreground area, mask
fragmentation, exposure, contrast, sharpness, flags), per-flag contact sheets,
`summary.csv`, and a hash manifest. Flags are heuristic review rankings, not labels:
confirm every example cited in the paper by eye. Blur is dataset-relative (bottom 2%)
and is excluded from `any_flag`. Per-query `query_top1_rate` is joined from completed
runs' `scores.npz` using the shared descending-score, lowest-index tie rule; the join
fails closed if a run's `visualizations/index.csv` disagrees with metadata query order.
Leopard "fragmented" masks are mostly vegetation occlusion, a difficulty rather than
dataset noise; describe them separately from empty or non-animal images.
Findings verified by eye on 2026-09-23 (source data, not pipeline bugs): CzechLynx has
washed-out IR/flash frames whose masked input is nearly pure white (1,560 images with
>30% saturated foreground, mostly `snpa`), plus masks that segment branches, sticks,
vegetation, or a static corner glare repeated across a sequence. The official
SeaStarReID2023 `masked_images/` files keep only a speck for 64 images whose raw
photos are valid full-frame close-ups (a provider mask failure); WhaleSharkID shows the
same speck pattern (raw files not yet checked). Dark Hyena night images are flagged as
underexposed but remain identifiable, and flagged Hyena queries are not less accurate,
so they are not dataset noise. Report only categories confirmed this way.
For pre-masked files, foreground is `max(RGB) > 12`, so very dark animals are
undercounted: `tiny_foreground`/`empty_foreground` flags and `foreground_fraction` on
night images (e.g. HyenaID2022 row 1616) can be false positives. CzechLynx uses its RLE
mask and is unaffected. When pre-masked metadata ships its own RLE `mask` column
(SalamanderID2025, SAM3), the audit uses that mask for foreground instead of the
threshold, so black salamander skin is not counted as background; the manifest's
`foreground_source` records which source each dataset used. `summary.csv` is rebuilt
from every per-dataset CSV and manifest sources are merged by dataset, so a
`--dataset` subset run keeps the other datasets. On 2026-09-30 a subset run had
overwritten both files; the nine older rows were rebuilt from their CSVs (identical
numbers) and their manifest `runs` are recorded as unknown with the scored-run count,
because the original run lists could not be recovered.
SalamanderID2025 findings verified by eye on 2026-09-30: the threshold flags catch only
one image (row 107, dim but readable, rejected as noise). The confirmed problems come
from handheld night capture: the handler's finger hides part of the pattern and splits
the mask into fragments (rows 1006, 1238; all 8 inspected `sam3_n_instances > 1` images
were finger splits, 62 images in total), and flash close-ups are out of focus (row 249;
28 images in the dataset-relative blur ranking). Finger-split queries are not less
accurate (mean top-1 0.34 vs 0.32 over 36 runs), so report occlusion as a difficulty,
not as harmful noise; the 5 blurry queries score 0.02 vs 0.33 (small n). Rows 1280 and
281 look like the mask includes finger skin but may be shadowed body; not confirmed.

**Paper figures from confirmed examples.** History, panel lists and per-image values:
`notes/history.md`, "Paper figure: data-quality examples and qualitative match examples".
- `scripts/plot_data_quality_examples.py` writes `reports/figures/data_quality_examples.{pdf,png}`.
  It shows raw source photos only (user decision 2026-09-30): masked inputs could be read
  as our masking error. Its `EXAMPLES` list holds only images confirmed by eye, and
  mask-only problems (masks on branches, provider mask failures, finger splits) stay in
  the text.
- `scripts/plot_match_examples.py` has two steps. `candidates` writes contact sheets of
  correct top-1 pairs from the pinned fine-tuned LoMa runs (`RUNS`, matcher only, k=50);
  the user picks one pair per dataset into `EXAMPLES` by eye; `render` draws them into
  `reports/figures/match_examples.{pdf,png,json}` (`--no-tags --output-stem
  match_examples_notext` for the text-free variant; `--web-export DIR` for the page).
  Keep background dimming light (`--dim`, per-panel `Example.dim`): strong dimming exposes
  provider and SAM 3 mask outlines.
- Provenance traps found 2026-09-30. Checkpoint directories were renamed after most runs
  (`legacy/` -> `legacy-loma-mined/` or `legacy-rdd-mined/`), and Nyala's
  `legacy/epoch_299/model.safetensors` was overwritten on 2026-09-09 (the original is
  `model__actual_nyala.safetensors`). Locate checkpoints by the manifest's SHA-256 and fail
  closed when none matches. Recomputed LoMa scores differ from `scores.npz` by up to about
  1e-2 because LoMa matches under bfloat16 autocast, so rank 1 always comes from
  `scores.npz`.

**Training-cost ablation (user decisions 2026-09-30 to 2026-10-01).** CzechLynx closed
only: fine-tuned LoMa (matcher only) against the class-weighted classifier probes, with
cumulative *training* GPU-hours on RTX 4090 on the x axis (mining, feature caches and
inference excluded). `scripts/eval_loma_epoch_curve.sh` evaluates intermediate LoMa
checkpoints into `experiments/compute-efficiency/`, never `experiments/probe/`, because the
table exporter keys runs by checkpoint label and k only, so an intermediate epoch there
would replace epoch 299. `scripts/plot_training_cost.py` writes
`reports/figures/training_cost*.{pdf,png,csv,json}`. The final figures are at k=250 from
probe array 522223:

    python scripts/plot_training_cost.py --candidate-k 250 --probe-job-dir logs/parallel_run/.../job-522223 \
        --metrics balanced_top_1,top_5 --output-stem training_cost_k250_balanced_top5

Rules for the paper: the curves are test-split curves, so no epoch may ever be selected
from them; the default `--cost train` counts pure training steps on both arms, and
`--cost job` the conservative whole-job accounting; state the known biases (LoMa trained on
80 % of the train images and the probes on 100 %, 300 vs 50 epochs, one seed each). Matcher-only
fine-tuning takes 5.1 GPU-hours for LoMa (job 508111) and 5.0 for RDD (job 508028); the
descriptor and joint runs, projections to 300 epochs, values and commands are in
`notes/history.md`, "Training-cost ablation and fine-tuning GPU-hours".

New visualizations belong inside the run’s `visualizations/` directory. Their
`index.csv` must map query/database identities, ranks, scores, correctness, and
artifact paths. Top-1 and failure contact sheets are optional when no images or
labels are available. Use `scripts/summarize_runs.py` for filtered Markdown/CSV
comparisons.

Run directories use UTC timestamps plus a short resolved-configuration hash. Do not
reuse a run directory or manually edit manifests, indexes, checkpoints, or generated
images. Explicit checkpoint paths take precedence; automatic discovery searches
`experiments/finetune/` before legacy `results/` and never selects full checkpoints
for inference.

## Development Environment

Use the shared conda environment for repository work:

    /shared/results/common/kargin/tck_miniconda3/envs/ex-reid

Typical activation:

    source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
    conda activate ex-reid

Do not create or switch to another environment unless the user explicitly requests
it or the shared environment is unavailable.
The only other repository environment is `wm-video` (explainer video tooling, see
`video/explainer/ENVIRONMENT.md`); never install video dependencies into `ex-reid`.

## Configuration and data contracts

Metadata CSVs must provide the configured identity and split columns. Image paths
must resolve relative to the configured dataset root unless absolute. When
no_background is true, the configured mask column must contain valid JSON COCO-RLE
data with dimensions matching the source image.

The shipped default model is megadescriptor-l. Supported identifiers are
megadescriptor-t, megadescriptor-l, lynx_megadescriptorV3,
lynx_megadescriptorV4, miewid, dinov2, dinov2-l, dinov3, and dinov3-l. The legacy identifier
megadescriptor is intentionally unsupported.

DINO backbones are `dinov2` = `facebook/dinov2-with-registers-small` (ViT-S/14, 384-d),
`dinov2-l` = `facebook/dinov2-with-registers-large` (ViT-L/14, 1024-d), `dinov3` =
`facebook/dinov3-vits16plus-pretrain-lvd1689m` (ViT-S+/16, 384-d), and `dinov3-l` =
`facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, 1024-d). All run at 224 px with
ImageNet normalization; the Large variants deliberately keep 224 px so a small-vs-large
comparison changes only capacity. Token layout is `[CLS, 4 registers, patches]` (261
tokens for /14, 201 for /16), verified on 2026-09-23 against all four downloaded
checkpoints. DINOv3 checkpoints are gated on Hugging Face and need an accepted license.
Probe `partial_rules` must be explicit for every DINO type in both `linear_probe` and
`efficient_probe`: DINOv2 names blocks `encoder.layer.N` with final `layernorm`, DINOv3
uses `layer.N` with final `norm`, and patterns keep a trailing dot for exact substring
matches. Partial mode unfreezes the last block plus the final norm (12.6M parameters for
the Large variants) and fails closed when the patterns match no parameter. Before
2026-09-23 the DINOv3 efficient-probe rule matched nothing (a silent frozen probe) and
linear-probe DINO partial fell back to the Swin default, whose bare `norm` tuned only
LayerNorms; no DINO runs existed, so no results were affected. The Swin default rules are
unchanged even though bare `norm` also matches every Swin block norm, because existing
MegaDescriptor partial results depend on them.
Token policy: `ViTCLSAdapter` returns the post-layernorm CLS token (`last_hidden_state[:, 0]`,
equal to HF `pooler_output`); `cosine`, WildFusion's global stage, Vismatch Stage A,
and `linear_probe` therefore use CLS, not GAP. `efficient_probe` uses only the patch
tokens (`hidden[:, -number_of_patches:]`, excluding CLS and registers) with learned
attention-query pooling. Its optional `embedding_retrieval` diagnostic uses GAP (mean
of patch tokens), so it is not comparable to the CLS-based `cosine` method. CLS-only
retrieval matches DINO's k-NN protocol, not its linear-eval CLS+mean-patch concatenation.

Classifier probes support open-world evaluation across all datasets. The shipped
default is `benchmark.classifier_evaluation.open_set_policy: open`: query identities
absent from the database/training identity mapping are encoded as `-1`, remain in
predictions, receive zero classification credit, and contribute zero recall to
balanced Top-1. Their validation loss is skipped because no classifier target exists.
`warn` preserves the legacy seen-only `classification_top_*` fields while also
emitting `classification_open_*`; `closed` fails before model construction. The
policy, seen/unseen image and identity counts, and coverage are persisted in run
metrics and reporting records. `embedding_retrieval: true` enables a separate,
optional cosine diagnostic under `embedding_*`; it does not add an unknown class.

The standalone `scripts/build_unseen_eval_metadata.py` creates an evaluation-only
unseen-identity split without changing probe code, launchers, or reporting code.
It selects query identities absent from the source database identities, groups
their images by the configured encounter/group columns, assigns the earliest
ordered group to `database`, and assigns later groups to `query`. It must never
be replaced with a random image-level split. The tool preserves source columns,
adds `unseen_eval_split`, and writes `metadata_unseen_eval.csv` plus
`unseen_eval_manifest.json`. The manifest records source/output SHA-256 hashes,
selection/exclusion reasons, path-overlap checks, duplicate-content hashes,
missing files, and all split parameters. Missing/malformed columns or values,
unreadable files, path overlap, duplicate content across generated sides, and
identities that cannot produce both sides fail closed. Generated metadata is an
external input to `train/probe.py`; archive it with the experiment and use
`dataset.split_col=unseen_eval_split`, `database_split_value=database`, and
`query_split_value=query`. Its unique metadata path keeps caches and run
identities separate from the source split. Existing launchers and probe scripts
remain unchanged.

Linear-probe training uses identity-weighted cross-entropy by default. Weights
are computed from database/training labels only, in deterministic label-index
order: raw `1 / n_identity`, optional mean-one normalization, then an optional
maximum cap (default `5.0`) without renormalizing after the cap.
`class_weighting=none` restores the unweighted training loss for paired
comparisons. The policy applies equally to `classifier`, `partial`, and `all` in
both `linear_probe` and `efficient_probe`; query/test loss and all metrics remain
unweighted. Retrieval methods remain unchanged. Singleton identities are
upweighted in the formula but receive no additional visual information, so
weighted and unweighted results must be reported separately.

W&B run names are generated when `wandb.name` is null. Probe names identify the
workflow, dataset, animal, split, backbone and pretrained/finetuned backbone
mode, method/matcher, checkpoint variant,
candidate budget, image variant, and a short run token; linear probes also show
their train mode and weighting policy. Finetune names identify the workflow,
dataset, animal, split, backbone, epoch schedule, and learning rate. An explicit
`wandb.name` remains authoritative. Do not identify experiments from the random
W&B run ID alone when the generated name is available.

For `train_mode: classifier` in linear and efficient probes, the backbone is a
frozen feature extractor and must remain in `eval()` mode during training. Only
the classifier objective is put in `train()` mode. This prevents
dropout/stochastic-depth randomness and stateful normalization updates from
changing the frozen representation.

## Checkpoints

Canonical finetune outputs are:

- checkpoint-final.pth
- checkpoint-final-full.pth
- checkpoint-latest-full.pth
- checkpoint-best.pth
- checkpoint-best-full.pth
- checkpoint-epoch-<n>.pth

Tagged model-only files such as checkpoint-final_<dataset_tag>.pth remain readable
for compatibility with historical runs. Explicit checkpoint paths take precedence;
automatic probe discovery searches the newest run for canonical model-only
files and then compatible tagged model-only final files. Never pass a *-full.pth
file to inference code expecting a model-only state dict.

## Implementation rules for AI sessions

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep changes focused on the requested behavior; do not rewrite generated artifacts.
- Avoid destructive commands such as resets, broad recursive deletion, or overwriting
  unrelated results.
- Use patch-based edits when possible and fail closed when expected source text differs.
- Prefer lightweight, deterministic tests that do not download models or require CUDA.
- Run relevant unit tests and syntax checks before handoff.
- After every code, configuration, documentation, or workflow change, update both
  AGENTS.md and CHANGELOG.MD before handoff.
- After every important architectural, compatibility, experiment, or workflow
  decision, record the decision and its rationale in AGENTS.md and CHANGELOG.MD.
- Keep AGENTS.md current as the operating guide and future-work source of truth;
  keep CHANGELOG.MD as the chronological record of changes and decisions.
- Do not silently change benchmark protocols, score ranges, split semantics, or
  external matcher behavior while fixing infrastructure issues.
- Probe calibration must use the dataset returned by `load_dataset_splits`; failed-run
  reporting must preserve the original exception and create missing report parents.
- Probe finalization must import and use `file_identity` from `reid.reporting.artifacts`;
  final WildFusion matching must not be considered successful until manifest/report
  assembly completes.
- Console metric reporting must handle both numeric metrics and string diagnostic
  fields such as Vismatch cache fingerprints without changing persisted metric values.
- Record assumptions, compatibility decisions, and unresolved issues in the handoff.

## External environment

The code depends on PyTorch/torchvision, timm, Hugging Face Transformers,
wildlife-datasets, wildlife-tools, pycocotools, OpenCV, and other packages listed
in requirements.txt. Backbone weights may require network access on first use.
Hydra is pinned to `hydra-core==1.3.2`; Vismatch is pinned to commit 4a743b75749a3770af59d275483ed341dea51ff0 and downloads matcher weights on first use.
wildlife-tools and wildlife-datasets are pinned (2026-10-03) to the git commits installed in
ex-reid, `e762a6c4` and `fc702c3c` (the latter from the `develop` branch); before that
requirements.txt listed each twice, from PyPI and from an unpinned git URL. The shared ex-reid environment must have an importable, non-broken Vismatch installation; it must not depend on a missing editable checkout.
Default paths are specific to the original shared compute environment.

## Known issues (open)

Audited on 2026-08-17 and deliberately deferred; re-checked against the code on
2026-10-03, when all five were still present. Each entry records the symptom, a
reproduction, and the measured impact so it can be picked up without re-investigation.

- **Per-epoch image-level metrics dominate probe runtime.**
  `_probe_retrieval_metrics` rebuilds the same `11924 x 27836` matrix and ranks it in full
  every epoch: measured about 1 minute and 2.7 GB of transient allocation per epoch, so
  roughly 50 minutes at the shipped `epochs: 50`. Previously invisible because both probes
  crashed at epoch 1. Restrict the per-epoch call to identity-level metrics and compute the
  `image_*` diagnostics once after training.

- **Probe per-epoch validation uses the query/test split.**
  `run_linear_probe` and `run_efficient_probe` build their `[*][val]` loader from
  `dataset_query`, logging test loss and test metrics every epoch. Reported metrics are not
  affected: they come from the post-loop evaluation of the final-epoch model, and no
  best-epoch selection occurs. The hazard is downstream, since per-epoch test curves in
  W&B invite epoch or hyperparameter selection on the test split. Same class of limitation
  as the finetune selection split, which also selects the best checkpoint on the test split.
  Fix: give the probes a validation split distinct from the query/test split, or stop
  logging per-epoch test metrics.

- **Vismatch qualitative top-1 can point at an unscored pair.**
  When a query row is entirely `-inf`, `stable_rank_1d(...)[0]` returns database index 0
  and a meaningless match image is drawn instead of the query being skipped.

- **`_predict_class_probabilities` runs under grad during probe training.**
  The training-loop calls in `run_linear_probe` and `run_efficient_probe`
  (`reid/engine/probe_runner.py:1235` and `:1502` on 2026-10-03) build a graph for the
  softmax and then detach it. Wasteful, not incorrect.

- **`_to_hwc_uint8` would destroy float images.**
  `reid/data/dataset_view.py:86` clips non-uint8 input to `{0, 1}` before masking. Not
  triggered today because the base dataset yields PIL images, but it would silently blacken
  inputs if a transform were ever applied before the view.

- **Wildlife launcher defaults point at renamed checkpoint folders (found 2026-10-03).**
  Profiles without explicit layout fields default to `<animal>/{loma,rdd}-finetuned/legacy/`
  and the newer ones name `legacy` explicitly, but several directories were renamed to
  `legacy-loma-mined/` or `legacy-rdd-mined/`. Default `epoch_299` paths are missing for
  ZindiTurtleRecall (LoMa, RDD), WhaleSharkID (LoMa), BelugaID (LoMa, RDD), LeopardID2022
  (LoMa, RDD) and HyenaID2022 (LoMa, RDD); ATRW, CowDataset and StripeSpotter have no
  checkpoints. The launcher validates paths before submission, so this cannot produce wrong
  results, but fine-tuned rows for those profiles need `LOMA_CUSTOM_CHECKPOINT_PATH` or
  `RDD_CUSTOM_CHECKPOINT_PATH` until the profile fields are updated. NyalaData's `legacy`
  path exists but holds the file that overwrote the paper checkpoint on 2026-09-09 (see
  "Paper figures from confirmed examples").

## Future-work checklist

Open items only; completed items are recorded in CHANGELOG.MD. Defects with a known
reproduction live under "Known issues" instead.

- [ ] Add optional integration tests with a fake/local backbone and synthetic images.
- [ ] Add CI for unit tests, syntax checks, and YAML/config validation.
- [ ] Replace environment-specific absolute paths with machine-local overrides.
- [ ] Implement truly disjoint calibration inputs for WildFusion and local matcher
  calibration; the current split setting selects one dataset and passes it to both
  sides of calibration.
- [ ] Include mask metadata/content fingerprints in standard and Vismatch feature
  caches so mask edits invalidate features, not only image-file edits.
- [ ] Make legacy checkpoint discovery recursive for the existing nested no-manifest
  `results/<dataset>/<animal>/mask_<...>/run_<...>` layout.
- [ ] Evaluate masking and Vismatch matcher settings separately for each animal dataset.
- [ ] Profile and optimize cached Vismatch feature extraction/reranking costs.
- [ ] Run the private Lynx golden-subset parity comparison for RDD-LightGlue before changing
  matcher defaults. The 2026-08-12 full-split comparison agreed on 65 of 66 top-1
  predictions, not 100 %, so this gate is still open (details under "Vismatch matcher policy").
- [ ] Complete matcher ablations for RDD-LightGlue, ALIKED-LightGlue, SuperPoint-LightGlue, and LoMa-B.
- [ ] Track wrapped-model licenses and downloaded-weight provenance for paper release.
- [ ] Consider atomic checkpoint writes and explicit checkpoint retention.
- [ ] Reconcile historical experiment metadata and stale generated CSV schemas.
- [ ] Make central run-index updates safe for concurrent jobs and use unique temporary
  files or locking instead of a shared `reports/runs.csv.tmp` path.
- [ ] Record SHA-256 checkpoint identities and repository dirty-state/diff identity in
  manifests so uncommitted experiment code remains reproducible.


## Project page

The public project page for the manuscript is a MkDocs Material site (decided
2026-10-02): `mkdocs.yml` at the repository root, Markdown sources and assets under
`docs/`, theme overrides under `overrides/`, dependencies pinned in
`requirements-docs.txt` (installed into the shared ex-reid environment; both `mkdocs==1.6.1`
and `mkdocs-material==9.7.7` are pinned because MkDocs 2.0 drops plugins and theme
overrides, which Material's build banner warns about; the first strict build passed on
2026-10-02), ignored build output in `site/`, deployment to the `gh-pages` branch with `mkdocs gh-deploy` run by the
user. `tests/test_project_page.py` checks the configuration and sources without MkDocs:
`strict: true`, every nav entry exists and every page is in the nav, local assets exist,
relative links resolve, and no cluster path (`/shared/`, `/home/kargin`), session link,
or the phrase "ECIR 2027 paper" appears in the sources.

**Embargo (binding until the ECIR 2027 notification, 2026-12-07).** The manuscript
("WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification",
submitted to ECIR 2027, under review) is double-blind. On 2026-10-03 the user made this
GitHub repository private and fast-forwarded the former local `project-page` branch into
`main`, so the page sources now live on `main`. Until the notification, without the user's
explicit instruction: do not run `mkdocs gh-deploy` or push a `gh-pages` branch, do not
make the repository public, and keep the explainer video unlisted. **Venue rule (user decision 2026-10-02):** the site never names
the venue or the submission status anywhere, not even in source comments; the status pill
says "Manuscript · preprint to follow" and the Paper page says the manuscript is not yet
public. When a preprint appears, the page says it is available online without naming the
venue; the venue is announced only after acceptance. `test_project_page` forbids "ECIR",
"LNCS", "Springer", "under review" and "submitted to" in the site sources.
The paper brief relayed by the paper-writing session (authors, abstract, outline,
method text, dataset table, figure captions, headline numbers, open items) is stored at
`reports/project_page/paper_brief_2026-10-02.md`, under the gitignored `reports/` tree
on purpose; re-read it before writing page text. The paper's LaTeX source is cloned
read-only at `/home/kargin/Projects/repositories/ECIR-Animal-ReID-Paper` (GitHub
`turhancan97/ECIR-Animal-ReID-Paper`, approved by the user 2026-10-02; snapshot of
`2773089`, refresh with `git pull` since the paper changes until the 2026-10-05
deadline). `paper/main.tex` inputs `paper/tex/{abstract,intro,related,method_v2_mp,
setup,results,conclusion}.tex` (not `method.tex` or the `_method*.tex` drafts); tables
in `paper/tables/` and figures in `paper/figures/` come from `results/make_tables.py`
and `results/make_figures.py`, except the hand-made `teaser.pdf`, `training_mirror.pdf`,
`data_quality_examples.pdf` and `match_examples.pdf`. Never reproduce the reviewers'
note macros (`\marcinp`, `\katya`) or the page budgets in headings. The masking
sentence of the paper is unsettled: the page states masking from the mask metadata and
marks it draft rather than quoting the manuscript. The paper session is reachable over
Remote Control as "Conference abstract word count and deadline"; it receives messages
but reports nothing back.

**Framing rules from the paper (team decision 2026-09-28, relayed 2026-10-02).** The
MegaDescriptor-L candidate list is a shared implementation detail, mentioned once as the
fixed starting point of all methods; never frame the work as "two-stage retrieval" or
show a "Stage A"; say "feature matching", not "re-ranking". Page conventions follow the
paper, not the table exporter: default candidate budget k=250, Top-5 and balanced Top-1
as primary metrics, class-weighted classifier probes only, eight datasets (BelugaID
dropped), no CzechLynx open-split results (the unseen-identity protocol replaces them).
State with every cosine or classifier baseline that MegaDescriptor-L was trained on six
of the eight datasets (all except Sea star, CzechLynx, Salamander). Never call the
descriptor-only result a "collapse". No BibTeX or preprint exists; the user writes all
BibTeX entries. Items the authors still mark as draft (SAM 3 masking wording, the caching explanation
of the cost gap, pending expert study, Fig. 2 revision) carry a "Draft" admonition on the
page. Settled on 2026-10-02 by the paper session: the objective is a triplet margin loss on
the relaxed score, confirmed from the training code; the page calls it "a triplet margin
objective, a contrastive loss" and never "softmax-based contrastive" or "InfoNCE" (an
old project note that was wrong); the abstract's "contrastively fine-tune" is kept on
purpose; same-hardware was confirmed from sacct; Conclusion and limitations were pushed. Two were settled
from this server on 2026-10-02 and sent back: the SalamanderID2025 SAM 3 prompt was
"Salamander" for all 1,384 images (`masks.csv`, `prompt_used`), and every training-cost
job (508111, 508028, arrays 508523 and 522223) ran on `rtx4090_batch` per sacct.

**Branding (user request 2026-10-02).** The page uses the team's brand palette, blue
`#3a7eab`, red `#cf4832`, grey `#d1d3d4`, with the validated extension recorded in the
paper repository's `context.md`: tints 40 %/75 % toward white (`#89b2cd`/`#cedfea`,
`#e29184`/`#f3d1cc`), neutrals `#6d6e71` for small text and `#58595b` for headings,
`#d1d3d4` for decoration only. `mkdocs.yml` sets `primary: custom` and `accent: custom`
and `docs/assets/css/extra.css` maps them (links, hero title and "ours" rows blue; red
accents, status pill and warnings; grey rules). The header bar is white with dark grey
text in the light scheme and the dark neutral `#2b3036` in the dark scheme (user decision
2026-10-02, after a blue bar hid the logo's blue head and a copper `#ab673a` bar was
rejected), with a hairline `#d1d3d4` rule under header and tabs and brand-blue active tabs. Icons come from the paper's
`paper/figures/icons/` set (copied to `docs/assets/icons/`, blue and grey variants plus a
`docs/assets/icons/`). The WildMatch logo package is excluded from the paper
repository by its `.gitignore` (`/paper/figures/logo/`, so the logo stays out of the
double-blind source); the user copied it to the gitignored
`reports/project_page/logo/` on 2026-10-02. Web-sized copies live in `docs/assets/logo/`
(`wildmatch-fullname.png` for the hero, 1200 px from `B1_fullname_two-color_transparent`;
`wildmatch-wordmark.png`; `wildmatch-logo-256.png` and `-512.png`, the mark from
`wildmatch-logo.png` with its white background removed (alpha from distance to white,
edges un-premultiplied) and used as the header logo at the user's request on 2026-10-02,
replacing the white tile `wildmatch-tile-192.png`, which stays in the assets;
`wildmatch-mark-256.png`; PNG favicons and the Apple touch icon) and `docs/assets/favicon.ico`
is the package's multi-size icon. The hero puts the full-name logo (11 rem) beside the title
in a head row, with authors and buttons full width below (user request 2026-10-02), with the extra `<link rel="icon">` tags in
`overrides/main.html`. The "dark" logo variants have dark-grey lettering for light
backgrounds, not dark-mode art, so both schemes use the two-colour wordmark. Interactive charts (step 3) should use brand blue for
"ours" and the validated gold `#b8860b` as a third hue, always with a second cue (marker
or line style), because the paper notes found no safe fourth hue.

**Page components.** Each data-driven part of the page has one exporter, a committed
output, one JavaScript module under `docs/assets/js/` (mounted by `page.js` wherever its
`#wm-*` element exists) and one test. Build history, selection decisions and measured
results: `notes/history.md`, "Project page: build history".

| Component | Exporter (GPU if noted) | Output | Test |
| --- | --- | --- | --- |
| Results, curves, adapt, training cost, page SVGs | `export_project_page_data.py` | `docs/data/*.json`, `docs/assets/figures/page/` | `test_export_project_page_data`, `test_project_page_numbers` |
| Score separation | `export_score_separation.py` | `docs/assets/demo/score_separation/` | `test_export_score_separation` |
| Cost of k | `export_budget_tradeoff.py` | `docs/data/budget_tradeoff.json` | `test_export_budget_tradeoff` |
| Rare and common individuals | `export_frequency_bins.py` | `docs/data/frequency_bins.json` | `test_export_frequency_bins` |
| Before/after demo | `export_before_after_demo.py` (GPU) | `docs/assets/demo/before_after/` | `test_export_before_after_demo` |
| Rank changes | `export_rank_change_demo.py` | `docs/assets/demo/rank_change/` | `test_export_rank_change_demo` |
| Mined pairs | `export_mined_pairs_demo.py` (GPU) | `docs/assets/demo/mined_pairs/` | `test_export_mined_pairs_demo` |
| Background masking | `export_masking_demo.py` (after `segment_with_sam3.py`, `lynx-app`, A100/H100) | `docs/assets/demo/masking/` | `test_export_masking_demo` |
| Synthetic matching | `export_synthetic_demo.py` (GPU) | `docs/assets/demo/synthetic/` | `test_export_synthetic_demo` |
| Data challenges | `export_data_challenges.py` | `docs/assets/datasets/challenges/`, `docs/data/image_quality_summary.json` | `test_export_data_challenges` |
| Demo hub cards | `build_demo_cards.py` | `docs/assets/demo/cards/` | `test_build_demo_cards` |
| Explainer video | `video/explainer/build.py` (`wm-video`) | poster, `docs/data/explainer.json` | `test_project_page_numbers` |

Binding rules from that history:
- `export_project_page_data.py` reads the paper repository's frozen `results/` snapshot,
  never this repository's live `reports/paper_tables/`. Rerun it, and the score-separation,
  budget and frequency exporters, whenever the paper's results change.
- No cluster path may reach the page. `export_project_page_data.py` fails on any `/shared/`
  or `/home/` fragment, `test_project_page` forbids them in the page sources, and most
  exporter tests check their committed export; write cluster paths as `<placeholders>`.
- Demo pairs and examples are chosen by appearance or by seeded random sampling, never by
  score. Rank-change samples are random so failures are shown honestly.
- Show the default matcher next to the fine-tuned one only on the before/after demo (user
  decision): the internal expert study found fine-tuned matches less intuitive.
- The synthetic demo is never presented as a benchmark: synthetic coats share one texture
  model, so different-individual scores are high.
- A reference mask and IoU appear only where the dataset ships its own masks (synthetic
  subset, CzechLynx); WildlifeReID-10k and SalamanderID2025 masks were computed by us.
- Matches are always computed on masked inputs and drawn on raw photos; the background
  analysis showed the fine-tuned matcher needs masks at inference.
- Charts colour by matcher family (LoMa blue, RDD red, WildFusion gold), fine-tuned solid
  with filled markers, default dashed with hollow markers, baselines grey, and every value
  is also in a table.
- The explainer MP4 is never committed and stays unlisted on YouTube (`9i3iE8Bs6n8`) until
  the notification date; only its poster frame, narration MP3 and timings are tracked.

**Publication checklist (keep current; tick items as they close).**

- [ ] Settle the two "Draft" admonitions with the authors: the masking sentence on Method
  and the cost-gap explanation on Training cost; then remove the admonitions.
- [ ] Rename the GitHub repository; update `site_url`, `repo_url`, `repo_name` in
  `mkdocs.yml`, the links on Paper & Code, and remove the provisional-name note there.
- [ ] Remove the draft banner (`{% block announce %}` in `overrides/main.html`), the
  "Draft page" warning on Home, the `copyright` draft line, and change the status pill
  text when the preprint is online ("available online", venue only after acceptance).
- [ ] Add the preprint link and the authors' BibTeX entry on Paper & Code; fill the
  Acknowledgements section.
- [ ] Confirm web display rights for CzechLynx photographs and the WildlifeReID-10k
  sub-datasets other than Hyena and Leopard; update the licensing table.
- [ ] Rerun the exporters after the final paper results snapshot (`export_project_page_data`,
  `export_score_separation`, `export_budget_tradeoff`, `export_frequency_bins`) and the RDD
  retrain, and re-check `tests/test_project_page_numbers.py`.
- [ ] Check in a browser at phone width: hub cards, data-challenge galleries, the two-panel
  Plotly views, and dark mode of the PNG figures.
- [x] Upload the explainer unlisted and put the video ID in `data-youtube` on `docs/index.md`
  (done 2026-10-03, `9i3iE8Bs6n8`). Before publication: replace the description's "to follow"
  line with the page URL and switch the video from unlisted to public.
- [x] Page sources on `main` (done 2026-10-03: the user fast-forwarded `project-page` into
  `main` and made the repository private).
- [ ] After the notification date only: make the repository public (or deploy from it),
  run `mkdocs gh-deploy` (user), and verify the social card and favicon on the live URL.


## Vismatch matcher policy

The public local-matcher method is `vismatch`; the selected matcher is configured
under `benchmark.methods.vismatch.matcher`. Supported initial profiles are
`rdd-lightglue`, `aliked-lightglue`, `superpoint-lightglue`, and `loma` (Vismatch's
LoMa-B wrapper). The production path extracts features once and matches cached features. `feature_matching_mode: feature_level`
is required for production; pairwise Vismatch calls are reserved for explicit diagnostics.
Old `rdd` method names and direct RDD repository paths are unsupported and receive a migration-specific error.
The `FrameFeatures` contract and matcher profiles remain dependency-light so unit tests can run without Vismatch, CUDA, downloaded weights, or masking packages. RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue use the pinned Lynx-compatible preprocessing: RGB float32 tensors in `[0,1]`, direct bilinear tensor resize to the configured target long side, and floor of each dimension to a multiple of 32. LoMa uses the LoMa fine-tuning protocol with the same tensor interpolation but floors dimensions to a multiple of 14 for its DINOv2-L/14 descriptor. LoMa uses normalized[-1,1] keypoints and a default mutual-match threshold of 0.10; its feature cache records processed and original image sizes.
Standard and Vismatch feature-cache fingerprints also include the resolved dataset root, metadata file, and `dataset.image_variant`; normal and pre-masked features must never share a cache identity. For what the cache identities cover, including the missing mask hash, see "Research-validity reporting policy".
Deep-feature cache keys must be constructed only after dataset and model-weight fingerprints are resolved; changing model contents must invalidate the key.
The 2026-08-12 full-split parity run found identical keypoint counts and descriptor
shapes but non-bit-identical feature tensors; mean absolute per-pair score difference
was 4.38e-05. The only ranking disagreement was a near-tie, so this result supports
behavioral equivalence but does not establish strict numerical identity.
The shipped probe YAML may intentionally select another Stage-A method (currently wildfusion); this does not disable the independently selectable `vismatch` method.
WildFusion and Local LightGlue derive their refinement `B` from `benchmark.candidate_k`; `local_batch_size` controls pair-processing batches, and `local_top_k` controls the ALIKED local keypoint budget. `local_top_k` defaults to 512 with `force_num_keypoints=True`; it is included in WildFusion cache/experiment identity so changing it does not reuse a different local-feature configuration.
Custom Vismatch checkpoints are selected with `benchmark.methods.vismatch.checkpoint_source`, `checkpoint_path`, and `checkpoint_components`. `default` preserves Vismatch-managed weights; `custom` accepts an exact model file or epoch directory. Component discovery uses tensor schemas and optional `checkpoint_manifest.json`, never filename ordering. RDD-LightGlue can load custom `rdd_extractor` and/or `lightglue` components, falling back to the default component in `auto` mode when one is absent. `descriptor_only` applies only `descriptor.*` RDD tensors and retains the default detector/LightGlue; detector tensors in the file are shape-validated and recorded as ignored. LoMa `descriptor_only` applies only `_descriptor.*` tensors and retains the default detector/matcher. Protocol metadata in `czechlynx_protocol.json` is validated and recorded when available. LoMa requires a validated LoMa-compatible checkpoint and explicit `loma_arch`; generic RDD/LightGlue files are rejected. Optimizer, scheduler, RNG, and scaler files are never loaded for probing. Component SHA-256 identities, applied/ignored prefixes, protocol metadata, and effective component mode are part of Vismatch feature-cache keys and run manifests.
The Vismatch `resize_max` field is the target long-side resolution, not a downscaling-only cap; the shipped default is 512. RDD-family Vismatch profiles use preprocessing identity `lynx_finetuning_v1` and `/32` dimensions. LoMa uses `lynx_loma_finetuning_v1` and `/14` dimensions. Changing the preprocessing identity or target resolution invalidates Vismatch feature caches. Cosine, WildFusion, local LightGlue, linear probe, and efficient probe retain their existing square-resize protocols.
LoMa match visualizations must use the processed-image coordinate space shown on the canvas: convert normalized keypoints to `FrameFeatures.image_size` coordinates and apply the Vismatch/LoMa half-pixel convention, without scaling points back to `original_image_size` unless the visualization also displays raw images.
The production batching defaults are `batch_mode: batched`, `match_batch_size: 16`,
and `extract_batch_size: 8`; `batch_mode: serial` remains the diagnostic/reference
workflow for parity checks. Matching displays a pair-counted tqdm progress bar with
percentage, throughput, and ETA; progress advances only after successful batches,
including after OOM retries. Extraction buckets images by matcher-native spatial shape.
Feature matching buckets candidate pairs across queries by exact left/right keypoint
cardinality and falls back to serial for empty, singleton, or otherwise incompatible
groups. No matcher input is padded to a common keypoint count; LoMa is therefore not
subjected to padding that would change assignment-softmax normalization. When
CUDA runs out of memory and `oom_backoff: true`, the current batch is retried at half
size, temporary CUDA memory is cleared, and effective batch sizes are recorded in
Vismatch timing metadata. Stage-A candidates and `candidate_k` remain unchanged. Vismatch
and WildFusion are shortlist-constrained: Vismatch initializes unscored positions to
`-inf`, matching WildFusion's sparse matrix behavior. The policy is recorded as
`score_matrix_policy=shortlist_only_neg_inf` with candidate/unscored pair counts and
candidate fraction. The 1e-4 score/top-1 parity gate remains required before interpreting
performance results.
The explicit `dataset.image_variant` field must be `background` or `no_background`.
Use `background` for normal `images/` inputs and `no_background` for pre-masked
`masked_images/` inputs or dynamically masked images. This field is provenance,
separate from `dataset.no_background`, which controls whether an RLE mask is
applied at load time.

## Research-validity reporting policy

- Primary retrieval metrics use deterministic descending scores with original database
  index as the tie-breaker. This rule is shared by evaluation, shortlisting, and
  classifier probes. Visualization ranking is now migrated: the run-local
  `visualizations/index.csv` uses `stable_rank_1d`, so it resolves ties identically to the
  prediction grid it annotates and to the metrics. No ranking path may use
  `argsort()[::-1]`, which reverses a stable ascending sort and orders ties backwards.
- Primary `mAP` includes every query; a query with no relevant gallery identity contributes
  AP=0. `mAP_eligible` is the legacy eligible-query-only diagnostic, and the coverage fields
  `mAP_query_coverage`, `num_queries_with_gallery_match`, and
  `num_queries_without_gallery_match` report how many queries had a gallery match.
- `mAP` and `mAP_eligible` are emitted only when `score_coverage == 1.0`, that is when every
  matrix position carries a real score. A shortlist matrix leaves ~99.6% of each row at
  `-inf` ordered by original database index, so a full-matrix mAP there measures metadata
  row adjacency rather than the method: Vismatch runs spanning `top_1` 0.360-0.412 all
  produced `mAP` in 0.0463-0.0469, against 0.0168 for a random ranking. Both fields become
  `nan` instead, and no un-gated variant is persisted, so the number cannot re-enter a
  comparison by accident.
- `mAP_at_k` is the primary metric for shortlist methods and is computed identically for
  full-matrix methods, so cosine, WildFusion, and Vismatch stay comparable. It truncates at
  `benchmark.candidate_k`, grants no credit to positions the method never scored, and divides
  by `min(relevant, k)` so a shortlist miss scores 0. `rerank_mAP_at_k` divides instead by
  the hits present in the scored top-k and isolates Stage-B ordering from Stage-A reach;
  read it together with `recall_at_k` and `candidate_recall_at_k`. Matcher ablations should
  compare `rerank_mAP_at_k`, which does not charge every matcher for the same Stage-A misses.
- In reports and paper tables, `k` is the candidate budget: the number of gallery images
  retained by Stage A for second-stage refinement. Increasing `k` can improve shortlist
  coverage but costs more computation. A dash (`--`) means the method evaluates the full
  gallery and does not have a shortlist budget.
- Evaluation cutoffs are validated before model loading: every `benchmark.top_k` entry and
  `benchmark.candidate_k` must fit inside the Vismatch shortlist. Vismatch candidate
  selection, WildFusion refinement (`B`), and the mAP@k cutoff all derive from this one
  setting. The old independent budget overrides, including Local LightGlue `B`, are rejected; use `benchmark.candidate_k`.
- Probe runs persist finite score-matrix entries to run-local `scores.npz` in sparse COO
  form. Metric definitions can then be revised without repeating a matcher run. Dense
  matrices above the entry budget are skipped rather than written.
- Linear and efficient probes report identity-level metrics as primary. Their existing
  image-level matrix and metrics remain under `image_*` diagnostic fields.
- Identity-level probe scores come from the classifier's per-identity output columns, not
  from a per-database-image maximum. The head emits one probability per identity, so all
  images of an identity share it and any maximum over them is a no-op; masking the class
  axis with an image-length mask raised `IndexError` and blocked both probes entirely.
  Database label indices outside the classifier head are rejected rather than silently
  reindexed. Probe classification results include explicit open-world coverage; do not
  interpret seen-only diagnostics as all-query performance.
- Vismatch and WildFusion use shortlist-constrained ranking. Vismatch overwrites only
  shortlisted candidates in a matrix initialized to `-inf`; invalid candidate scores
  also become `-inf`. It reports `candidate_hit_rate`, `candidate_recall_at_k`,
  `num_candidate_pairs`, `num_unscored_pairs`, and `candidate_fraction`.
  The previous finite Stage-A fallback mixed incompatible cosine and matcher score
  scales and is not a supported production policy.
- File digests are memoized only inside an explicit `file_digest_cache()` block, which
  `run_probe` and `run_finetune` wrap around a whole run. The block asserts that the files
  being hashed are stable for its duration. Never make this memoization process-wide: tmpfs
  reuses one `st_mtime_ns` for rapid same-size rewrites, so a global cache could serve a
  stale digest and silently break content-addressed cache identities. Entries are keyed on
  device/inode/size/mtime so the safety-check, dataset-digest, and Vismatch cache-key paths
  share them despite constructing paths differently.
- Split safety checks are intentionally disabled in the shipped `conf/probe.yaml`
  (`safety_checks.enabled: false`), a user decision confirmed on 2026-09-23. Parallel
  launchers inherit it through the copied config, so their runs skip path-overlap and
  SHA-256 duplicate-content checks; `test_hydra_config` pins the `false` default. Do not
  re-enable it silently. When checks are off, verify split leakage separately (for
  example with a single `safety_checks.enabled=true` run) before publishing a new dataset
  or split; the statements below describe behavior when checks are enabled.
- Unseen query identities are always reported by split safety checks. Classifier probes
  resolve `benchmark.classifier_evaluation.open_set_policy`: `closed` fails before model
  construction, while `open` and `warn` continue and serialize coverage. Retrieval
  methods continue with their existing open-set ranking semantics. There is no
  warn-while-required mode; the old `warn_only_unseen` flag was unreachable and has
  been removed.
- Vismatch preprocessing runs exactly once per image. `prepare_image()` returns a
  `PreparedImage` (tensor plus source and processed sizes) that supplies the batch-bucketing
  shape and is consumed directly by `extract_prepared`/`extract_prepared_batch`. Do not
  reintroduce a shape probe that re-runs the resize, and keep buckets holding prepared
  tensors rather than full-resolution sources. Any change here must keep extracted features
  bit-identical, since feature-cache identities do not cover this code path.
- Vismatch merges its Stage-A metrics under a `stage_a_<method>_` prefix after
  `run_vismatch_benchmark` returns, since that call replaces the metrics dict. Do not coerce
  merged metric values to float: some are string diagnostics.
- Split safety preserves path-overlap checks and additionally hashes resolved files with
  SHA-256. Duplicate content across protected splits fails closed and is summarized with
  sample paths and unreadable-file counts.
- Standard and Vismatch feature caches include image-content SHA-256, preprocessing,
  metadata, image variant, model/checkpoint identity, and matcher profile/weight identity,
  but do not yet include mask metadata/content hashes. Image contents at a fixed path
  invalidate caches; mask edits require the future cache-fingerprint fix.
- Automatic inference checkpoint discovery searches recursively under modern finetune
  experiments, ignores failed/incomplete manifests and `*-full.pth`, prefers completed
  canonical model-only files, then tagged legacy files, and preserves explicit-path priority.
  Nested legacy runs without manifests are not yet discovered (see "Future-work checklist").
- Finetune resume fails closed when the checkpoint leaves no epochs to run
  (`start_epoch >= train.epochs`). A zero-epoch run would still write final checkpoints and
  a completed manifest, hiding an unraised `train.epochs`; the guard runs before training
  setup so nothing is written.
- Finetune reports select and reload the best model-only checkpoint for primary metrics;
  final-epoch metrics remain nested as `final_epoch_metrics`. The selection split is the
  test split, a documented limitation (see "Known issues", probe per-epoch validation).
- Accumulation divides each raw loss by the actual microbatch count in its group, including
  a partial final group; optimizer-step boundaries and scheduler behavior are unchanged.
- WildFusion calibration excludes same-image diagonal pairs by default and warns on the
  database-derived fallback. The configured split currently produces one calibration
  dataset used on both sides; a truly disjoint protocol is in the future-work checklist.
  `official_same_set: true` enables exact all-pairs compatibility calibration for parity.

These validity changes are forward-only. Historical generated artifacts, aggregate CSVs,
and old caches are not rewritten automatically; rerun affected experiments before using
their numbers. In particular, historical `mAP` values in `reports/runs.csv` predate the
coverage gate, are not comparable across methods, and cannot be recomputed because those
runs did not persist `scores.npz`.


## Immutable parallel probe submissions

The selected dataset launcher creates a submission directory under `logs/parallel_run/submissions/<submission_id>/` containing the copied Hydra config (`probe.yaml`), task table (`tasks.tsv`), and JSON manifest (`manifest.json`). The manifest is passed to Slurm with `--export=ALL,PROBE_PARALLEL_MANIFEST=...`; array tasks must read it rather than rereading `conf/probe.yaml`, shell checkpoint variables, or mutable dataset settings.

Dataset profiles explicitly pair dataset/animal settings with expected custom-checkpoint owners. Submission-time SHA-256 hashes and owner declarations are validated before model loading or cache writing. A missing, changed, or mismatched checkpoint/config fails closed and cancels only the current array element. The active benchmark grid and `probe.sh` contract remain unchanged. Use the selected launcher with `--list-tasks` or `--dry-run` for inspection, and never alter submitted manifests, copied configs, or checkpoint inputs.
Exactly one `DATASET_PROFILES` entry must be active. The wildlife launcher includes
templates for NyalaData, WhaleSharkID, BelugaID, ZindiTurtleRecall, ATRW, Giraffes,
LeopardID2022, HyenaID2022, GiraffeZebraID, CowDataset, StripeSpotter, and
SeaStarReID2023, plus a commented SalamanderID2025 profile; the current active entry
is the uncommented row in the file. The
added profiles use official pre-masked metadata and profile-specific checkpoint
layout/epoch fields; the newer WildlifeReID-10k profiles name `legacy/epoch_299/model.safetensors`
for both LoMa and RDD, which no longer exists for several animals (see "Known issues"). The launcher derives default LoMa and RDD checkpoint
paths from the active profile and fails before task generation when zero or multiple
profiles are active. Explicit checkpoint overrides remain supported but must belong
to the active animal; `animal_name`, when supplied, must match it.

### SalamanderID2025

SalamanderID2025 (added 2026-09-29) is not part of WildlifeReID-10k: 1,384 images of
584 fire salamanders at `/shared/sets/datasets/vision/czechlynx/SalamanderID2025`
(repository symlink `dataset/czechlynx/SalamanderID2025`), with a time-closed
`split` column of `database` (1,138) / `query` (246) values; every query identity is
in the database and 372 database identities are singletons. Backgrounds were removed
with `scripts/segment_with_sam3.py` (SAM3, prompt "Salamander", all detected instances
merged because an occluding finger splits one animal into several instances; one image,
`query/images/9d1fc96e28c0058e_1277.jpg`, needed a reviewed 0.10 threshold). The script
writes `masked_images/`, `masks.csv` (COCO-RLE and SAM3 quality fields) and
`split_time_closed_no_background.csv`, whose `path` points at the masked image and which
keeps the original row, adds `original_path`, `mask`, `sam3_*`, and `split_train_test`
(database->train, query->test) for the lynx-finetuning wildlife pipeline. Probes use
the masked metadata with `no_background=false`, `image_variant=no_background`, and
`split`/`database`/`query`; results land in
`experiments/probe/SalamanderID2025/SalamanderID2025/split/` and share the `split`
split-protocol figure group with WildlifeReID-10k. Unlike the other datasets, LoMa is
fine-tuned on LoMa-mined and RDD-LightGlue on RDD-mined pairs (user decision), stored as
`wildlife-reid-10k/SalamanderID2025/{loma,rdd}-finetuned/legacy-{loma,rdd}-mined/`; state
this in the paper. A 2026-09-29 run with `safety_checks.enabled=true` found no path or
content overlap between database and query. SAM3 runs in the `lynx-app` conda env on an
A100/H100 only (its CUDA 13 PyTorch has no V100 kernels).

### JaguarReID

JaguarReID (added 2026-10-03) is the labelled part of the Kaggle Jaguar Re-ID data at
`/shared/sets/datasets/vision/czechlynx/jaguar` (repository symlink
`dataset/czechlynx/jaguar`): 1,895 training photos of 31 jaguars (13-183 each, no
singletons). Kaggle's `test.csv` is an unlabelled list of 137,270 pairs over 371 test
photos and is not used. The source PNGs are RGBA with the background removed in the alpha
channel, but the RGB channels keep the original background, and the shared loader reads
images with OpenCV's default flag (alpha dropped), so the files cannot be used directly.
The files carry no capture time.

`scripts/prepare_jaguar_metadata.py` (CPU) has three steps, all writing into the dataset
root. `prepare` writes `masked_images/<filename>` (RGB times alpha, black background) and
`jaguar_reid_base.csv` (`image_id`, `identity`, masked `path`, `original_path`, COCO-RLE
`mask` from alpha, 256-bit difference hash `dhash`, `masked_sha256`) with
`jaguar_reid_base_manifest.json`. `embed` writes `jaguar_reid_dinov2_small_cls.npz`:
DINOv2-small CLS embeddings of the masked images on the CPU (a generic model, not the
MegaDescriptor backbone being evaluated, so the split is not tuned against it). `split`
writes `jaguar_reid_v2_no_background.csv` (base columns plus `dup_group`, `split_v2`
database/query and `split_train_test` train/test for the lynx-finetuning wildlife
pipeline) and `jaguar_reid_v2_manifest.json` (source hashes, parameters, counts, leakage).
The `v2` file name and the `split_v2` column name are kept because the probe records the
split column as the run's split protocol and the runs made on this split record the file's
SHA-256; renaming would orphan them.

**Burst-aware split (user decision 2026-10-04: keep it; the first, hash-only split was
deleted).** The photos come in long bursts of near-identical frames, so a per-image split
puts copies of a query in the gallery. `split` joins two photos of the same jaguar into
one group when their hash distance is at most 32, when their files are adjacent
(`train_0688`/`train_0689`; file order follows capture order within a jaguar) and their
embedding cosine is at least 0.85, or when their cosine is at least 0.95 anywhere. Groups
are joined transitively (only same-jaguar pairs; a cross-jaguar pair at hash distance 12
or less fails closed as a probable label error). Per jaguar, whole groups go to `query`
until about 25 % of its photos are there (groups that would overshoot the target by more
than 20 % are skipped, seed 0), and every jaguar stays on both sides. Result: 416 groups
(largest 90: Lua lying on one log for files 0835-0924, checked by eye), 1,408 database and
487 query photos, all 31 jaguars on both sides, query share per jaguar 11-33 % (fewest
queries: 3), no adjacent similar pair, no pair at cosine 0.95 and no hash twin across the
sides; a safety-check run found no path overlap, no duplicate content and no unseen query
jaguar.
Why these settings: a hash-only grouping missed most bursts (adjacent frames of one
sequence often differ by 80-120 bits because the mask cut-out changes), and on such a
split Top-1 was near 1.00 for queries with a near-identical database twin. Contact sheets
of adjacent same-jaguar frames showed bursts at cosine 0.85 and above, mixed bursts and
new poses at 0.80-0.85, and mostly new poses or scenes below 0.80; different jaguars
reach cosine 0.88 at the 99th percentile, so similarity alone cannot join non-adjacent
photos. The cutoff matters: MegaDescriptor-L cosine Top-1 (full gallery, mean of five split
seeds) is 0.53 at a 0.90 cutoff, 0.41 at 0.85 and 0.34 at 0.80 (seed-to-seed standard
deviation about 0.03), and no cutoff gives a plateau, so this is a documented judgement
call, not a time-based split.
On 2026-10-04 the first split's files and results were deleted at the user's request: its
metadata and manifest in the dataset folder, its six runs under `split/`, their task logs
(`job-524007`, removed from `logs/index.csv` with `scripts/summarize_logs.py --write-index`)
and the hash review sheets. `reports/runs.csv` still lists those six runs with missing
manifests, because that index must not be edited by hand and has no rebuild tool.

Probes use `no_background=false`, `image_variant=no_background`, `split_v2`
database/query, and land in `experiments/probe/JaguarReID/JaguarReID/split_v2/`. First
results (2026-10-04, k=10, shortlist ceiling 68.4 %): cosine 45.2 / 60.0 Top-1 / Top-5
(balanced Top-1 41.5), WildFusion 60.4 / 65.3 (57.8), default LoMa 61.8 / 66.9 (58.9),
default RDD-LightGlue 60.0 / 66.1 (56.4). Shortlist ceilings from cosine: 81.5 % at k=50,
87.1 % at k=100, 91.8 % at k=250, 95.3 % at k=500, 100 % at k=1000. The full k grid and the
linear probes have not been run yet. The wildlife launcher carries the `jaguar` profile; no
fine-tuned matchers exist yet, so only default rows can run until `lynx-finetuning` gets a
Jaguar config (shared recipe, both matchers on LoMa-mined pairs, checkpoints under
`wildlife-reid-10k/JaguarReID/{loma,rdd}-finetuned/legacy-loma-mined/`, the paths the
profile expects). The profile is in `BENCHMARK_ONLY_PROFILES`; JaguarReID is not a paper
dataset. The image-quality audit (2026-10-03, RLE mask foreground) flagged 80 of 1,895
photos by heuristics; none has been checked by eye.
**Kaggle submission workflow deleted (user decision 2026-10-04).** The former standalone
pipeline (`reid/engine/kaggle_jaguar_runner.py`, `scripts/kaggle_jaguar_submit.py`,
`config/kaggle_jaguar.yaml`, `kaggle_jaguar_submit.sh`: its own random split, ArcFace
fine-tuning, cosine/WildFusion retrieval with Vismatch score fusion, identity-balanced
all-vs-all mAP and Kaggle CSV export) was removed together with its outputs
(`kaggle_runs/`, 14 runs from 2026-03-13 to 03-15, and its 10 GB alpha-masked image cache
under the dataset's `cache/`). Nothing in JaguarReID used it. Its results (for example
0.847 mAP) are no longer reproducible and were never comparable with JaguarReID results;
recover the code from git history before 2026-10-04 if Kaggle submissions are needed.

LoMa descriptor exports may omit standard BatchNorm running-stat buffers. Descriptor
loading preserves those non-learned buffers from the active model defaults while
remaining strict for descriptor parameters, unexpected tensors, and tensor shapes.
