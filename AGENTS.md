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
- reid/engine/kaggle_jaguar_runner.py: standalone Jaguar workflow.
- reid/data/: dataset views, COCO-RLE masking, and split safety checks.
- reid/evaluation/metrics.py: top-k, balanced top-1, and mAP calculations.
- reid/training/: checkpoint serialization and accumulation helpers.
- reid/reporting/: run identities, manifests, metrics, visualization indexes, and summaries.
- conf/: Hydra configuration for probe and finetuning.
- config/: standalone Jaguar YAML configuration and other non-Hydra configs.
- train/ and scripts/: command-line entrypoints.
- tests/: dependency-light regression tests.
- experiments/, reports/, results/, benchmark_runs/, kaggle_runs/, cache/, and visualizations/:
  generated artifacts; do not edit them manually.
- mkdocs.yml, docs/, overrides/: MkDocs Material project page (see "Project page");
  site/ is its ignored build output.

## Standard commands

Run from the repository root:

~~~bash
python train/finetune.py
python train/finetune.py train.epochs=10
python train/probe.py
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
bash probe-parallel-czechlynx.sh --list-tasks
bash probe-parallel-wildlife.sh --list-tasks
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id
python scripts/summarize_runs.py --format markdown
python scripts/export_paper_tables.py
python scripts/export_class_balance.py
python scripts/audit_image_quality.py --dataset leopard --limit 400  # dev subset
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile models/*.py reid/**/*.py train/*.py scripts/*.py
mkdocs build --strict   # project page; needs requirements-docs.txt installed
~~~

Use --dry-run, --pair-limit, or --fast for Jaguar development runs.
Do not run full GPU training or Vismatch benchmarks as a default validation step.

## Hydra configuration

`train/probe.py` and `train/finetune.py` use Hydra 1.3 as their primary single-run
configuration interface. Defaults live in `conf/probe.yaml` and `conf/finetune.yaml`;
use nested dotlist overrides such as `benchmark.method=vismatch` or
`train.epochs=10`. Hydra/OmegaConf performs type conversion and rejects unknown or
misspelled configuration paths. The legacy argparse flags and `--config` option are
not supported by these entrypoints.

Hydra does not change the working directory. New probe and finetune runs use the
reporting-managed `experiments/` layout while legacy aggregate CSVs remain under
`benchmark_runs/` and `results/`. Hydra multirun sweeps are intentionally outside
the current experiment contract. Jaguar remains on its
existing argparse and `config/kaggle_jaguar.yaml` workflow; its internal finetune
template points to `conf/finetune.yaml`.

`probe-parallel-czechlynx.sh` and `probe-parallel-wildlife.sh` are separate
self-submitting Slurm launchers and must not modify or replace `probe.sh`. Each
builds tasks from its explicit `VARIANTS` table and crosses them with the
`CANDIDATE_K_VALUES` list. The active tables near the top of the selected launcher
are the source of truth for its comparison grid. The CzechLynx launcher supports
the independent `split-time_closed` and `split-time_open` profiles; either or
both may be uncommented. The wildlife launcher takes one active profile at a time:
a WildlifeReID-10k animal or SalamanderID2025 (see "SalamanderID2025" below).
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
checkpoints; five wildlife RDD checkpoints they reference (HyenaID2022, LeopardID2022,
ATRW, CowDataset, StripeSpotter) were already missing on disk on 2026-09-29. Inference
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
`sea_star`, `whale_shark`, `turtle`, `salamander`, `lynx_closed`, `lynx_open`) with exact
database/query image counts per identity, a `summary.csv` (counts, Gini, singleton
fraction, top-decile query share), and a `manifest.json` with source metadata
SHA-256 hashes. Its profiles come from `reid/reporting/paper_datasets.py`
(`PAPER_PROFILES`), which mirrors the launcher metadata files, identity columns,
split values, roots, and mask handling; keep it in sync when a paper split changes. Rows whose
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
`scripts/plot_data_quality_examples.py` renders the confirmed examples to
`reports/figures/data_quality_examples.{pdf,png}` as one row of raw source photos in
letterboxed square panels. Since 2026-09-30 it shows raw photos only (user decision):
masked model inputs could be read as our error because we removed backgrounds
ourselves (SAM3 for SalamanderID2025), so every panel must show a problem visible in
the raw photo. Panels, in order, are (a) Overexposure (CzechLynx row 38308,
LeopardID2022 row 2824), (b) Insufficient detail (ZindiTurtleRecall row 11453, whose
turtle spans a few dozen pixels at model input size and whose file is stored rotated
90 degrees with no EXIF orientation tag, so models also see it sideways; WhaleSharkID
row 614, a hazy backlit silhouette with no visible spot pattern), (c) Empty frame
(CzechLynx row 36426, a stick; its 368 px file carries 20 px left and 118 px bottom
black padding, so `fill=True` trims that padding and centre-crops to a square so the
photo fills its panel; use `fill` only for source padding, never to hide content), (d) Corruption (CzechLynx row 2307, colour banding),
and (e) Blur (HyenaID2022 row 550). Labels use formal image-quality terms and the
figure uses Times (Times New Roman when installed, else the bundled Times-compatible
STIX, embedded in the PDF) to match the CVPR body text. Group labels wrap onto two lines when wider than their columns. Panels carry
only the group label and dataset name; per-image notes stay in `EXAMPLES` as
provenance and are not drawn. Mask-only
problems (masks on branches, Sea Star/Whale Shark provider mask failures, Salamander
finger splits) stay text-only. Its `EXAMPLES` list holds only images confirmed by eye.
`scripts/plot_match_examples.py` (added 2026-09-30) renders one successful fine-tuned
LoMa match per paper dataset, in alphabetical order (Czech Lynx closed, Hyena, Leopard,
Nyala, Salamander, Sea Star, Turtle (Zindi), Whale Shark), as three panels per row with
the last row of two centred (`--columns 3`; 6.875 x 2.97 in, half the height of the first
4 x 2 layout, to save page space) in
`reports/figures/match_examples.{pdf,png}` plus a `match_examples.json` sidecar. User
decisions: one method (fine-tuned LoMa, `matcher_only`, k=50, runs pinned in `RUNS`),
raw photos with background, keypoints mapped from normalized LoMa coordinates with
`x = W(x_n+1)/2 - 0.5`, only the 30-50 strongest lines drawn, labels with dataset name,
"rank 1" and match count. `candidates` lists correct top-1 queries from the run's
`scores.npz` (shared tie rule), rejects same-encounter/same-day pairs where metadata
records them and dHash near-duplicates, sorts by MegaDescriptor-L cosine (hardest
first), and writes `match_candidates/<dataset>.csv` and `_sheet.png`; the user picks one
pair per dataset by eye into `EXAMPLES`, then `render` draws them. The user's picks
(2026-09-30, contact-sheet rows): Nyala 5, Hyena 3, Leopard 3, Sea Star 5, Whale Shark 5,
Turtle 3, Salamander 2, and for Czech Lynx row 12 of the later daylight sheet
(`lynx_041`, 17-09-2022 vs 12-01-2020, 396 matches), which replaced the pink infrared
row-4 pair the user rejected; each `Example.note` records its cosine, match count
and probe score. `render` crops every photo to one aspect ratio (`--aspect`, default
4:3) around its matched keypoints so all cells share one shape; the crop trims only the
long side, and correspondences with an endpoint outside a crop are dropped before the
strongest 40 are drawn, while the label keeps the total match count.
Paper styling (2026-09-30): Times/STIX embedded fonts, panel titles "(a) Nyala" ...,
corner tags "Query", "Top-1 match" and "<N> matches" ("rank 1" belongs in the caption),
black source padding trimmed from edge bands (`padding_box`), 30 spatially spread
high-confidence matches, reduced to 10 at the user's request (`--paper-matches`;
`spread_selection`: greedy in `stable_rank_1d` order with a minimum endpoint spacing of
12% of the photo height, halved until enough matches exist)
drawn in one cyan with a dark outline and white-edged endpoints, and photos embedded at
320 px height (about 300 dpi at `figure*` width; PDF about 3.6 MB). At the user's request the
default is now a light dim (`--dim 0.2`: background brightness and saturation 0.8,
mask feathered six times wider), which keeps mask outlines nearly invisible; `--dim 0`
disables it. Strong dimming (brightness 0.55) was rejected: on 2026-09-30 it
exposed provider and SAM3 mask outlines (a hard pink block on the CzechLynx IR frame,
patchy water on WhaleSharkID, a flat grey Hyena background), which would read as our
masking error, the same reason the data-quality figure shows raw photos only.
Dimming is also set per panel with `Example.dim`: 0.4 for Leopard, Nyala and Salamander and
0.35 for Sea Star and Turtle, whose masks follow the animal cleanly; Whale Shark keeps
the light default 0.2 because its mask shows outlines when dimmed harder. Czech Lynx uses
0.35 since its daylight replacement pair masks cleanly (user request). Photos are embedded at 260 px height (about 300 dpi at three panels per row)
and, at the user's request, corner tags are 6.5 pt (`--tag-size`) and panel titles 9 pt
(`--title-size`; the title band grows with it, so the figure is about 3.07 in tall). `render --no-tags --output-stem match_examples_notext` (2026-10-02, user request) draws no
text on the photos at all (no Query/Top-1 match/match-count tags; dataset titles stay) into
`match_examples_notext.{pdf,png,json}`, leaving `match_examples.*` untouched; without tags the
tag-area exclusion is skipped, so a few drawn correspondences can differ from the tagged version.
Refinements (2026-09-30): "Query"/"Top-1 match" tags appear only on panel (a) (the
caption states the convention) while every panel keeps its match-count tag; matches
with an endpoint under a tag are dropped before the spread selection
(`points_outside_boxes`); the line outline is thicker (1.35 pt, alpha 0.7) for contrast
on water; Hyena uses `dim=0.0` because dimming its blown-out white background left a
grey halo; the figure has no side margin (spans `\linewidth`) and the gap between
panels (0.07 in) is about three times the gutter inside a pair. `candidates` gained
presentability filters for replacing a weak panel: `--min-foreground` (mask share of
each raw photo), `--min-saturation`, and `--min-warm` (share of orange-brown animal
pixels; measured 0.0 on CzechLynx pink infrared frames, which saturation alone does not
reject, and 0.57-0.74 on daylight coats), plus `--tag` to keep earlier sheets. The
daylight CzechLynx search used `--pool 3000 --min-foreground 0.15 --min-warm 0.5`
(793 of 3,000 pairs passed; 40 with at least 40 matches) and wrote
`match_candidates/lynx_closed_daylight*`, including a paper-style preview. Two provenance traps
found on 2026-09-30: (1) checkpoint directories were renamed after most runs
(`legacy/` -> `legacy-loma-mined/`, `legacy-rdd-mined/` for Whale Shark) and Nyala's
recorded `legacy/epoch_299/model.safetensors` was overwritten on 2026-09-09 by another
file (the original survives as `model__actual_nyala.safetensors`), so the script locates
the checkpoint by the manifest's SHA-256 and fails closed when none matches; (2) the
matcher reads the probe's own Vismatch feature cache (key rebuilt from the run's
recorded `vismatch_cache_fingerprint`), because batched and single-image LoMa extraction
give slightly different keypoints. Recomputed LoMa scores still differ from
`scores.npz` by up to about 1e-2 (V100 and H100 alike), because LoMa matches under a
bfloat16 autocast (`cfg.mp`); the 1e-4 parity gate therefore cannot be met for LoMa, and
rank 1 always comes from `scores.npz`, never from the recompute. Probe runs that load
the canonical `model.safetensors` path for Nyala after 2026-09-09 use a different file
than the paper-table run did.

**Training-cost ablation (revived 2026-09-30 at the team's request; minimum-effort design,
user decisions).** CzechLynx closed only; fine-tuned LoMa (matcher only, k=50) against the
six classifier probes (frozen/partial/full x weighted/unweighted, run 508523). Axis:
cumulative *training* GPU-hours on RTX 4090 (both arms trained on that GPU type); LoMa's
mining and feature-cache costs are excluded, and inference cost is not reported. Metric:
test Top-1 and balanced Top-1. Accepted limitation: the curves use the test split because
no clean validation split exists, so no epoch is ever selected from them; each method's
fixed final epoch stays the headline. Probe curves come from the existing per-epoch
`[linear_probe] epoch N/50 ... val_top1=` log lines (identical to the final `top_1`) and
the tqdm training time per epoch; no probe is retrained. LoMa points come from the saved
checkpoints `epoch_000..250` (epoch 299 is run 20260920T122915Z_0015f14a), evaluated by
`scripts/eval_loma_epoch_curve.sh`, a 6-task Slurm array that reuses that run's frozen
config copy and changes only the checkpoint path, a disposable feature cache
(`.../cache/vismatch_epoch_curve`, about 20 GB per epoch, since the cache key includes the
checkpoint), and the outputs, which go to `experiments/compute-efficiency/` (runs, legacy
CSV, run index). They must stay out of `experiments/probe/`: the paper-table exporter keys
runs by checkpoint label and k only, so an intermediate epoch there would replace epoch 299.
`scripts/plot_training_cost.py` collects everything and writes
`reports/figures/training_cost.{pdf,png,csv,json}`. User decision (2026-09-30): report the
weighted-loss probes only, in three panels (Top-1, Top-5, balanced Top-1). Top-1 and Top-5
are logged per probe epoch (both checked against the run's final metrics); balanced Top-1
was not, and no probe checkpoints or per-query scores were saved, so probes contribute only
their final balanced Top-1 marker. State in the paper that LoMa's Top-5 is capped by the
k=50 shortlist (a correct identity outside it can never rank in the top 5), which is why
full fine-tuning leads on Top-5 (55.4 vs 47.7) while LoMa leads on Top-1 and balanced Top-1.
`--candidate-k 100` or `250` renders the same figure at that budget (`training_cost_k100.*`,
`training_cost_k250.*`) from the existing epoch-299 and default runs; training cost is identical (k only changes how
many candidates LoMa re-ranks at test time, and probes ignore k), but matching time
roughly doubles at k=100 (2,140 s vs 1,080 s for 11,924 queries) and is five times higher
at k=250 (5,447 s), where LoMa also overtakes full fine-tuning on Top-5 (58.3 vs 55.4). Intermediate k=100 epochs would
need their own evaluation array; k=50 stays the paper's main budget.
Since 2026-09-30 the linear- and efficient-probe epoch lines end with
`val_balanced_top1=` (the already-computed `classification_balanced_top_1`, which equals
the run's final `balanced_top_1`); the value is appended last so older parsers keep
working. The user reruns the three weighted CzechLynx closed probes to get balanced curves;
`plot_training_cost.py --probe-job-dir logs/parallel_run/.../job-<new id>` draws them
whenever every epoch has the field, and takes job durations from the task metadata
start/end times (within 3 s of sacct). For comparable GPU-hours the rerun must use the
original resources (`rtx4090_batch`, QOS `batch`), which `SBATCH_PARTITION`/`SBATCH_QOS`
override without editing the launcher. The reruns land in `experiments/probe/` with the same
identity as runs 508523, so the paper-table exporter will then pick the newer ones.
On 2026-10-01 the user switched `scripts/eval_loma_epoch_curve.sh` to k=250 on
`rtx4090_batch` (all six intermediate checkpoints evaluated, plus the existing epoch-299
k=250 run) and reran the weighted probes as array 522223 (RTX 4090, `val_balanced_top1`
logged). The requested figure is two panels, balanced Top-1 left and Top-5 right, at k=250:
`plot_training_cost.py --candidate-k 250 --probe-job-dir .../job-522223 --allow-incomplete
--metrics balanced_top_1,top_5 --output-stem training_cost_k250_balanced_top5`.
`--allow-incomplete` draws a still-running probe up to its last epoch with both a metric
line and a finished training bar (no final marker, no metric check, train cost only);
rerun without it once partial and full finish. The frozen rerun reproduced run 508523
(Top-5 21.14, balanced 9.76). Finding to report carefully: at k=250, LoMa reaches its
plateau after the first fine-tuning epoch (epoch 0: balanced 35.2, Top-5 57.5 at 0.02
GPU-h, vs default 31.3 / 55.2), and later checkpoints move within about 1.5 points
(epoch 299: 34.7 / 58.3). These are test-split curves, so no earlier epoch may be
selected from them.
Final k=250 figures (2026-10-01, all LoMa evaluations and probes of array 522223 done):
`training_cost_k250_balanced_top5.*` (balanced Top-1 | Top-5) and `training_cost_k250_weighted.*`
(Top-1 | Top-5 | balanced Top-1), rendered without `--allow-incomplete`. Values: LoMa FT
5.1 GPU-h, 49.1/58.3/34.7 (Top-1/Top-5/balanced); full FT 10.0 GPU-h, 30.1/55.4/19.4; partial 8.1 GPU-h, 24.0/47.9/17.6;
frozen 3.8 GPU-h, 8.8/21.1/9.8; default LoMa 46.0/55.2/31.3; cosine 16.2/31.8/9.1. All three
rerun probes reproduce array 508523 exactly; the log-x variant was re-rendered too.
`--x-scale log` (2026-10-01, `training_cost_k250_balanced_top5_logx.*`) spreads LoMa's first
checkpoint (0.02 GPU-h) from the rest, so its jump above default LoMa and the flat plateau
over two decades of compute are visible; zero-cost methods (default LoMa, cosine) then
appear only as reference lines, since 0 has no position on a log axis.
`--cost job` (added 2026-09-30, "fair plot" request) renders the conservative accounting
(`training_cost_job.*`): whole Slurm job GPU-hours (sacct elapsed: LoMa 508111 8,026 s x 4
GPUs; probe tasks 0/2/4 of 508523), i.e. training plus per-epoch evaluation/validation,
setup and checkpointing, with the one-off overhead charged at epoch 0. Totals: LoMa 8.9,
frozen/partial/full probes 5.8/10.0/11.9 GPU-h. The default `--cost train` counts pure
training steps for both arms (5.1 vs 3.8/8.1/10.0). The earlier "13.1 GPU-h" LoMa figure
came from an older training run, not the one behind the paper result. Costs
count pure training only, on both arms: probe epochs use the tqdm training bar of array
508523 (per-epoch evaluation excluded), and LoMa checkpoint `epoch_E` uses the summed
`time/train_s` of epochs 0..E in `lynx-finetuning/logs/czechlynx-loma-ft/czechlynx-loma-ft-508111.out`
times 4 GPUs (validation excluded). Totals: LoMa 5.10 GPU-h for 300 epochs; frozen,
partial and full probes 3.8, 8.1 and 10.0 GPU-h for 50 epochs. The fine-tuned LoMa curve
starts at the default-LoMa result at 0 GPU-h. The script fails closed if a probe's
epoch-50 log value differs from its run's `top_1` or a LoMa checkpoint's hash changed,
and skips LoMa epochs whose evaluation has not completed. Known biases to state: LoMa trained on 80% of the train images and the
probes on 100%; 300 vs 50 epochs; one seed each.

**Fine-tuning GPU-hours, CzechLynx closed (extracted 2026-10-02 for the paper).** Pure
training steps x 4 RTX 4090 GPUs (`rtx4090_batch`), summed up to the epoch the paper probes
load; LoMa from per-epoch `time/train_s` in the `.out` JSON records, RDD from the `Epoch N: 100%`
tqdm bars in the `.err` files (lynx-finetuning/logs/czechlynx-{loma,rdd}-ft/). Matcher only:
LoMa 5.1 (job 508111, epoch 299), RDD 5.0 (508028, epoch 299). Descriptor only: LoMa 65.2
(509262, epoch 252), RDD 60.1 (509313, epoch 175; old recipe, effective batch 4). Joint:
LoMa 28.8 to epoch 100 (84.0 to epoch 299; jobs 521758+521759), RDD 35.1 to epoch 100
(63.8 to epoch 185; 521756+521757). Whole-job equivalents: 8.9, 6.8, 95.9, 96.0 (+3.0 for two
failed starts), 118.4 for all 300 joint-LoMa epochs, 100.8 for 186 joint-RDD epochs. The
descriptor runs and the RDD joint run stopped at the 24 h limit, before the 300-epoch cosine
schedule ended, and the joint probes so far use epoch 100.
Projected to the full 300-epoch schedule (epoch 299; measured epochs plus remaining epochs x
mean measured epoch time, which varies by 2-12% across epochs): LoMa descriptor 77.3, RDD
descriptor 102.5, RDD joint 102.9 GPU-h; LoMa joint is measured at 84.0 and both matcher-only
runs at 5.1/5.0. Label projected values as estimates in the paper.

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
automatic probe/Jaguar discovery searches the newest run for canonical model-only
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
Hydra is pinned to `hydra-core==1.3.2`; Vismatch is pinned to commit 4a743b75749a3770af59d275483ed341dea51ff0 and downloads matcher weights on first use. The shared ex-reid environment must have an importable, non-broken Vismatch installation; it must not depend on a missing editable checkout.
Default paths are specific to the original shared compute environment.

## Known issues (open)

Audited on 2026-08-17 and deliberately deferred. Each entry records the symptom, a
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
  as the finetune selection split.

- **Vismatch qualitative top-1 can point at an unscored pair.**
  When a query row is entirely `-inf`, `stable_rank_1d(...)[0]` returns database index 0
  and a meaningless match image is drawn instead of the query being skipped.

- **`_predict_class_probabilities` runs under grad during probe training.**
  `reid/engine/probe_runner.py:875` builds a graph for the softmax and then detaches it.
  Wasteful, not incorrect.

- **`_to_hwc_uint8` would destroy float images.**
  `reid/data/dataset_view.py:86` clips non-uint8 input to `{0, 1}` before masking. Not
  triggered today because the base dataset yields PIL images, but it would silently blacken
  inputs if a transform were ever applied before the view.

## Future-work checklist

- [ ] Add optional integration tests with a fake/local backbone and synthetic images.
- [ ] Add CI for unit tests, syntax checks, and YAML/config validation.
- [x] Migrate probe and finetuning configuration to Hydra with strict dotlist overrides and resolved snapshots.
- [ ] Replace environment-specific absolute paths with machine-local overrides.
- [ ] Pin external Git dependencies to reproducible commits.
- [ ] Implement truly disjoint calibration inputs for WildFusion and local matcher
  calibration; the current split setting selects one dataset and passes it to both
  sides of calibration.
- [ ] Include mask metadata/content fingerprints in standard and Vismatch feature
  caches so mask edits invalidate features, not only image-file edits.
- [x] Route visualization index rankings through the shared stable ranking helper;
  `visualizations/index.csv` now uses `stable_rank_1d`. Vismatch qualitative top-1
  selection still needs an all-unscored guard, tracked under Known issues.
- [x] Make `sort_run_rows` total: numbers sort in the requested direction, then text,
  then empty/`NaN` cells last in original order, so `scripts/summarize_runs.py --sort-by`
  works on mixed and gated metric columns.
- [x] Remove the unused per-epoch `similarity_epoch` allocation from classifier probes.
  The remaining per-epoch retrieval cost is tracked separately below and still needs
  optimization.
- [ ] Give the probes a validation split distinct from the query/test split, or stop
  logging per-epoch test metrics, so epoch and hyperparameter choices cannot use it.
- [ ] Make legacy checkpoint discovery recursive for the existing nested no-manifest
  `results/<dataset>/<animal>/mask_<...>/run_<...>` layout.
- [ ] Evaluate masking and Vismatch matcher settings separately for each animal dataset.
- [x] Fix runtime annotation import validation for the batched Vismatch path.
- [x] Add an ex-reid-gated runtime smoke test that invokes batched extraction.
- [x] Keep explicit batching defaults in the user-preserved probe YAML.
- [x] Add production-batched Vismatch feature extraction and feature-level reranking
  with a permanent serial parity/reference mode.
- [ ] Profile and optimize cached Vismatch feature extraction/reranking costs.
- [ ] Run the private Lynx golden-subset parity comparison for RDD-LightGlue before changing matcher defaults.
- [x] Run the available full-split Lynx parity comparison on 2026-08-12: 66 queries,
  217 gallery sequences, one frame per sequence; top-1 agreement was 65/66
  (98.48%), top-5 agreement was 100%, and mAP differed by -0.0018. The fixed
  golden-subset gate remains open because the comparison was not 100% top-1.
- [ ] Complete matcher ablations for RDD-LightGlue, ALIKED-LightGlue, SuperPoint-LightGlue, and LoMa-B.
- [ ] Track wrapped-model licenses and downloaded-weight provenance for paper release.
- [ ] Consider atomic checkpoint writes and explicit checkpoint retention.
- [x] Record completed finetuning total runtime in train_metrics.csv.
- [ ] Reconcile historical experiment metadata and stale generated CSV schemas.
- [x] Add readable experiment manifests, run indexing, visualization indexes, and summary reports.
- [ ] Make central run-index updates safe for concurrent jobs and use unique temporary
  files or locking instead of a shared `reports/runs.csv.tmp` path.
- [ ] Record SHA-256 checkpoint identities and repository dirty-state/diff identity in
  manifests so uncommitted experiment code remains reproducible.
- [x] Repair stale configuration expectations and synthetic-image fixtures so the full
  ex-reid regression suite is green before relying on it as a release gate.


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
submitted to ECIR 2027, under review) is double-blind and this GitHub repository is
public. All project-page files therefore live on the local `project-page` branch, which
must not be pushed, merged into `main`, or deployed before the notification without the
user's explicit instruction. **Venue rule (user decision 2026-10-02):** the site never names
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

**Page data and figures (step 2, done 2026-10-02).** `scripts/export_project_page_data.py`
reads the paper repository's `results/` directory (the frozen snapshot behind the
manuscript's tables and figures; its CzechLynx closed main CSV already differs from this
repository's live `reports/paper_tables/`), never the live tables, and writes committed JSON
under `docs/data/`: `results.json` (one record per dataset, method and budget), `curves.json`
(series and k-independent baselines per dataset plus the unseen-identity protocol),
`adapt.json` (Table 2 with the GPU-hours constants copied from the paper's `make_tables.py`),
`training_cost.json`, and `manifest.json` (source SHA-256 hashes, paper commit, conventions).
It applies the paper conventions: eight datasets, main k=250, matcher-only fine-tuned and
default Vismatch runs on MegaDescriptor-L candidates, class-weighted classifiers only
(unweighted and unknown-weighting rows dropped), descriptor and joint runs only in
`adapt.json`, no BelugaID and no `split-time_open`; it drops `manifest_path` and fails on
any `/shared/` or `/home/` fragment. On 2026-10-02 the export reproduced the paper's Table 2
and Table 3 values exactly. It also renders static page figures to
`docs/assets/figures/page/` as SVG in light and dark variants (Material's `#only-light` /
`#only-dark` image suffixes): small multiples of Top-5 and balanced Top-1 against k for
LoMa and of Top-5 for RDD, a default-to-fine-tuned dumbbell at k=250 (`gain_loma_k250`),
the "what to adapt" cost-against-gain scatter (`adapt_cost_k250`), the unseen protocol and
the training-cost curves. Chart conventions follow the dataviz validator run on
2026-10-02: colour by matcher family (LoMa blue, RDD red, WildFusion gold), fine-tuned
solid with filled markers, default dashed with hollow markers, baselines grey (emphasis
form); brand blue sits just under the validator's chroma floor and gold versus red is in
the CVD floor band, so line style and marker shape always carry identity too; the 40 %
tints fail the dark-surface checks and are not used for series. Small multiples omit the
cosine flat lines (they would squeeze the curves; the table carries them). Tests in
`tests/test_export_project_page_data.py` use a synthetic results directory and check the
committed `docs/data` files parse and contain no private paths. Rerun the exporter after
`git pull` in the paper clone whenever the paper's results change.

**Interactive components (step 3, done 2026-10-02).** Vanilla ES modules under
`docs/assets/js/`, registered in `mkdocs.yml` as `assets/js/page.js` with `type: module`
(the entry mounts each component where its `#wm-*` mount point exists and re-mounts on
Material's `document$`); `wm-common.js` holds the palette, series styles, theme detection
(`data-md-color-scheme` on `body`, re-rendering on change), data loading relative to the
module URL (`docs/data/<name>`), Plotly bootstrapping (CDN build `plotly-2.35.2`, deferred)
and small DOM helpers. `explorer.js` (`#wm-accuracy-explorer`) draws accuracy against k
from `curves.json` with dataset, metric, method and baseline toggles and a table view;
`results-table.js` (`#wm-results-table`) filters, sorts and downloads `results.json` with
budget-independent baselines shown with every budget; `training-cost.js`
(`#wm-training-cost`) draws `training_cost.json` with metric and linear/log axis toggles
(no whole-job accounting: the paper snapshot only carries training-step costs);
`match-viewer.js` (`#wm-match-viewer`) draws query and top-1 photos with correspondence
lines on a canvas from `docs/assets/match/match_examples.json`, which
`scripts/plot_match_examples.py render --web-export docs/assets/match` writes (web-sized raw
JPEGs, long side 1000 px, every correspondence in exported-photo pixels, confidence order);
until that GPU run happens the viewer shows a pending note. Chart conventions match the
static figures (colour by matcher family, solid/filled versus dashed/hollow, grey
baselines, hover for exact values, table views so no value is tooltip-only). JavaScript is
syntax-checked with `node --check`; there is no headless browser test, so visual checks
happen in `mkdocs serve`.

**Cross-check (step 4, done 2026-10-02).** Against the paper clone at `9e78102` and the
export: the abstract is verbatim; the dataset table equals `tab_datasets.tex`; Table 2,
Table 3 and the training-cost final points equal `adapt.json`, `curves.json` and
`training_cost.json`; the prose claims hold in the data (smallest k=250 Top-5 gain on Sea
star, +0.2; Top-5 up on all eight datasets at k=250; balanced Top-1 down only on Leopard
and Turtle; fine-tuned LoMa beats the fully fine-tuned weighted classifier on 8/8 in
balanced Top-1 and 7/8 in Top-5, Sea star going to the classifier; DINOv3-L cosine beats
MegaDescriptor-L cosine only on CzechLynx and Sea star; both fine-tuned matchers beat
default and WildFusion at every unseen-protocol budget in both metrics). The paper's
"two to five times longer candidate list" is its own summary: measured at k=250 the
default matcher needs 2x (CzechLynx, Hyena, Salamander, Sea star), 4x (Leopard, Whale
shark) or more than 4x (Nyala, Turtle), so the page quotes the paper's wording rather
than restating it. `tests/test_project_page_numbers.py` makes the table checks permanent
by parsing the Markdown tables and comparing them with `docs/data` (and with
`tab_datasets.tex` when the paper clone exists); it also pins the prose claims above.
The implementation-details sentence follows the pushed wording ("the training steps of
fine-tuning take 5.1 GPU-hours"). Match-viewer assets were exported by the user on
2026-10-02 (`docs/assets/match/`, 8 pairs, 2.6 MB, every keypoint inside its photo).
The paper's Conclusion (limitations and future work) was pushed as `226037a` later on
2026-10-02 and is mirrored as "Limitations and outlook" on the home page; it supersedes
the draft limitation bullets in the brief.

**Reproduce pages (step 5, done 2026-10-02).** `docs/reproduce/{index,mining,finetuning,
probing}.md` distil the three repositories' operating guides (`rdd-parallel-benchmark/README.md`,
`lynx-finetuning/README.md`, `CZECHLYNX.md`, `WILDLIFE.md`, and this repository's AGENTS.md)
into the pipeline used for the manuscript: canonical symlink views, 512 px / 512-keypoint
feature caches with the fine-tuning preprocessing, strong-match mining (20 images per
collection, 5 positives and 5 hard negatives per anchor, 10 anchors per collection, pretrained
weights only), the shared fine-tuning recipe and wrapper commands, the checkpoint layout with
protocol files, the descriptor and joint ablation variants, the probe commands, the two
launchers, the unseen-protocol generator, the metrics and the exporters. Cluster-specific
paths are written as `<placeholders>` (the leak test forbids real ones); the `legacy` protocol
is named as the paper's protocol and `strict` as unused. Metadata files per dataset come from
the launcher profiles.

**Synthetic match demo (requested by the user via the paper session, decisions 2026-10-02).**
`scripts/export_synthetic_demo.py` holds `INDIVIDUALS`: ten synthetic lynxes with two renders
each (`query` = render A, `gallery` = render B), read from the local dataset
(`/shared/sets/datasets/vision/czechlynx/CzechLynx_v2`, `CzechLynxDataset-Metadata-Synthetic.csv`,
40,000 rows with COCO-RLE `mask`; CzechLynx synthetic subset, Picek et al., Zenodo 17592004,
CC BY 4.0). Every query is matched against the ten gallery renders, so each query has exactly
one correct answer; visitors switch queries (user clarification 2026-10-02: ten *queries* to
switch between, not ten gallery items for one query). The first six individuals (173, 129,
242, 88, 79, 138) and lynx 173's two renders are the paper's teaser selection (paths from the
paper's `results/teaser_prep.py`); the other query renders (lynx 27, 174, 289, 45) and all
other gallery renders were chosen by appearance only, before any matching, with thresholds
calibrated on the paper's renders: side view (mask box at least 1.25x wider than tall),
blue-pixel share <= 0.0066 and strongly saturated share <= 0.55 inside the mask, mask share
0.18-0.42, long side 450-1400 px, gallery renders preferring a different scene than the query
render, all accepted from contact sheets by eye; scores were never used for selection. A
first random pick of extra individuals (lynx 265, 226, 101, 206; mask share and size only)
was rejected by the user because the renders had unnatural coats (blue patches, dense
high-contrast spots) and two faced the camera, and the fine-tuned matcher gave them hundreds
of confident matches (scores 0.29-0.62 against the lynx 173 query) although the correct
render still ranked first. Result of the ten-query export (2026-10-02, kept as it came):
7 of 10 queries rank their own individual first (scores 0.61-0.83, 350-440 matches); lynx 242,
79 and 45 rank another lynx first by 0.004-0.025 (242 -> 289, 79 -> 27, 45 -> 27), and
cross-individual scores are generally high (0.5-0.67) because synthetic coats come from a
shared texture model, unlike the paper's six teaser negatives against the snow query, which
score 0.001-0.016. The home page states this; never present the demo as a benchmark. The
matcher is configured from the paper run `20260920T122915Z_0015f14a` (same checkpoint as the
k=250 run `20260920T131759Z_612b4791`: `loma-b-finetuned-loma-mined-legacy/epoch_299`,
matcher only, LoMa-B, 512 px, 512 keypoints, default threshold 0.10); the model input is the
raw render with the dataset mask applied (black outside, exactly like `BenchmarkDatasetView`),
so keypoints map straight onto the raw render, which the page displays resized to a 1000 px
long side. The user allowed running the export on the login node's V100 (2026-10-02). Output
`docs/assets/demo/synthetic/` holds twenty JPEGs (`Q_<identity>.jpg`, `G_<identity>.jpg`) and
`synthetic_demo.json` (`gallery`, `queries` with ranked `candidates`, `correct_rank`, every
correspondence, confidences and confidence order, model settings and checkpoint SHA-256,
attribution, `synthetic: true`). User decisions: fine-tuned matcher only (no default-LoMa
comparison, because the internal expert study found fine-tuned matches less intuitive and a
side-by-side would invite that reading); strongest-N slider (default 10) with hover
confidence; first placed on the home page below the hero, then moved to its own
`docs/demo.md` page (nav tab "Demo" after Home, a "Demo" hero button and a one-line pointer
on the home page) on 2026-10-02 because the home page had become dense (user request), and
then to `docs/demo/synthetic.md` when the Demo page was split into a section (see "Demo
section" below);
labelled synthetic with attribution.
`synthetic-demo.js` (`#wm-synthetic-demo`) renders two compact filmstrips (inline labels "Query" and "Gallery, ranked";
small thumbnails; captions `lynx N` and `#rank · score`; blue active frame; the gallery render
of the query's own individual has a red inner frame; details in the title tooltip), the pair canvas with confidence-weighted lines and hover tooltips, and shows a pending
note until the export exists. `tests/test_export_synthetic_demo.py` covers the GPU-free
helpers (mask application, RLE decoding, row lookup, ranking, payload, `INDIVIDUALS`
consistency) and validates the committed export when present.

**Background-masking demo (piece 1 of the SAM 3 demo plan, started 2026-10-02).** The user
asked for a SAM 3 capability demo; of the recommended pieces (1 before/after slider with
readouts, 2 mask-to-matching comparison, 3 honest hard cases, 4 prompt sensitivity) piece 1 is
built first, on the CC BY 4.0 synthetic renders because the SalamanderID2025 photographs are
Kaggle competition data whose rules (Section 2.4b, read 2026-10-02) forbid publishing or
redistributing the data to non-participants; written permission from the sponsor
(University of West Bohemia, Picek's group) is needed before any Salamander photograph,
including the match-figure pair and the match-viewer pair already on the page, can go
public. `scripts/export_masking_demo.py` (CPU) reads the SAM 3 output of
`scripts/segment_with_sam3.py --segment` for the twenty match-demo renders
(`--write-renders-csv` writes their list; the run itself needs the `lynx-app` environment on
an A100/H100 node, submitted by the user, default output `reports/project_page/sam3_synthetic/`)
plus the dataset's own masks, and writes `docs/assets/demo/masking/`: web renders, binary
SAM 3 and dataset mask PNGs at web size, and `masking_demo.json` (per render: prompt,
threshold, confidence, merged instance count, foreground share, IoU with the dataset mask;
summary mean/min IoU). `masking-demo.js` (`#wm-masking-demo`, Demo page section "Background
masking with SAM 3") shows a render filmstrip, a canvas with a draggable divider between the
raw render and the masked model input (composited client-side from the mask PNG), SAM 3
(red) and dataset (blue) mask outlines, and the readouts. Tests in
`tests/test_export_masking_demo.py` run the export end to end on synthetic inputs and validate
the committed export when present. The Salamander images drop in later by pointing
`--root/--metadata/--sam3-dir/--renders` at the Salamander data, once permission exists.
Run on 2026-10-02 by the assistant on the `dgxh100` node at the user's request, with the
prompt "animal" (the paper's prompt, user decision), merge `union`, threshold 0.5: all twenty
renders detected in one instance with scores 0.945-0.969, no fallbacks; IoU with the dataset
masks 0.926-0.982, mean 0.960. The lowest values are snow renders where the dataset mask
includes the cast shadow and SAM 3 does not, so the disagreement is in the reference mask, not
a SAM 3 failure. The SAM 3 outputs stay under the gitignored
`reports/project_page/sam3_synthetic/`; the export is committed.
**Real photographs added 2026-10-02.** The SalamanderID2025 team allowed showing its photos on the
paper and the research page (user report, 2026-10-02), and the user asked for a real-photo
group covering every paper dataset with two or three hand-picked images each, so the demo has
two groups (tabs in `masking-demo.js`): `synthetic` (the twenty renders) and `real`, the
sixteen match-figure photos (query and top-1 per dataset, the user's own picks; `real_items()`
resolves raw paths from `docs/assets/match/match_examples.json` and `PAPER_PROFILES`). A
reference mask and IoU appear only where the dataset itself ships segmentation masks, i.e. the
synthetic subset and CzechLynx (RLE); the WildlifeReID-10k pre-masked images and the
SalamanderID2025 masks were computed by us, so those photos show the SAM 3 mask alone (user
correction 2026-10-02, replacing a first version that compared against those computed masks).
`dataset_mask`, `iou_with_dataset_mask` and `reference_mask_source` are `null` for them and the
group summary counts `items_with_reference`. SAM 3 runs
(`--write-real-csvs` writes one render list per dataset root under
`reports/project_page/sam3_real/{czechlynx,wildlife,salamander}/`; run by the assistant on
`dgxh100` with prompt "animal") detected every photo except the two sea stars, which
"animal" misses even at the 0.25 fallback threshold; the WildlifeReID-10k run was repeated
with `--fallback-prompts "sea star"`, which segments them (scores 0.94/0.96, recorded as the
prompt used). CzechLynx IoU with its dataset masks: 0.992 and 0.989. The page states the sea-star
fallback.
Licences: CzechLynx photos and the WildlifeReID-10k sub-datasets other than Hyena/Leopard
(CDLA-Permissive) are still unverified for web display; the user chose to include them.

**Background-match analysis (user hypothesis, 2026-10-02).** `scripts/analyze_background_matches.py`
(GPU, ex-reid) matches raw, unmasked WildlifeReID-10k photo pairs with the default and a
fine-tuned LoMa matcher, classifies each correspondence endpoint as animal or background with
the provider's pre-masked image (`max(RGB) > 12`, so dark animals are undercounted), draws a
contact sheet (cyan = both endpoints on the animal, orange = at least one on the background)
and also reports each matcher's score on the masked inputs for reference; outputs go to the
gitignored `reports/project_page/analysis/`. Run on Nyala (checkpoint
`model__actual_nyala.safetensors`, SHA `c0cbe570...`, 6 same-individual pairs including the
match-figure pair, 3 different-individual pairs, seed 0, H100). Hypothesis: a matcher fine-tuned
on masked images would focus on the animal when the background is present. Result: not
supported. The detector places only 16-50 % of keypoints on the animal in raw photos; the
default matcher keeps 40 % of its matches on the animal on average (scores 0.04-0.12), the
fine-tuned one 38 %, and the fine-tuned matcher is bimodal on raw inputs: either hundreds of
matches with 72-97 % touching the background (two same pairs scoring 0.59/0.63, and one
different-individual pair scoring 0.263 through background matches) or a near-total collapse to
1-35 matches on pairs whose masked-input score is 0.54-0.60. Masks are therefore required at
inference for the fine-tuned matcher, consistent with the paper's limitation on mask
dependence; nothing suggests an animal bias. Nine pairs, one seed.
CzechLynx repeat (same day, `--mask-col mask --identity-col unique_name --split-col
split-time_closed`, exact RLE masks, closed-split checkpoint `epoch_299`, the match-figure pair
plus 8 same and 4 different random pairs, seed 1): default 49 % of matches on the animal,
fine-tuned 39 %; the fine-tuned matcher returns 280-390 matches on 7 of 9 same pairs with
61-83 % touching the background, scores two different-lynx pairs photographed in similar
scenes 0.52 and 0.62 through background correspondences (0.003 and 0.001 on masked inputs),
and collapses to 3-13 matches where no shared scene exists. Confidence gained by fine-tuning
flows into repeated background texture when the mask is absent, so masking at inference is a
hard requirement; this is the mechanism behind the paper's mask-dependence limitation. The
script takes `--mask-col/--identity-col/--split-col` so either layout works, and
`--joint-checkpoint` adds the joint descriptor + matcher checkpoint (`checkpoint_components=full`)
as a third column. Same CzechLynx subset with the joint `epoch_299` (SHA `a91cbac9...`,
2026-10-02): on raw photos the joint model behaves like the matcher-only one (36 % of matches on
the animal versus 39 % and 49 %, 200-370 matches with 59-83 % touching the background on most same pairs, mean raw
score 0.36 same / 0.16 different versus 0.50 / 0.29 for matcher-only and 0.07 / 0.03 for
default); it scores the two shared-scene different-lynx pairs lower (0.013, 0.161 versus 0.524,
0.617) but two other different pairs higher (0.285, 0.178 versus 0.003, 0.002), one of which it
also scores 0.132 on masked inputs. Training the descriptor therefore does not restore an
animal focus without masks either; keypoint locations are identical across the three models
because the detector stays frozen.

**Before/after demo (user request 2026-10-02).** `scripts/export_before_after_demo.py` (GPU)
matches each of the eight match-figure pairs (`reports/figures/match_examples.json`: query and
correct top-1 photo per dataset, the user's picks) twice on the background-removed inputs the
pipeline uses, with the default LoMa matcher and with the dataset's fine-tuned matcher (paths
and SHA-256 from the sidecar, verified before loading; Nyala's is `model__actual_nyala`), and
writes `docs/assets/demo/before_after/` (web copies of the raw photos, 1000 px long side, and
`before_after.json` with both results per pair: score, match count, every correspondence in
exported-photo pixels, confidences, order). Matches are computed on masked inputs and drawn on
the originals, as in the paper's qualitative figure; raw-input matching was rejected after the
background analysis above. Single-image extraction moves a few keypoints relative to the
probe's batched extraction, so fine-tuned scores differ from the probe's recorded ones by up
to 0.02 (Czech Lynx 0.731 vs 0.734, Hyena 0.604 vs 0.587, Turtle 0.666 vs 0.646). Export of
2026-10-02 (default -> fine-tuned, score and matches): Czech Lynx 0.072/130 -> 0.731/402,
Hyena 0.106/127 -> 0.604/358, Leopard 0.064/103 -> 0.664/384, Nyala 0.098/134 -> 0.662/378,
Salamander 0.233/205 -> 0.653/381, Sea Star 0.415/331 -> 0.772/425, Turtle 0.044/80 ->
0.666/374, Whale Shark 0.199/190 -> 0.655/370. `before-after-demo.js` (`#wm-before-after`,
first section of the Demo page, `data-pair` sets the initial pair, Czech Lynx) shows a two-button
matcher toggle with each matcher's score and match count, the pair canvas with
confidence-weighted lines and hover confidences, a strongest-N slider (default 30) and a pair
selector over the eight datasets. This is the one place the page shows the default matcher
next to the fine-tuned one; the user asked for it explicitly as the most direct visual
argument. Tests in `tests/test_export_before_after_demo.py`.

**Mined-pairs browser (user request 2026-10-02).** `scripts/export_mined_pairs_demo.py` (GPU for
the correspondences; `--no-matches` for CPU) joins the CzechLynx time-closed LoMa mining
outputs in `rdd-parallel-benchmark/outputs/czechlynx-time-closed/legacy/loma/`: the combined
training index (`strong-matches_train_combined.json`, 3,009 anchors, paths relative to the
canonical view `CzechLynx_processed_time_closed`) with the 360 per-collection reports
(`strong-matches_train_<id>.json`), which keep every mined pair's pretrained-LoMa score. View
frames are symlinks into `CzechLynx_v2/CzechLynx_masked/...`; the raw photo is the same file
under `CzechLynx_v2/CzechLynx/...` (verified for all anchors). Anchors are chosen by
appearance only (full 5+5 pools, one per individual, warm-pixel share >= 0.5 and animal share
>= 0.10 on the anchor's raw photo, seed 3, ten anchors); partners are exactly the mined ones,
including night and infrared frames. Each anchor/partner pair is re-matched with the
pretrained matcher on the masked images so the page can draw the correspondences behind the
score; recomputed scores differ from the mined ones by up to 0.04 (single-image extraction
versus the mining run's cached features), and both are stored. Output
`docs/assets/demo/mined_pairs/` (109 photos at 560 px long side, `mined_pairs.json`).
`mined-pairs-demo.js` (`#wm-mined-pairs`, Demo page section "Mined pairs: what weak supervision
looks like", after the before/after demo) shows an anchor filmstrip, two rows of five cards
(positives blue, hard negatives red) with score bars and identities, and the pair canvas with
hover confidences for the selected partner. In 530 of the 2,830 full-pool anchors the best
hard negative outscores the best positive under the pretrained matcher (median mined scores
0.107 positive vs 0.098 negative), which is why the page says the pools overlap before
fine-tuning. Tests in `tests/test_export_mined_pairs_demo.py`.

**Rank-change explorer (user request 2026-10-02).** `scripts/export_rank_change_demo.py` ranks every
CzechLynx closed test query under the paper's three k=250 runs: cosine
(`cosine/default/20260919T223430Z_5463f66e`, whose dense score matrix was never persisted, so
the MegaDescriptor-L embeddings are re-read from the probe's own feature cache through
`probe_runner.load_dataset_splits/load_backbone/build_transforms/make_dataset_view/
extract_deep_features_with_cache` with the run's `config.snapshot.yaml`; both cache lookups hit
on 2026-10-02, so no extraction ran), default LoMa (`vismatch/loma/20260920T131759Z_daf95fda`)
and fine-tuned LoMa (`vismatch/loma/20260920T131759Z_612b4791`), the last two from their
`scores.npz` shortlists. Ranking uses the shared stable rule; the true rank is the first gallery
position with the query's identity, `null` when the identity is outside the 250 shortlist.
Rank-1 counts over all 11,924 queries reproduce the paper exactly (cosine 1,934 = 16.2 %,
default 5,480 = 46.0 %, fine-tuned 5,858 = 49.1 %). Categories by rank-1 correctness: rescued
(fine-tuned right, default wrong) 1,020; regressed 642; already right 4,838; still wrong 5,424.
The page shows these whole-split counts and a seeded random sample (8 rescued, 8 still wrong,
4 regressed, 4 already right; seed 5), never an appearance-based pick, so failures are shown
honestly. Output `docs/assets/demo/rank_change/`: 346 raw photos at 320 px and
`rank_change.json` (per query: identity, category, per method true rank and top 5 with
identity, score, correct flag). `rank-change-demo.js` (`#wm-rank-change`, Demo page section
"Rank changes", before the mined-pairs browser) has category tabs with counts, a query
filmstrip and three method columns of five ranked thumbnails with the true individual framed
blue. Tests in `tests/test_export_rank_change_demo.py` check the stable shortlist ranking, the
category logic and the committed export's internal consistency (counts add up, top-1 flags
match true ranks).

**Demo section (user request 2026-10-02, replacing the single cluttered Demo page).** The
demos live on their own pages under `docs/demo/` and the "Demo" tab opens a hub,
`docs/demo/index.md` (Material `navigation.indexes`), with one card per demo (thumbnail,
one sentence, link) in the paper's narrative order chosen by the user: `before-after.md`,
`rank-changes.md`, `mined-pairs.md`, `masking.md`, `synthetic.md`; Material's prev/next
footer walks them in that order. Each page keeps its former section text and mount point,
opens with a one-line "What to look for" lead (`{ .wm-lead }`) and ends with a button back
to the hub; the hub carries the shared "How to read the demos" notes and attributions, the
synthetic page keeps its own "How to read it". `page.js` is unchanged: it mounts whatever
`#wm-*` mount point the current page has, so each demo page loads only its own widget.
Card thumbnails (480 x 300 JPEG, `docs/assets/demo/cards/<page>.jpg`) are composed by
`scripts/build_demo_cards.py` (CPU, Pillow) from photos already committed under
`docs/assets/demo/`, never from dataset files: the Czech Lynx before/after pair, the first
rescued rank-change query with its fine-tuned top 3 (blue rule under correct ones), the first
mined-pairs anchor with two positives (blue) and two hard negatives (red), the lynx 173
gallery render half raw / half SAM 3-masked, and lynx 173's two renders. Rerun it after a
demo re-export. Hub card styles are `.wm-hub` / `.wm-hub-card` in `extra.css` (brand-blue
hover and image rule). `tests/test_build_demo_cards.py` covers the compose helpers, the
fail-closed builders and the committed cards against the hub's references and pages.
Home-page links point at `demo/index.md`. Strict build verified 2026-10-02.

**Score-separation view (user request 2026-10-02, "shows directly what the margin loss
does").** `scripts/export_score_separation.py` (CPU) reads, for each of the eight paper
datasets, the paper repository's `results/<stem>_ablation.csv` to pin the exact default and
matcher-only fine-tuned LoMa runs at k=250 (`run_id`, `manifest_path`), loads each run's
`scores.npz` (all 250 shortlist pairs per query) and derives the identity order from the
run's `config.snapshot.yaml` metadata filter (the same filter as `load_dataset_splits`).
It fails closed unless the manifest is completed with the recorded run id, the matrix shape
equals the split sizes, every score is in [0, 1], and the Top-1 recomputed with the shared
stable rule equals both the run's `metrics.json` and the paper CSV (this verifies the label
order; all sixteen runs passed on 2026-10-02). Output `docs/assets/demo/score_separation/
score_separation.json`: per dataset and matcher, 50-bin histograms on [0, 1] of same- and
different-individual pair scores, a 50-bin histogram on [-1, 1] of the per-query margin
(best same-individual score minus best different-individual score, over queries whose
shortlist holds their individual), and statistics (AUROC with average-rank ties, medians,
means, histogram overlap coefficient, positive-margin fraction). Population caveat stated on
the page: "different individual" means hard shortlist candidates, not random pairs; both
matchers score the identical pairs. Results: AUROC rises and overlap falls on all eight
datasets (CzechLynx 0.591 -> 0.772 and 0.820 -> 0.603; Nyala 0.634 -> 0.863; Salamander
0.933 -> 0.971; Sea star 0.880 -> 0.909), while the positive-margin fraction falls slightly
on Leopard (0.851 -> 0.838) and Turtle (0.846 -> 0.797), the paper's two balanced-Top-1
exceptions; the page says so. `score-separation-demo.js` (`#wm-score-separation`, section "Score separation" of
`docs/results.md`, after the main results and before "What to adapt"; first built as a Demo
page and moved the same day because the user judged it a result, not a demo) draws two Plotly
panels (default | fine-tuned, blue same / red different, overlaid bars) with dataset and
view selectors (pair scores or per-query margin, zero line), share/count and log-axis
toggles, hover counts and shares, and a statistics table (AUROC, medians, overlap, counts;
in margin view: queries with margin, median, positive fraction, recorded Top-1). It has no Demo hub card. Tests in
`tests/test_export_score_separation.py` (statistics helpers, stable top-1, margins, and
the committed export's internal consistency including AUROC up / overlap down everywhere).
Rerun the exporter after the paper's k=250 LoMa runs change (e.g. after the RDD retrain, no
change, since it reads LoMa only).

**Candidate-budget trade-off (user idea 2026-10-02, "makes the cost of k concrete").**
`scripts/export_budget_tradeoff.py` (CPU) writes `docs/data/budget_tradeoff.json`: for each
paper dataset and each measured budget k in 10/50/100/250/500/1000, the shortlist share
(`candidate_recall_at_k`, identity-level, asserted equal between the default and fine-tuned
run because both rank the same candidates), Top-5 of both matchers (paper CSV), the Vismatch
matching time (`vismatch_rerank_sec`; feature extraction, candidate selection, setup and
cache I/O excluded) normalised to minutes per 1,000 queries and ms per pair, and the
hardware per run: Slurm job from `logs/index.csv`, partition from `sacct` at export time
(`--no-sacct` skips it), mapped to a GPU name. Nothing is interpolated. Hardware found on
2026-10-02: CzechLynx, Hyena, Leopard, Sea star and the Turtle fine-tuned runs on
`rtx4090_batch` (1.80 ± 0.03 ms per pair), Salamander on `dgxh100` (1.37 ms), the Turtle
fine-tuned k=10 run on `dgxa100` (2.85 ms), and the Nyala, Whale shark and Turtle default
runs of 2026-08-21/22 without task records (pace 1.8 ms, matching the RTX 4090 runs, but
reported as "not recorded", never inferred). Per budget, `display` shows the mean of both
runs' times when they share a partition, else the run on the dataset's dominant partition
(falling back to the default run), and the page prints the GPU next to every time.
`budget-tradeoff.js` (`#wm-budget-tradeoff`, Results page section "The cost of k" between the
accuracy explorer and "All numbers") has a dataset selector, a slider that snaps to the six
budgets, four readout tiles (shortlist share, Top-5 default, Top-5 fine-tuned with the gain,
matching minutes per 1,000 queries with ms per pair and GPU) and a two-panel Plotly chart
(accuracy: shortlist share grey dotted, Top-5 default dashed hollow, fine-tuned solid; cost:
minutes in gold) with a red dashed marker at the chosen k. Tests in
`tests/test_export_budget_tradeoff.py` pin the normalisations, the display choice, and in
the committed export that the shortlist share and matching time grow with k and that Top-5
never exceeds the shortlist share. Rerun after the paper's LoMa runs change.

**Rare and common individuals (user idea 2026-10-02; recomputes the earlier ad hoc long-tail
check, which was never persisted).** `scripts/export_frequency_bins.py` (CPU) reuses the
score-separation exporter's run pinning, label loading and fail-closed checks for the paper's
k=250 default and fine-tuned LoMa runs, bins every query by the number of gallery images of
its individual with fixed edges shared by all datasets (1, 2-4, 5-9, 10-29, 30+; queries whose
individual has no gallery image are excluded and counted, 0 on every paper dataset), and
writes `docs/data/frequency_bins.json`: per bin and matcher Top-1 and Top-5 (shared stable
rule, Top-5 from the first five stable positions), the shortlist share (identical for both
matchers, asserted), the fine-tuning gain with a paired percentile bootstrap 95 % interval
(2,000 resamples, seed 0) and the identity count; bins under 20 queries are flagged `small`.
The overall Top-1 reproduces the recorded run metric (test). Findings (2026-10-02): in Top-5
no populated bin on any dataset loses, and the largest gains often fall on singletons
(Leopard +9.8, Nyala +14.3, Whale shark +11.1, Turtle +25.6 points); CzechLynx gains a near
constant +1.7 to +3.2 in every bin (its gallery has no singleton identities, so that bin is
empty); the Top-1 losses of Leopard and Turtle sit in the 2-29 bins (Turtle 2-4: -5.2
[-7.6, -3.0]; 5-9: -6.1; 10-29: -6.3) while their singletons and 30+ bins hold or gain. The
user's recollection "helps every bin about equally" holds for CzechLynx and for Top-5 in
direction, not in size; the page states the measured pattern. `frequency-bins.js`
(`#wm-frequency-bins`, Results page section "Rare and common individuals" after "Score
separation") draws grouped bars per bin (default hatched and translucent, fine-tuned solid,
both brand blue; small bins fainter), the shortlist ceiling as grey diamonds, the gain in
points above each bin, a table with intervals, and dataset/metric selectors defaulting to
Top-5 (the paper's primary metric). Tests in `tests/test_export_frequency_bins.py`.

**Datasets section and Data challenges page (user request 2026-10-02).** `docs/datasets.md`
moved to `docs/datasets/index.md` (nav section "Datasets" with `navigation.indexes`; links in
`qualitative.md`, `results.md` and `tests/test_project_page_numbers.py` updated) and gained a
subpage `docs/datasets/challenges.md` that shows the confirmed examples behind the paper's
data-quality figure in far more detail. `scripts/export_data_challenges.py` has two steps:
`candidates` ranks images per category from the image-quality audit CSVs
(`experiments/image-quality/<dataset>.csv`, original metadata `row_index`; the audit drops
rows outside the two split sides, so containment, not equal length, is checked), resolves
the RAW photo (never the masked input: `original_path` for Salamander, `masked_images/` ->
`images/` for WildlifeReID-10k, the metadata path for CzechLynx) and writes review contact
sheets to the gitignored `reports/project_page/data_challenges/`; `render` exports the
`EXAMPLES` list (web JPEGs at 900 px and `challenges.json` with audit measurements, flags,
query Top-1 over completed runs and source SHA-256) to `docs/assets/datasets/challenges/`
(48 photos, 3.2 MB) and the audit summary to `docs/data/image_quality_summary.json`
(`summary.csv` rows of the eight page datasets, `lynx_closed` for CzechLynx, BelugaID
excluded). Categories: noise = overexposure, empty frame, insufficient detail, corruption,
blur; difficulties = night/infrared frames, occlusion; mask-only problems (provider mask
failures on Sea star and Whale shark, masks on branches, finger-split masks, dark animals
undercounted by the brightness threshold) are text only, following the paper figure's rule.
The 48 examples were proposed by the assistant from the sheets and approved by the user on
2026-10-02; rejected as false positives: Hyena overexposure candidates (identifiable), lynx
frames with the animal at the edge (truncation, not emptiness), dark Leopard/Hyena frames
flagged empty, Whale shark spot close-ups flagged tiny (provider mask specks), all Nyala
tiny/blur candidates (full-body, sharp), CzechLynx 29895 (purple IR cast, not corruption),
Salamander 63 (no finger visible) and 1280 (uncertain). `data-challenges.js` mounts one
gallery per category (`#wm-challenge-<key>`, raw photo, caption with dataset, side, the
ranking measurement and query Top-1 where available, full-size copy on click) plus
`#wm-image-quality-table` (per-dataset flag counts, flagged share, flagged vs clean query
Top-1, runs). Tests in `tests/test_export_data_challenges.py` (examples valid and unique,
every category has examples and a mount point, export matches `EXAMPLES`, no private paths).

**Review pass and restructuring (2026-10-03, user asked for one pass and approved every
suggestion).** Nav reduced from ten tabs to seven: Results is a section (`docs/results/index.md`,
`results/qualitative.md`, `results/training-cost.md`; the former `results.md`, `qualitative.md`
and `compute.md`) and Code merged into `docs/paper.md` as "Paper & Code" (sections Citation,
Acknowledgements placeholder, Contact, Code with the repository map and third-party table);
`code.md` is gone and the home hero has four buttons. Results order: at-a-glance tiles
(`#wm-results-glance`, `results-glance.js`: datasets with higher Top-5, median Top-5 and
balanced Top-1 gains with ranges, training GPU-hours from `adapt.json`), main results, the
explorer, all numbers, unseen identities, applicability beyond LoMa, what to adapt, matcher
versus classifier, then "Why it works" with score separation, rare and common individuals and
the cost of k as subsections (intro tightened). The Qualitative page lost its match viewer
(`match-viewer.js` deleted) because the before/after demo shows the same eight pairs with
more information; it links there instead. The home pointer now sends readers to the
before/after demo rather than the synthetic one. Web copies of the two heavy PNG figures
(`match_examples_web.jpg` 283 KB, `data_quality_examples_web.jpg` 96 KB, 1,600 px wide,
made with Pillow from the paper PNGs) replace the 2.7 MB and 2.2 MB originals on the pages;
the originals stay in `docs/assets/figures/` for the manuscript. The licensing table says the
synthetic renders are used in demos, Salamander is shown with the team's permission under
Kaggle's redistribution rules, and CzechLynx and the other WildlifeReID-10k photographs await
confirmation. `overrides/main.html` adds Open Graph and Twitter card tags with
`docs/assets/logo/wildmatch-social.png` (1200 x 630, the full-name logo, which already carries the tagline, centred on
white over a brand-blue rule; rendered with Pillow). `test_project_page_numbers` reads the
moved pages.

**Video explainer (started 2026-10-03, user request).** A 70-80 s 3Blue1Brown-style animation
for the home page, built with Manim Community and narrated with Kokoro-82M (open weights,
stock voice, user decision over ElevenLabs). Source lives in `video/explainer/` (outside
`docs/` so MkDocs does not render it): `script.md` holds the narration and shot list, which
the user approves before any rendering. Rules: no venue or submission status, no author
names on screen, only two spoken numbers ("eight datasets", "about five GPU-hours"), real
correspondences and mining pools reused from the demo exports, captions and a transcript
under the embed, the MP4 hosted unlisted and never committed, unlisted until the
notification date like the rest of the page. Script approved by the user on 2026-10-03.
Tooling runs in the separate `wm-video` conda environment (user approval 2026-10-03; recipe
and install pitfalls in `video/explainer/ENVIRONMENT.md`: Manim 0.21.0, Kokoro 0.9.4, ffmpeg,
pango with harfbuzz for the ManimPango build; voice `af_heart`, speed 0.95, about 134 words
per minute). The `ex-reid` environment is unchanged.
First full render on 2026-10-03: `video/explainer/explainer.py` (one `Explainer` scene, seven
shots cut to `audio/timings.json` through `fill_to`, which waits until the renderer clock
reaches each shot's end; brand palette; `Text` only, no LaTeX, because the environment has
none), `build.py` (muxes `audio/narration.wav` onto the Manim render with ffmpeg, writes
`out/wildmatch_explainer.{mp4,srt}` and `out/transcript.md`; `out/`, `media/` and the WAVs are
gitignored, `narration.mp3` and `timings.json` are committed). Shot 6 shows this query's real
top-5 under both k=250 runs from `candidates.json` (`candidates.py`, ex-reid): the default
matcher already ranked lynx 041 first (two of five correct, 0.11 vs 0.07) and the fine-tuned
one has four of five correct (0.74 vs 0.69), so the shot 6 narration was changed from "the
right individual rises to the top" to "photos of the right individual fill the top of the
candidate list" and only that shot was re-synthesised (total 78.5 s). Render command and
checks: `manim -qh --fps 30 --media_dir media -o explainer.mp4 explainer.py Explainer` then
`python build.py`; frames inspected per shot before the full render (headers that overflowed
the frame were shortened). User review of the first render (2026-10-03) asked for: a cleaner lynx pair without
padding bands or camera stamps, ten different lynxes in the database shot, cleaner pool
photos, non-overlapping labels in the margin shot, one photo per dataset instead of arrows,
and a photo collage fading behind the logo. Second render the same day: the pair is now
lynx 041 (query 23504, gallery 18829, daylight, both photos over 600 px), chosen from
`reports/project_page/explainer/pair_sheet2.jpg`, a sheet of the daylight match-candidate
pairs filtered by size, padding and by their real k=250 lists (fine-tuned top 5 must hold
more correct photos than the default's: this pair goes from 2/5 at 0.05 to 5/5 at 0.72-0.65);
the two earlier picks were rejected because lynx 133's fine-tuned list held one correct photo
and lynx 086's query was 208 px. The pair is matched on the CPU with both matchers through
`scripts/export_before_after_demo.export(..., device="cpu", sidecar=video/explainer/pair/
sidecar.json)` into `video/explainer/pair/` (recorded probe score 0.720, recomputed 0.713);
`candidates.py <before_after.json>` writes its real lists. `video/explainer/assets/` holds
ten database tiles of ten different individuals picked by eye from the mined-pairs photos
(`lynx_identities_sheet.jpg`) and the lynx 029 pools (the cleanest anchor row on
`anchor_pools_sheet.jpg`; thin padding bands trimmed, the anchor's bottom stamp strip cropped
for display only). The scene trims the pair's padding bands and maps correspondences through
the crop box, puts the "positive" label above the brace and "hard negative" below the ticks,
shows the eight datasets as the before/after query photos, and ends with an 18-photo collage
at 18 % opacity under a translucent panel with the logo. The user approved the second render and asked for three more things (2026-10-03), done in
the third render: the database rows no longer collide with the name tags (row gap 0.85),
burned-in subtitles in a reserved bottom band on every shot (`install_captions`: a
`VGroup` with an updater that swaps the caption on the renderer clock, using
`build.split_caption` at 92 characters; content moved up to leave the band free; the
captions are brought to the front over the closing collage), and the typeface is Computer
Modern, the 3Blue1Brown/LaTeX face: `fonts/cmr10.ttf` and `cmss10.ttf` copied from
matplotlib's bundled fonts (AMS/Knuth licence) and registered with `manimpango` at import;
no LaTeX is needed. The user then reported old captions showing through new ones; the
cause is Manim's Cairo renderer, which lists the moving mobjects' family members once per
`play()` and keeps drawing them, so a caption replaced by swapping submobjects stayed
visible behind the new box. `install_captions` now keeps one persistent box and text and
morphs them with `become()` (opacity 0 between captions), and the 0.25 s caption overlap
was removed; all 19 boundaries were checked frame by frame. Still open: user review of the fourth render, hosting (unlisted), the
home-page embed with a click-to-load facade and the transcript, and a test that pins the
transcript's two numbers.

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
- [ ] Merge `project-page` into `main` and run `mkdocs gh-deploy` (user) only after the
  notification date; verify the social card and favicon on the live URL.

**Plan.** Step 1 (done 2026-10-02): scaffold with real text from the brief, static
copies of the current figures under `docs/assets/figures/`, mount points
(`#wm-accuracy-explorer`, `#wm-results-table`, `#wm-match-viewer`,
`#wm-training-cost`) for the interactive components, and Reproduce stubs. Step 2 (done, see above; the match-example photos for the viewer are still to be
exported from `match_examples.json`). Step 3 (done, see above). Step 4 (done, see above). Step 5 (done, see above). Step 6: the publication checklist above.

## Vismatch matcher policy

The public local-matcher method is `vismatch`; the selected matcher is configured
under `benchmark.methods.vismatch.matcher`. Supported initial profiles are
`rdd-lightglue`, `aliked-lightglue`, `superpoint-lightglue`, and `loma` (Vismatch's
LoMa-B wrapper). The production path extracts features once and matches cached features. `feature_matching_mode: feature_level`
is required for production; pairwise Vismatch calls are reserved for explicit diagnostics.
Old `rdd` method names and direct RDD repository paths are unsupported and receive a migration-specific error.
The `FrameFeatures` contract and matcher profiles remain dependency-light so unit tests can run without Vismatch, CUDA, downloaded weights, or masking packages. RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue use the pinned Lynx-compatible preprocessing: RGB float32 tensors in `[0,1]`, direct bilinear tensor resize to the configured target long side, and floor of each dimension to a multiple of 32. LoMa uses the LoMa fine-tuning protocol with the same tensor interpolation but floors dimensions to a multiple of 14 for its DINOv2-L/14 descriptor. LoMa uses normalized[-1,1] keypoints and a default mutual-match threshold of 0.10; its feature cache records processed and original image sizes.
Standard and Vismatch feature-cache fingerprints also include the resolved dataset root, metadata file, and `dataset.image_variant`; normal and pre-masked features must never share a cache identity. Mask payload/content is not yet hashed and remains future work.
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
Matching displays a pair-counted tqdm progress bar with percentage, throughput, and ETA; progress advances only after successful batches, including after OOM retries.
and `extract_batch_size: 8`; `batch_mode: serial` remains the diagnostic/reference
workflow for parity checks. Extraction buckets images by matcher-native spatial shape.
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
  index as the tie-breaker. This rule is shared by evaluation, shortlisting, Jaguar,
  and classifier probes. Visualization ranking is now migrated: the run-local
  `visualizations/index.csv` uses `stable_rank_1d`, so it resolves ties identically to the
  prediction grid it annotates and to the metrics. No ranking path may use
  `argsort()[::-1]`, which reverses a stable ascending sort and orders ties backwards.
- Primary `mAP` includes every query; a query with no relevant gallery identity contributes
  AP=0. `mAP_eligible` is the legacy eligible-query-only diagnostic, and coverage fields
  report how many queries had a gallery match.
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
  read it together with `recall_at_k` and `candidate_recall_at_k`.
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
  also become `-inf`. It reports candidate hit/recall and scored/unscored pair counts.
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
  Existing nested legacy runs without manifests are not yet discovered when searching
  from the repository-level `results/` root.
- Finetune resume fails closed when the checkpoint leaves no epochs to run
  (`start_epoch >= train.epochs`). A zero-epoch run would still write final checkpoints and
  a completed manifest, hiding an unraised `train.epochs`; the guard runs before training
  setup so nothing is written.
- Finetune reports select and reload the best model-only checkpoint for primary metrics;
  final-epoch metrics remain nested as `final_epoch_metrics`. The current test/validation
  split remains the selection split and is a documented limitation.
- Accumulation divides each raw loss by the actual microbatch count in its group, including
  a partial final group; optimizer-step boundaries and scheduler behavior are unchanged.
- WildFusion calibration excludes same-image diagonal pairs by default and warns on the
  database-derived fallback. The configured split currently produces one calibration
  dataset used on both sides, so a truly disjoint calibration protocol remains future work.
  `official_same_set: true` enables exact all-pairs compatibility calibration for parity.

These validity changes are forward-only. Historical generated artifacts, aggregate CSVs,
and old caches are not rewritten automatically; rerun affected experiments before using


## Immutable parallel probe submissions

The selected dataset launcher creates a submission directory under `logs/parallel_run/submissions/<submission_id>/` containing the copied Hydra config, task table, and JSON manifest. The manifest is passed to Slurm with `--export=ALL,PROBE_PARALLEL_MANIFEST=...`; array tasks must read it rather than rereading `conf/probe.yaml`, shell checkpoint variables, or mutable dataset settings.

Dataset profiles explicitly pair dataset/animal settings with expected custom-checkpoint owners. Submission-time SHA-256 hashes and owner declarations are validated before model loading or cache writing. A missing, changed, or mismatched checkpoint/config fails closed and cancels only the current array element. The active benchmark grid and `probe.sh` contract remain unchanged. Use the selected launcher with `--list-tasks` or `--dry-run` for inspection, and never alter submitted manifests, copied configs, or checkpoint inputs.
Exactly one `DATASET_PROFILES` entry must be active. The wildlife launcher includes
templates for NyalaData, WhaleSharkID, BelugaID, ZindiTurtleRecall, ATRW, Giraffes,
LeopardID2022, HyenaID2022, GiraffeZebraID, CowDataset, StripeSpotter, and
SeaStarReID2023, plus a commented SalamanderID2025 profile; the current active entry
is the uncommented row in the file. The
added profiles use official pre-masked metadata and profile-specific checkpoint
layout/epoch fields; the four newest profiles use `legacy/epoch_299/model.safetensors`
for both LoMa and RDD. The launcher derives default LoMa and RDD checkpoint
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

LoMa descriptor exports may omit standard BatchNorm running-stat buffers. Descriptor
loading preserves those non-learned buffers from the active model defaults while
remaining strict for descriptor parameters, unexpected tensors, and tensor shapes.
