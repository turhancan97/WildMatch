# Changelog

All notable repository changes are recorded here. The current development line is
kept under Unreleased until a project version is assigned.

## [Unreleased] - 2026-08-13

### Changed

- 2026-10-03: video explainer started: narration script and shot list in `video/explainer/script.md`
  (Manim + Kokoro); the user approved the script and the separate `wm-video` conda environment
  (recipe in `video/explainer/ENVIRONMENT.md`); both tools smoke-tested.
  First full render the same day: `explainer.py` (seven Manim shots on the demo exports),
  `candidates.py` (real top-5 lists for shot 6), `build.py` (narration mux, captions,
  transcript); shot 6 wording corrected to the measured candidate list.
  Second render after the user's review: lynx 041 daylight pair (CPU-matched, real lists
  2/5 -> 5/5), ten different lynxes, lynx 029 pools, separated margin labels, dataset photos
  and a collage ending; selection sheets under `reports/project_page/explainer/`.
  Third render: Computer Modern typeface (matplotlib's cmr10, registered with manimpango),
  burned-in subtitles in a bottom band, database rows spaced from their tags.
  Fourth render: captions morph one persistent mobject in place (Cairo kept replaced
  submobjects on screen, so new captions were drawn over old ones).
  Approved as final. Home-page embed added: click-to-load facade with a poster frame, hosted
  player by video ID or local MP4 fallback for the draft, transcript from the committed
  timings, and a test pinning the two spoken numbers to the page data.
- 2026-10-03: project-page review pass (all suggestions approved by the user): seven nav tabs
  (Results section with Qualitative and Training cost; Paper & Code merged), Results reordered
  with at-a-glance tiles and a "Why it works" group, match viewer removed in favour of the
  before/after demo, home pointer retargeted, web-sized figure copies, licensing table updated,
  Open Graph/Twitter preview tags with a social card, Acknowledgements placeholder, and a
  publication checklist in AGENTS.md.
- 2026-10-02: project-page Datasets section with a "Data challenges" subpage (user request):
  `scripts/export_data_challenges.py` ranks audit candidates into raw-photo review sheets and
  exports the user-approved examples (48 photos in seven categories, noise vs inherent
  difficulties) and the audit summary; `data-challenges.js` renders the galleries and the
  per-dataset flag table. `docs/datasets.md` moved to `docs/datasets/index.md`. Tests in
  `tests/test_export_data_challenges.py`.
- 2026-10-02: project-page "Rare and common individuals" view (user idea). `scripts/export_frequency_bins.py`
  bins the paper's k=250 queries by gallery images per individual and reports Top-1/Top-5 of both LoMa
  matchers, the shortlist ceiling and bootstrap intervals on the gain; `frequency-bins.js` draws it on
  the Results page. Finding: in Top-5 no bin loses and singletons often gain most; Leopard's and
  Turtle's Top-1 losses sit in the 2-29 bins. Tests in `tests/test_export_frequency_bins.py`.
- 2026-10-02: project-page candidate-budget trade-off (user idea). `scripts/export_budget_tradeoff.py`
  collects, per dataset and measured k, the shortlist share, Top-5 of both LoMa matchers and the
  Vismatch matching time (minutes per 1,000 queries, ms per pair) with the GPU from Slurm
  accounting; `budget-tradeoff.js` adds a k slider with readout tiles and a two-panel chart to
  the Results page ("The cost of k"). Tests in `tests/test_export_budget_tradeoff.py`.
- 2026-10-02: project-page score-separation view (user request). `scripts/export_score_separation.py`
  builds same- vs different-individual score histograms and per-query margins for the eight
  paper datasets from the paper's k=250 default and fine-tuned LoMa runs' `scores.npz`,
  verifying the identity order by reproducing each run's Top-1. Shown as the "Score
  separation" section of the Results page with `score-separation-demo.js` (two Plotly panels,
  dataset and view selectors, statistics table); first drafted as a Demo page, moved to Results
  at the user's request the same day. AUROC improves on all eight
  datasets; the positive-margin share falls slightly on Leopard and Turtle, which the page states.
  Tests in `tests/test_export_score_separation.py`.
- 2026-10-02: the project page's Demo page was split into a `docs/demo/` section (user
  request, the single page had become cluttered): a hub `demo/index.md` with one card per
  demo (thumbnails from `scripts/build_demo_cards.py`, composed from the committed demo
  exports) and five pages in narrative order (before/after, rank changes, mined pairs,
  background masking, synthetic matching), each with a lead line and a back button. No
  JavaScript changed; each page now loads only its own widget. Tests in
  `tests/test_build_demo_cards.py`; `mkdocs build --strict` passes.
- 2026-10-02: project page scaffold (MkDocs Material) on the local `project-page`
  branch: `mkdocs.yml`, `requirements-docs.txt` (mkdocs-material 9.7.7), `docs/` pages
  (home with abstract and authors, method, datasets, results, qualitative, training cost,
  reproduce stubs, code, paper), `overrides/main.html` review-period banner, static copies
  of the current figures, mount points for four interactive components, and
  `tests/test_project_page.py`. Text follows the paper brief relayed by the paper-writing
  session (stored under the gitignored `reports/project_page/`). Decisions: MkDocs Material
  over plain HTML or Astro; page conventions follow the paper (k=250, Top-5 and balanced
  Top-1, weighted probes only, eight datasets, unseen-identity protocol instead of the
  CzechLynx open split); the branch stays unpushed and undeployed until the ECIR 2027
  notification because the submission is double-blind and the repository is public.
  Verified for the paper: SalamanderID2025 SAM 3 prompt "Salamander" (all rows of
  `masks.csv`) and all training-cost jobs on `rtx4090_batch` (sacct).
  The paper's LaTeX repository was then cloned read-only as a sibling checkout
  (`ECIR-Animal-ReID-Paper`, `dbe837b`, pulled to `9e78102` the same day once the implementation-details and GPU-hour edits were pushed) with the user's approval, and the page's masking
  statement was marked draft because the manuscript sentence is unsettled.
  Brand palette applied (user request): custom Material primary/accent mapped to
  #3a7eab/#cf4832 with #d1d3d4 rules, brand greys for text, icon cards on the home page
  from the paper's icon set, interim logo/favicon from the match icon (the logo folder
  was gitignored in the paper repository); `test_project_page` checks palette and logo.
  The user then copied the logo package to the gitignored `reports/project_page/logo/`;
  web-sized copies (hero wordmark, header tile, favicon set) replaced the interim icons.
- 2026-10-02: project-page data export (step 2). `scripts/export_project_page_data.py`
  reads the paper repository's `results/` snapshot (user instruction) and writes
  `docs/data/{results,curves,adapt,training_cost,manifest}.json` under the paper's
  conventions, plus brand-palette SVG page figures (light and dark) under
  `docs/assets/figures/page/`: k-sweep small multiples (LoMa Top-5 and balanced Top-1,
  RDD Top-5), a k=250 default-to-fine-tuned dumbbell, the adaptation cost-against-gain
  scatter, the unseen protocol and the training-cost curves. Exported Table 2 and Table 3
  values match the manuscript exactly. Pages embed the new figures; the copied
  training-cost PNGs were removed. Tests in `tests/test_export_project_page_data.py`.
- 2026-10-02: first `mkdocs build --strict` of the project page passed; `mkdocs==1.6.1` is
  now pinned next to `mkdocs-material==9.7.7` so the MkDocs 2.0 rewrite (no plugins or
  theme overrides) cannot be installed by accident.
- 2026-10-02: project-page interactive components (step 3): `docs/assets/js/` ES modules
  (accuracy explorer, sortable results table with CSV download, training-cost plot, match
  viewer) over the exported JSON, Plotly from the pinned CDN build, theme-aware; styles in
  `extra.css`. `scripts/plot_match_examples.py render` gained `--web-export DIR`, writing
  web-sized raw photos and every correspondence for the viewer (GPU run pending).
- 2026-10-02: project-page cross-check (step 4). Every transcribed table and prose claim
  was verified against the paper clone (`9e78102`) and the exported data; all matched.
  `tests/test_project_page_numbers.py` parses the Markdown tables and pins them to
  `docs/data` and the paper's `tab_datasets.tex`. Pages gained the paper's explanations
  for the descriptor result, the RDD narrowing and the qualitative figure. The user
  exported the match-viewer assets (`docs/assets/match/`).
- 2026-10-02: project-page Reproduce section (step 5) written from the three repositories'
  guides: pair mining, matcher fine-tuning (recipe, wrappers, checkpoint layout, ablation
  variants), evaluation (data contract, probe commands, launchers, unseen protocol,
  metrics, exporters) and a shared-settings overview, with placeholder paths.
- 2026-10-02: home page gained "Limitations and outlook" from the paper's newly pushed
  Conclusion (`226037a`); the earlier draft limitation bullets are superseded.
- 2026-10-02: the project page no longer names the venue or the submission status
  (user decision): status pill, Paper page, banner and configuration comments reworded;
  the leak test now forbids venue names and "under review"/"submitted to" in site sources.
- 2026-10-02: header logo is now the background-free two-colour mark
  (`wildmatch-logo-256.png`, derived from `wildmatch-logo.png`), and the hero places the
  full-name logo beside the title instead of in a full-height left column (user request).
- 2026-10-02: header bar recoloured, first to copper `#ab673a` (rejected), then to white with
  dark grey text (dark scheme: `#2b3036`) so the two-colour logo mark reads as designed; the
  brand blue remains for links, hero title, active tabs and chart series.
- 2026-10-02: objective wording settled (paper session, confirmed from the training code):
  the page describes a triplet margin objective as a contrastive loss and the two "Draft"
  notes about triplet-versus-contrastive wording were removed.
- 2026-10-02: synthetic keypoint-match demo for the home page (user request): 
  `scripts/export_synthetic_demo.py` matches the paper's CC BY 4.0 CzechLynx synthetic
  teaser renders with the fine-tuned LoMa matcher (paper run configuration, masked input
  reproduced from the teaser crops) and writes ranked candidates with every correspondence
  and confidence; `synthetic-demo.js` shows the ranked strip, confidence-weighted lines,
  a strongest-N slider and hover confidences. Decisions: fine-tuned only, query vs all six,
  home page, labelled synthetic. The script then switched to the local CzechLynx synthetic
  renders and their dataset masks (user pointer) and was run on the login node's V100 with
  the user's permission; the export (seven renders, ranked matches) is committed.
- 2026-10-02: synthetic demo gallery extended to ten individuals (user request). A first
  pick by mask share and size alone (lynx 265, 226, 101, 206) had unnatural coats and
  frontal poses and drew hundreds of confident matches (scores 0.29-0.62); the user chose
  selection by appearance instead, so lynx 27, 174, 289 and 45 were picked before matching
  with thresholds calibrated on the paper's renders (side view, natural colours) and their
  scores kept as they came (0.001-0.003). Export rerun on the local GPU and recommitted.
- 2026-10-02: the synthetic demo became a ten-query demo (user clarification): ten individuals
  with two renders each, render A as query and the ten render-B images as the shared
  gallery, a query strip to switch between them, `correct_rank` per query; gallery renders
  chosen by the same appearance rules from an A/B contact sheet. Export rerun on the GPU:
  7 of 10 queries correct at rank 1, three near-ties with another individual, stated on
  the page.
- 2026-10-02: the synthetic demo moved from the home page to its own Demo page (user request);
  the home page keeps a hero button and a one-line pointer.
- 2026-10-02: demo and match-viewer canvases now fit the page width (height derived from the
  available width, redrawn on resize) instead of overflowing sideways; fixed the demo's
  recursive image callback that left the canvas blank.
- 2026-10-02: the demo's query and gallery selectors became two compact filmstrips (inline
  label, small thumbnails, rank and score caption, red inner frame on the query's own
  individual) to shorten the page (user request).
- 2026-10-02: background-masking demo (SAM 3, piece 1) scaffolded on the synthetic renders:
  `scripts/export_masking_demo.py`, `masking-demo.js` (divider, outlines, readouts, IoU with
  the dataset mask), Demo page section and tests. The SAM 3 run on the twenty renders is
  pending (A100/H100, `lynx-app` env). Salamander photographs are blocked by the Kaggle
  competition rules (no redistribution) until the sponsor grants permission.
- 2026-10-02: SAM 3 run for the masking demo done on the H100 node (prompt "animal", as in
  the paper): twenty renders, one instance each, scores 0.945-0.969, IoU with the dataset
  masks 0.926-0.982 (mean 0.960; snow shadows explain the lowest). Export committed.
- 2026-10-02: masking demo gained a real-photo group (user request after the Salamander team
  permitted display): the sixteen match-figure photos of all eight datasets, SAM 3 run with
  "animal" (fallback "sea star" for the two sea stars, which "animal" misses), compared
  with the dataset's own mask where one ships; group tabs in the viewer; Demo page text.
- 2026-10-02: masking demo compares against a reference mask only for the synthetic subset and
  CzechLynx, the datasets that ship segmentation masks (user correction); WildlifeReID-10k and
  Salamander photos show the SAM 3 mask alone.
- 2026-10-02: `scripts/analyze_background_matches.py` tests whether the fine-tuned LoMa matcher
  favours the animal when photos keep their background (Nyala, nine pairs, H100). It does not:
  38 % of its matches land on the animal versus 40 % for the default matcher, and on raw inputs
  it either matches the background heavily or collapses; masks remain required at inference.
- 2026-10-02: the same analysis on a CzechLynx subset with exact RLE masks (13 pairs): default
  49 % of matches on the animal, fine-tuned 39 %; two different-lynx pairs in similar scenes
  score 0.52 and 0.62 through background matches. The script now supports RLE-masked layouts.
- 2026-10-02: `--joint-checkpoint` adds the joint descriptor + matcher LoMa model to the
  background analysis; on the CzechLynx subset it is as background-prone as the matcher-only
  model (36 % of matches on the animal) and still scores some different-lynx pairs high.
- 2026-10-02: before/after fine-tuning demo (user request): `scripts/export_before_after_demo.py`
  matches the eight match-figure pairs with the default and the fine-tuned LoMa matcher on
  masked inputs and `before-after-demo.js` toggles between them on the original photos, with
  scores, match counts, hover confidences and a pair selector; first section of the Demo page.
- 2026-10-02: mined-pairs browser (user request): `scripts/export_mined_pairs_demo.py` joins
  the CzechLynx LoMa mining index with the per-collection reports (pretrained scores), picks
  ten anchors by appearance, re-matches the 100 pairs for correspondences, and
  `mined-pairs-demo.js` shows anchor, five positives and five hard negatives with scores.
- 2026-10-02: rank-change explorer (user request): `scripts/export_rank_change_demo.py` ranks
  all CzechLynx test queries under cosine (probe feature cache), default and fine-tuned LoMa
  (run `scores.npz`), reproducing the paper's Top-1 counts, and exports whole-split category
  counts plus a random sample per category; `rank-change-demo.js` shows filters, query strip
  and three top-5 columns with the true individual highlighted.
- 2026-10-02: paper clone at `2773089` (intro names the triplet margin objective; the
  match figure switched to the text-free render and its caption to "of the 350 to 420
  matches found for each pair, lines show the 10 most confident"); the page's match figure
  and captions follow.
  Hero subtitle flows full width under the logo row and the author line is smaller.
- 2026-10-02: `test_czechlynx_split_is_explicit_and_open_profile_can_be_enabled` failed once
  the editable CzechLynx launcher had only the LoMa `joint-fine-tuned` row active, because
  enabling the open profile in its copy hit the closed-only joint guard. The test now
  comments out joint rows in its temporary copy and re-enables the cosine row so profile
  multiplication always has a split-agnostic task. Launcher behaviour is unchanged.
- `plot_data_quality_examples.py` now shows raw source photos only, in one row:
  (a) washed-out frames, (b) no animal, (c) corrupted frame, (d) blurred frame. Masked
  model inputs and the mask-on-wrong-object panel were removed because background
  removal could be read as our error (user decision, 2026-09-30).
  The per-image notes under each panel were then dropped, since the group labels
  already describe each column.
  Added an (e) "No local details" group (ZindiTurtleRecall row 11453; WhaleSharkID
  row 614, which replaced row 5104), with group labels wrapping to two lines when
  wider than their columns. A Nyala example (rows 1071, then 1355) was tried and
  removed to keep the figure compact.
  Final layout for the paper: (a) Overexposure, (b) Insufficient detail (moved to second
  position), (c) Empty frame, (d) Corruption, (e) Blur, set in Times (STIX fallback,
  embedded) to match the CVPR text. The empty-frame panel trims its source file's
  black padding and fills its box (`fill=True`), and panels are now resized to the
  panel size in both directions so small sources are upscaled rather than letterboxed.

### Added

- `plot_match_examples.py render --no-tags --output-stem`: text-free match figure
  (`match_examples_notext`), titles only.
- Recorded CzechLynx closed fine-tuning GPU-hours (matcher, descriptor and joint, LoMa and
  RDD) in AGENTS.md for the paper.
- `plot_training_cost.py --x-scale log` for a log training-cost axis
  (`training_cost_k250_balanced_top5_logx`).
- Final k=250 training-cost figures `training_cost_k250_balanced_top5` and
  `training_cost_k250_weighted` from the finished weighted probes (array 522223) and the
  seven k=250 LoMa checkpoints.
- `plot_training_cost.py` gained `--allow-incomplete` (still-running probes up to their
  last logged epoch), `--metrics` (panel selection and order) and `--output-stem`. Used for
  the k=250 figure `training_cost_k250_balanced_top5` (balanced Top-1 left, Top-5 right)
  with the rerun probes of array 522223 and the six k=250 LoMa checkpoint evaluations.
- Linear- and efficient-probe per-epoch log lines now end with `val_balanced_top1=`
  (already computed each epoch, previously only sent to W&B). `plot_training_cost.py`
  draws probe balanced Top-1 curves when the log has the field, gained
  `--probe-job-dir`, and reads job durations from the task metadata.
- Revived the training-cost ablation (team request, 2026-09-30) with a minimum-effort
  design: CzechLynx closed, fine-tuned LoMa vs the six classifier probes, test Top-1
  against training GPU-hours on RTX 4090, with no epoch selection. Probe curves reuse the
  existing per-epoch logs; `scripts/eval_loma_epoch_curve.sh` evaluates the saved LoMa
  checkpoints (epochs 0-250) with the epoch-299 run's frozen config, writing to
  `experiments/compute-efficiency/` so the paper tables cannot pick them up.
  `scripts/plot_training_cost.py` (tests in `tests/test_plot_training_cost.py`) draws Top-1
  and balanced Top-1 against training GPU-hours from those runs and the existing logs,
  counting pure training time only (LoMa 5.10 GPU-h; probes 3.8/8.1/10.0 GPU-h).
  It then switched to the weighted-loss probes only, with Top-1, Top-5 and balanced Top-1
  panels (user decision); probes have per-epoch Top-1/Top-5 curves and a final balanced
  Top-1 marker. `--candidate-k 100|250` renders variants from existing runs. `--cost job` renders the conservative
  whole-Slurm-job accounting (LoMa 8.9 vs full FT 11.9 GPU-h).
- `scripts/plot_match_examples.py`: qualitative figure with one correct fine-tuned
  LoMa (k=50) top-1 match per paper dataset on raw photos (4 x 2 grid,
  `reports/figures/match_examples.{pdf,png}` plus JSON sidecar), with a `candidates`
  step writing per-dataset shortlists and contact sheets for selection by eye. The
  checkpoint is resolved by the manifest's SHA-256, because directories were renamed and
  Nyala's recorded `model.safetensors` was overwritten on 2026-09-09. Features come from
  the probe's own Vismatch cache. Recomputed LoMa scores match `scores.npz` only to about
  1e-2 because LoMa runs under bfloat16 autocast, so rank 1 is always read from
  `scores.npz`. Tests in `tests/test_plot_match_examples.py`. `EXAMPLES` holds the
  user's eight picks, and `render` crops each photo to a shared 4:3 box around its
  matched keypoints (`--aspect`) so the grid has no uneven whitespace.
  Paper styling: Times fonts, "(a)"-style titles, Query/Top-1 match/match-count corner
  tags, trimmed source padding, 30 spatially spread matches in outlined cyan, and a
  300 dpi embed (3.6 MB PDF). Mask-based background dimming is opt-in (`--dim`)
  because it exposed mask outlines that would read as masking errors. Then, at the
  user's request, the default became 10 matches and a light, widely feathered dim
  (`--dim 0.2`), which keeps the outlines nearly invisible.
  To save page space the figure became three panels per row with a centred last row
  of two (2.97 in tall instead of 5.8 in), in alphabetical dataset order, with stronger
  per-panel dimming (`Example.dim`) where the mask follows the animal cleanly.
  Side tags now appear on the first panel only, dots never sit under a tag, lines have
  a thicker outline, Hyena is undimmed, and the figure spans the full text width.
  `candidates` gained `--min-foreground`, `--min-saturation`, `--min-warm` and `--tag`,
  used to search for a daylight CzechLynx replacement pair.
  The user chose daylight pair 12 (`lynx_041`, 2022 vs 2020) for the Czech Lynx panel. Its background
  dimming was raised to 0.35 at the user's request; the mask shows no outline.
  Corner tags were enlarged to 6.5 pt and panel titles to 9 pt (`--tag-size`, `--title-size`).
- Extended the image-quality audit to SalamanderID2025. Pre-masked metadata that ships
  an RLE `mask` column now supplies the foreground instead of the brightness threshold,
  and the manifest records each dataset's foreground source. Confirmed Salamander
  problems are finger occlusion that fragments the mask (62 multi-instance images, no
  measured accuracy cost) and blurred flash close-ups. `plot_data_quality_examples.py`
  reads Salamander raw images from `original_path`.

- Joint descriptor + matcher fine-tuning support (2026-09-30), probed and reported as its
  own family. The Vismatch checkpoint loader recognises `rdd_train_component` /
  `loma_train_component` = `joint` as a complete model and rejects partial component
  modes for it; `probe-parallel-czechlynx.sh` gains commented closed-split
  `joint-fine-tuned` rows with `CZECHLYNX_CLOSED_JOINT_{RDD,LOMA}_CHECKPOINT`, and the
  submission manifest enforces `joint-fine-tuned` <=> `checkpoint_components=full` (a
  `custom` label with `full` is now rejected). Paper tables write separate
  `<animal>_<split>_joint_{rdd,loma}_*` files and keep `full-fine-tuned` rows out of the
  main/ablation tables; figures add `joint_{rdd,loma}_*` and no longer fold
  `full-fine-tuned` into the matcher series. No existing run used `full`, so published
  tables are unchanged. The training side lives in lynx-finetuning (`joint` presets for
  RDD and LoMa, LoMa `--keep_every` and resume).

- SalamanderID2025 support (2026-09-29). `scripts/segment_with_sam3.py` removes
  backgrounds with text-prompted SAM3 (instances merged, fallback threshold and prompt,
  reviewed per-image threshold overrides, rows never dropped) and builds pre-masked
  metadata with an optional mapped split column; `tests/test_segment_with_sam3.py`
  covers merging, RLE, parsing and fail-closed metadata building. `PAPER_PROFILES` gains
  a `salamander` entry, so class-balance and image-quality exports include it (existing
  CSVs are byte-identical). `probe-parallel-wildlife.sh` gains a commented Salamander
  profile (database/query split values, SAM3 pre-masked metadata, `legacy-loma-mined` /
  `legacy-rdd-mined` checkpoints); `test_probe_parallel` activates it in a temporary copy
  and checks task generation, checkpoint ownership and the Hydra overrides. The sibling
  repositories gain `configs/wildlife/SalamanderID2025.json` (rdd-parallel-benchmark) and
  a WILDLIFE.md section (lynx-finetuning). A cosine probe with split safety checks enabled
  found no path or content leakage; its baseline is Top-1 3.7%.

- Paper-table discovery now treats the backbone/model as part of a run's identity
  and adds a `Backbone` column to CSV and LaTeX exports. Cosine baselines for
  MegaDescriptor-T/L, DINOv2/L, and DINOv3/L therefore remain separate instead of
  being collapsed to the newest run; manifests without a model field are labeled
  `unknown`.

- Added `dinov2-l` (`facebook/dinov2-with-registers-large`, ViT-L/14) and `dinov3-l`
  (`facebook/dinov3-vitl16-pretrain-lvd1689m`, ViT-L/16) backbones: 1024-d CLS
  embeddings at 224 px, verified end to end on the downloaded checkpoints.

- Documented the DINOv2/DINOv3 checkpoints and token policy in AGENTS.md after
  verifying the token layout on the real checkpoints: CLS for cosine, WildFusion,
  Vismatch Stage A, and linear probe; patch tokens with attention pooling for the
  efficient probe; GAP only in the efficient-probe `embedding_retrieval` diagnostic.
  No code changes.

- Restricted paper-table ablation budgets for `unseen_eval_split` to
  `10, 50, 100, 160` and reject invalid main-table candidate budgets for that
  split. Other split protocols retain the standard six-budget table grid.

- Added `scripts/audit_image_quality.py`, which flags candidate low-quality images
  (empty/tiny foreground, fragmented masks, over/underexposure, low contrast, and a
  dataset-relative blur ranking) on the exact model input and writes per-image CSVs,
  per-flag contact sheets, and a summary joined with per-query top-1 rates from
  completed runs. Moved the paper dataset profiles into
  `reid/reporting/paper_datasets.py`; `export_class_balance.py` output is byte-identical.
  First full audit: washed-out CzechLynx frames and non-animal masks, and SeaStarReID2023
  provider-mask failures, were confirmed by eye; dark Hyena images were rejected as noise.
- Added `scripts/plot_data_quality_examples.py`, a full-width paper figure of six
  confirmed examples (washed-out, wrong-object mask, corrupted frame, blurred frame)
  showing each raw photo above the exact model input.

- Added `scripts/export_class_balance.py`, which writes per-identity database/query
  image counts for each paper dataset/split to `experiments/class-balance/`, plus a
  cross-dataset `summary.csv` and a source-hash `manifest.json`. It reads the same
  metadata, identity columns, and split values as the parallel launchers; the paper
  dataset table reports CzechLynx closed and open splits only. No probe, launcher, or
  reporting behavior changes.

### Fixed

- `audit_image_quality.py --dataset <subset>` no longer overwrites `summary.csv` and
  `manifest.json` with only that subset: the summary is rebuilt from every per-dataset
  CSV and manifest sources are merged. The nine rows lost to one such run on 2026-09-30
  were rebuilt from their CSVs; their original run lists are marked unknown.

- Fixed DINO partial fine-tuning rules. The DINOv3 efficient-probe rule used DINOv2
  module names and matched no parameters, silently producing a frozen probe; linear-probe
  DINO partial fell back to the Swin default and tuned only LayerNorms. Every DINO type
  now has explicit last-block-plus-final-norm rules in both probes, and partial mode fails
  closed when its patterns match nothing. No DINO runs existed, so no results changed.
  Swin rules are unchanged.

- `sort_run_rows` now produces a total order: numeric cells sort in the requested
  direction, then text cells, then empty and `NaN` cells last in their original order.
  `scripts/summarize_runs.py --sort-by top_1` no longer raises `TypeError` on failed-run
  rows, and sorting by the gated `mAP` column no longer returns a silently unsorted list.
- Updated stale tests: the batched-extraction smoke test's fake backend now carries the
  feature-cache timing counters the real backend initializes, and launcher tests validate
  whichever classifier-probe rows and CzechLynx profiles are active instead of requiring
  specific rows to be uncommented. Launcher behavior is unchanged.

### Decided

- 2026-09-30: joint descriptor + matcher fine-tuning for RDD-LightGlue and LoMa on
  CzechLynx closed: detectors frozen, pretrained initialization, one learning rate,
  LoMa-mined pairs, legacy protocol, epoch 299, effective batch 32, reported separately
  from matcher-only and descriptor-only fine-tuning.

- 2026-09-29: SalamanderID2025 runs through the existing wildlife probe launcher instead
  of a copied launcher, relaxing the rule that it serves only WildlifeReID-10k profiles
  (CzechLynx profiles stay excluded). Masked images only, legacy protocol, full wildlife
  method grid and k grid, and inclusion in the main cross-dataset table. Each matcher is
  fine-tuned on its own mined pairs (LoMa on LoMa-mined, RDD on RDD-mined), an explicit
  exception to the LoMa-mined recipe used for the other datasets.

- 2026-09-29: RDD-LightGlue and LoMa matcher fine-tuning (sibling `lynx-finetuning`
  repository) now share one training recipe: relaxed pair score over every triplet,
  AdamW, effective batch 32, LoMa-mined pairs for every dataset, fixed epoch 299. A
  colleague noted that the relaxed objective described in the paper applied to LoMa only;
  the difference had no documented reason (the relaxed score arrived with the first LoMa
  implementation, RDD kept its July objective). Every RDD run feeding the paper is
  retrained into the canonical checkpoint directories after the old ones are archived as
  `<name>__filtered-archive`, then re-evaluated. Fixed in the same work in
  `lynx-finetuning`: padded keypoints entered LightGlue's assignment softmax and could be
  matched; wildlife RDD training silently used RDD-mined pairs where no LoMa index existed;
  and a 2026-09-27 regression routed every cached pseudo-accuracy batch into the live-image
  path, crashing cached matcher runs at their first evaluation. Five wildlife RDD
  checkpoints referenced by existing probe manifests (HyenaID2022, LeopardID2022, ATRW,
  CowDataset, StripeSpotter) were found missing on disk; the retrain supersedes them.
  AGENTS.md now names the actual `*-finetuned-loma-mined-legacy` CzechLynx checkpoint
  directories instead of the stale `loma-b-finetuned-legacy` / `rdd-finetuned-legacy`.
  No code in this repository changed; the CzechLynx closed RDD descriptor launcher default
  moves from `epoch_175` to `epoch_299` once the retrained checkpoint exists.

- Split safety checks stay disabled in the shipped `conf/probe.yaml`
  (`safety_checks.enabled: false`) by explicit user decision. Parallel launchers
  inherit this setting. `test_hydra_config` now pins the `false` default, and AGENTS.md
  records that leakage must be verified separately before publishing a new split.


### Added

- Restricted paper-figure candidate budgets for `unseen_eval_split` to the
  valid gallery sizes `10, 50, 100, 160`; other split protocols retain the
  existing plotting grid. Explicit unseen-split budget requests are filtered
  to the same valid set.

- Added the standalone `scripts/build_unseen_eval_metadata.py` utility for
  evaluation-only unseen-identity gallery/query splits. It uses encounter and
  temporal grouping, assigns the earliest group to the gallery, rejects random
  image-level fallback, validates paths and cross-side SHA-256 duplicate content,
  and writes an auditable generated CSV plus manifest. Existing probe, launcher,
  cache, and historical artifacts are unchanged.

- Added an opt-in `czechlynx_unseen_eval` profile to
  `probe-parallel-czechlynx.sh`. It consumes the standalone generated metadata
  through `unseen_eval_split=database/query`, validates the file before task
  construction, and keeps unseen-evaluation logs, manifests, experiments, and
  caches separate from the existing closed/open split profiles. It remains
  commented by default and now points to the repository-local generated
  `metadata_unseen_eval.csv` unless overridden.

- Added publication-oriented plot styling to `scripts/plot_paper_figures.py`.
  The default paper style uses a color-blind-safe palette, redundant line and
  marker encodings, a logarithmic candidate-budget axis, shared metric y-limits
  across panels, publication typography, and high-resolution PNG plus
  embedded-font-friendly PDF/SVG output. Presentation and diagnostic styles,
  categorical-axis overrides, independent y-limits, and optional endpoint labels
  are available from the command line. Data selection, split separation,
  descriptor separation, and historical artifacts are unchanged.

- Added repeatable `--exclude-method` filtering to paper figures. The supported
  families are `wildfusion`, `rdd`, and `loma`; each removes both default and
  fine-tuned matcher series, and the corresponding descriptor-family figures
  are suppressed consistently. With no option, plotting behavior is unchanged.

- Updated paper-table rendering so linear and efficient classifier-probe rows
  visibly include their loss policy in the checkpoint cell, for example
  `frozen (weighted)` and `frozen (unweighted)`. The existing audit
  `class_weighting` column and run-selection identity are unchanged.

- Added descriptor-only Vismatch checkpoint integration for RDD-LightGlue and
  LoMa. `checkpoint_components=descriptor_only` validates descriptor tensor
  schemas and optional training protocol metadata, applies only descriptor
  tensors, retains default detector/matcher components, and records hashes,
  applied/ignored prefixes, and provenance in cache identities and manifests.
  Descriptor variants are opt-in in both parallel launchers and require
  profile-specific checkpoint ownership/path settings, including explicit
  cross-species evaluation metadata.

- Added separate descriptor reporting. Completed descriptor runs are omitted
  from matcher tables/plots and produce dedicated RDD and LoMa paper-table
  files and accuracy-versus-budget figures with default matcher and context
  baselines. Historical matcher artifacts and generated outputs are unchanged.
  Parallel manifests support explicit cross-species descriptor profiles by
  separating the evaluation animal from the checkpoint owner and validating the
  owner against an owner-identifiable checkpoint path.
  LoMa descriptor loading preserves standard BatchNorm running-stat buffers when
  a descriptor export omits those non-learned buffers, while remaining strict for
  all trainable tensor names and shapes.

- Added identity-weighted cross-entropy to `efficient_probe`, matching the existing
  `linear_probe` policy. Efficient probes now default to training-only normalized
  inverse-frequency weights capped at `5.0`, retain `class_weighting=none` for
  paired comparisons, and keep validation loss and reported metrics unweighted.
  Both parallel launchers support weighted/unweighted efficient-probe rows for all
  three train modes; task provenance and W&B names include the weighting policy.

- Added open-world evaluation for `linear_probe` and `efficient_probe` across all
  datasets. The default `benchmark.classifier_evaluation.open_set_policy: open`
  preserves unseen query samples, gives them zero classification credit, skips them
  only for validation loss, and records image/identity coverage. `warn` retains
  seen-only compatibility fields alongside all-query diagnostics, while `closed`
  fails before model construction. Added the optional disabled-by-default
  `embedding_retrieval` diagnostic and enabled probe safety checks by default.

- Added balanced Top-1 accuracy to the default paper-figure metric set. Standard
  plotting now emits a `balanced_top_1_vs_k` figure alongside Top-1, Top-5, and
  Top-10; explicit `--metric` selections remain supported.

- Made paper figures split-aware. `scripts/plot_paper_figures.py` now renders
  separate figures for each discovered `split_protocol`, including distinct
  CzechLynx closed/open output filenames, and supports an optional repeated
  `--split-protocol` filter. Closed and open split results are no longer
  silently collapsed into one plot.

- Corrected the CzechLynx parallel launcher profiles to explicitly probe the
  no-background image variant. Both closed and open profiles now pass
  `dataset.no_background=true` and `dataset.image_variant=no_background`,
  matching the probe configuration and preventing background/no-background
  comparisons from being mixed.

- Paper-table discovery now keeps `linear_probe` training modes separate. The
  resolved `all`, `partial`, or `classifier` mode is included in run selection,
  preventing the newest mode from replacing the other linear-probe results. The
  paper-facing checkpoint column labels these modes as `full fine-tuned`,
  `partial fine-tuned`, and `frozen`; older artifacts without a readable mode are
  labeled `unknown`.

- Added readable default W&B names for probe and finetune runs. Names include
  dataset, animal, split, model/mode, method-specific settings, checkpoint variant,
  linear-probe weighting where applicable, and a short run token. Explicit
  `wandb.name` values remain unchanged and take precedence.

- Added identity-weighted cross-entropy for `linear_probe`. New runs default to
  training-only inverse-frequency weights, normalized to mean one and capped at
  5.0 after normalization without a second renormalization. Explicit
  `class_weighting=none` runs remain available for paired comparisons; query
  loss and reported metrics stay unweighted. Parallel launcher task identities
  now distinguish `weighted` and `unweighted` linear-probe variants.

- Efficient probes now use the same configurable identity-weighted training loss
  policy as linear probes. Their validation loss and reported metrics remain
  unweighted, and efficient weighted/unweighted modes are kept separate in
  launcher tasks, W&B names, manifests, and paper-table audit records.


### Fixed

- Restored safe handling of the legacy CzechLynx checkpoint environment
  variables under `set -u`, so active custom Vismatch variants cannot abort the
  launcher before Slurm submission. Profile-specific checkpoint paths remain
  authoritative.
- Classifier-mode linear and efficient probes now keep the frozen backbone in
evaluation mode while training only the probe objective, preventing dropout and
normalization-state changes in the extracted representation.

### Added

- Added split-aware CzechLynx parallel probing. The launcher now supports the
  `split-time_closed` and `split-time_open` profiles, with optional simultaneous
  submission, explicit per-split custom checkpoint paths, split-tagged task logs,
  and split provenance in manifests and task metadata.
- Added split-aware paper-table discovery and export. Closed and open CzechLynx
  protocols receive separate main/ablation LaTeX and CSV files, with an optional
  `--split-protocol` filter; combined tables are intentionally not produced.
- Configured the CzechLynx open-split profile with its validated epoch-299 LoMa
  and RDD checkpoint roots under `czechlynx-time-open`; the paths remain
  independently overridable per profile.

- Added explicit wildlife and CzechLynx parallel-launcher support for linear
  probing with `classifier`, `partial`, and `all` train modes. Active modes are
  passed as Hydra overrides and recorded in task logs/metadata; redundant
  candidate-budget repetitions are avoided for these non-shortlist probe variants.
- Removed the CzechLynx dataset profile from `probe-parallel-wildlife.sh` so
  CzechLynx parallel probing is available only through
  `probe-parallel-czechlynx.sh`.
- Added immutable submission manifests for parallel probe arrays, including copied Hydra configuration, explicit dataset profiles, checkpoint ownership, and SHA-256 provenance.
- Added the active `ZindiTurtleRecall` dataset profile to `probe-parallel-wildlife.sh`, using its no-background metadata and matching custom LoMa/RDD checkpoint ownership.
- Added `scripts/export_paper_tables.py` and dependency-light reporting helpers for per-animal CVPR LaTeX/CSV tables, fixed-budget comparisons, and candidate-budget ablations sourced directly from completed experiment artifacts.
- Wrapped generated paper tables in a `graphicx` `\\resizebox` so wide CVPR ablation tables fit the available column width.
- Added the opt-in `--detailed-comments` exporter flag; timestamp, run-ID, and manifest comments are now omitted by default.
- Paper tables now display the internal `custom` checkpoint variant as `fine-tuned` without changing run selection or provenance identities.
- Added standardized primary compute, matcher, feature, setup, calibration, cache, and total runtime fields for probe runs; paper tables now show primary compute and total runtime separately.
- Updated parallel launcher regression checks to honor the active candidate-budget list configured by the user.
- Added compact CVPR-style ablation LaTeX tables with method sections,
  default/fine-tuned row shading, same-budget delta arrows, and primary compute
  runtime. mAP, mAP@k, and total runtime remain in the companion CSV audit files.
- Changed the paper-table main comparison to the compact paired layout at
  `candidate_k=50`, including default-versus-fine-tuned gain arrows and the same
  primary-runtime presentation as the ablation tables.
- Added `scripts/plot_paper_figures.py` for per-animal, multi-panel accuracy-versus-
  candidate-budget figures in PNG/PDF formats, with consistent paper styling and
  explicit handling of missing or incomplete runs.

### Changed

- CzechLynx parallel tasks no longer use a single global animal/checkpoint
  selection. Active profile entries define their own split and checkpoint
  provenance, while the benchmark variants and candidate budgets remain shared.

- Matcher-only timing excludes Stage-A selection, feature extraction, setup, calibration, cache I/O, and visualization; historical runs without the new fields require reruns for matcher-speed comparisons.
- Clarified that `k` is the gallery-candidate budget for shortlist-based second-stage
  matching; larger values can improve coverage at higher compute cost, while `--`
  denotes full-gallery evaluation.

- Parallel tasks now receive their complete configuration through the submission manifest and emit explicit dataset/checkpoint overrides. The existing benchmark grid and `probe.sh` remain unchanged.
- Parallel launcher checkpoint defaults are now derived from the sole active profile's animal; zero or multiple active profiles fail fast before task generation.
- Added ready-to-activate WildlifeReID-10k profiles for ATRW, Giraffes,
  LeopardID2022, and HyenaID2022, including their official masked metadata and
  known fine-tuned checkpoint layouts.
- (2026-09-07) Added ready-to-activate WildlifeReID-10k profiles for
  GiraffeZebraID, CowDataset, StripeSpotter, and SeaStarReID2023. They use the
  established mdsplit no-background metadata protocol and profile-derived
  `legacy/epoch_299` LoMa/RDD checkpoint paths.
- Added commented dataset-profile templates for CzechLynx, NyalaData, WhaleSharkID, and BelugaID; users activate exactly one profile for a run.

### Fixed

- Prevented mutable post-submission configuration or checkpoint variables from creating mismatched animal/checkpoint runs. Invalid tasks fail before model loading/cache creation and cancel only their own array element.

### Known Issues

- Additional datasets must be enabled by adding a complete, validated `DATASET_PROFILES` entry and their custom checkpoint paths. Historical logs are not migrated.

### Added

- Added shortlist-safe retrieval metrics `mAP_at_k`, `rerank_mAP_at_k`, `recall_at_k`,
  `num_queries_with_relevant_in_top_k`, and `score_coverage`, computed identically for
  full-matrix and shortlist-constrained methods so matchers stay comparable.
- Added `benchmark.map_at_k` (default `100`) as the shared evaluation cutoff, and a
  startup validator that rejects any `top_k`/`map_at_k` larger than the Vismatch
  `candidate_k` shortlist.
- Added sparse score-matrix persistence (`scores.npz`, COO of finite entries) plus
  `save_score_matrix`/`load_score_matrix`, so run metrics can be recomputed without
  repeating a matcher run.
- Added `wildfusion_B` and `wildfusion_refined_fraction` so the WildFusion refinement
  budget is auditable next to Vismatch's `candidate_k`.
- Added regression coverage for full-matrix mAP suppression, unscored-tail credit,
  shortlist-reach versus ordering separation, and score-matrix round-tripping.
- Added self-contained `experiments/` artifacts for new probe and finetune runs, including manifests, metrics, timings, training history, and a central `reports/runs.csv` index.
- Added run-local visualization indexes, top-1/failure contact sheets, and `scripts/summarize_runs.py` for filtered Markdown/CSV reporting.
- Added Hydra 1.3 configuration entrypoints and dependency-light composition tests for probe and finetuning.
- Added configurable WildFusion ALIKED keypoint budgets via `local_top_k` (default 512), with forced keypoint counts and cache/experiment identity tracking.
- Added explicit `dataset.image_variant` provenance to standard and Vismatch feature-cache identities, preventing normal and pre-masked feature reuse; added regression coverage.

- Added a repository policy requiring AI agents to update AGENTS.md and
  CHANGELOG.MD after changes and important decisions.
- Added AGENTS.md with architecture, operating rules, configuration contracts,
  validation commands, compatibility conventions, and future work.
- Added lightweight supported-model constants and validation.
- Added regression tests for model defaults, checkpoint discovery, and gradient
  accumulation boundaries.
- Kept configuration regression checks compatible with the user-preserved wildfusion Stage-A default.
- Added pair-counted Vismatch matching progress with ETA and retry-safe progress updates.
- Added dependency-light batching tests covering shape bucketing, candidate placement,
  partial batches, serial mode, score parity scaffolding, and CUDA OOM backoff.
- Added batched Vismatch extraction and feature-level matching for LoMa and
  RDD-LightGlue-compatible profiles.
- Added research-validity regression coverage for stable ranking, unmatched-query
  mAP, identity-level probes, content-aware caches, duplicate-content safety,
  recursive checkpoint discovery, accumulation scaling, and calibration policy.
- Added SHA-256 fingerprint helpers and explicit WildFusion calibration diagnostics.
- Added dependency-light Vismatch checkpoint discovery for exact files and epoch directories, including RDD extractor, LightGlue, and LoMa component schemas.
- Added an optional ex-reid-gated smoke test for constructing RDD-LightGlue with a custom checkpoint and aligned Hydra regression coverage with the preserved LoMa default.
- Added `probe-parallel-czechlynx.sh` and `probe-parallel-wildlife.sh`, separate
  self-submitting capped Slurm arrays for explicit method/checkpoint/`candidate_k`
  comparisons. They preserve `probe.sh`, validate custom matcher checkpoints,
  support task listing and dry-run inspection, and record resolved task commands.
- Added a dedicated `logs/parallel_run/` destination for parallel-array stdout and
  stderr so those logs are separate from ordinary probe logs.
- Added descriptive per-task parallel log copies, task metadata with lifecycle and
  failure summaries, an atomic `logs/index.csv`, and `scripts/summarize_logs.py`
  filters for dataset, animal, method, matcher, checkpoint, and status. Historical
  logs remain untouched.

### Changed

- Changed Vismatch extraction to preprocess each image once. Batch bucketing called
  `prepared_shape()`, which ran the full resize only to read the output shape, and
  extraction then repeated it, costing a measured 8.6 minutes per 39,760-image run.
  `prepare_image()` now returns a `PreparedImage` carrying the tensor and both sizes;
  pending buckets hold that resized tensor instead of the full-resolution source, cutting
  a sampled image from 18.0 MB to 3.2 MB. `prepared_shape` was removed, `extract_frame`
  and `extract_frames_batch` remain as wrappers, and `extract_prepared` now runs under
  `torch.no_grad()` to match the batched path. Features are bit-identical: verified
  against the pre-refactor call sequence for single, same-shape batched, and mixed-shape
  fallback extraction, so existing feature caches remain valid.
- Changed probe and finetune runs to hash each file once instead of three times. Split
  safety checks, `dataset_digest`, and the Vismatch `_cache_key` each read every image, so
  a 39,760-image probe run performed three full SHA-256 passes. `file_digest_cache()` is an
  explicit context manager wrapping a run, keyed on device/inode/size/mtime so the three
  call sites share entries despite building paths differently. Digests are unchanged and
  all cache identities are byte-identical; measured 20x on repeated hashing, saving roughly
  7 minutes per run warm and around 16 minutes cold.
- Decided against memoizing `sha256_file` process-wide: tmpfs reuses a single `st_mtime_ns`
  across rapid same-size rewrites, so an implicit global cache could return a stale digest
  and defeat the content-addressing these hashes exist to provide. Caching is therefore
  opt-in and run-scoped, and unscoped calls re-read the file exactly as before.
- Changed `mAP` and `mAP_eligible` to report `nan` unless every score-matrix position
  carries a real score. On a shortlist matrix the `-inf` tail is ~99.6% of the row and is
  ordered by original database index, so the previous value largely measured metadata row
  adjacency: across Vismatch runs whose `top_1` spanned 0.360-0.412, `mAP` stayed inside
  0.0463-0.0469, versus 0.0168 for a random ranking. Use `mAP_at_k` for those methods.
- Decided that all retrieval metrics are truncated at a single shared cutoff rather than
  ranking the full gallery. `top_k` already behaved this way, which is why it stayed
  trustworthy while `mAP` did not; `mAP_at_k` makes the rest of the metric set consistent.
  `mAP_at_k` divides by `min(relevant, k)` and remains end-to-end, while `rerank_mAP_at_k`
  divides by the hits inside the scored top-k to isolate ordering from shortlist reach.
- Consolidated the Hydra probe comparison budget under `benchmark.candidate_k`
  (default `100`). Vismatch shortlist selection, WildFusion refinement `B`, Local
  LightGlue refinement `B`, and
  `mAP_at_k`/`rerank_mAP_at_k`/`recall_at_k` now derive from this single value.
  The old independent `benchmark.map_at_k`, Vismatch `candidate_k`, and WildFusion
- Added dependency-light regression coverage for shared-budget resolution, database-size
  capping, invalid values, removed overrides, and preserved metric/provenance fields.
- Recorded that historical `mAP` values in `reports/runs.csv` and existing run manifests
  predate this gate and are not comparable across methods. They are not rewritten, and
  they cannot be recomputed because those runs did not persist their score matrices.
- Preserved historical generated artifacts and aggregate CSVs while routing future probe and finetune runs through the readable experiment layout.
- Updated automatic checkpoint discovery to search modern finetune experiments before legacy results, without changing explicit-path priority or model-only safety.
- Moved probe and finetune defaults to `conf/probe.yaml` and `conf/finetune.yaml`, making Hydra the primary single-run configuration interface.
- Preserved project-managed output directories and added fully resolved finetune configuration snapshots.
- Documented the required shared ex-reid conda environment path and activation
  instructions for future AI sessions.
- Standardized shipped finetune and probe defaults on megadescriptor-l.
- Added canonical checkpoint outputs while retaining historical tagged model-only
  checkpoint files.
- Updated probe and Jaguar checkpoint discovery to prefer canonical files and fall
  back to tagged final checkpoints.
- Updated README documentation for supported models, outputs, dependencies,
  checkpoint behavior, and accumulation behavior.
- Documented that primary mAP includes every query, with `mAP_eligible` and
  coverage fields retained for interpretation; linear and efficient probes now
  make identity-level metrics primary and retain image-level diagnostics.
- Documented deterministic original-index tie-breaking, candidate recall, and
  content-hash cache/split identities.
- Migrated RDD-family Vismatch matcher profiles to the Lynx-compatible float-tensor
  preprocessing protocol with a 512-pixel target long side and `/32` spatial rounding.
- Kept LoMa on its matcher-specific Lynx fine-tuning protocol: the same float-tensor
  interpolation, but `/14` spatial rounding for its DINOv2-L/14 descriptor.
- Preserved the existing square-resize protocols for cosine, WildFusion, local
  LightGlue, linear probe, and efficient probe.

### Fixed
- Fixed parallel Slurm array tasks failing with `Permission denied` when they used
  the copied spool script's directory. The launcher now prefers `SLURM_SUBMIT_DIR`
  before creating `logs/parallel_run/`.
- Fixed Vismatch discarding its Stage-A method metrics. `run_vismatch_benchmark` returns a
  fresh metrics dict, so Stage-A results are now merged after it under a
  `stage_a_<method>_` prefix, mirroring the existing Stage-A timing keys. Values are not
  coerced to float because some metrics are string diagnostics. This keeps Stage-A
  shortlist provenance such as `wildfusion_B` on the run instead of dropping it.
- Fixed the unreachable `warn_only_unseen` branch in split safety checks. It was only ever
  passed as `not require_closed_set`, so it was `True` exactly when the guarded block was
  skipped. Unseen identities are now always reported, with `require_b_labels_in_a` choosing
  the response: strict closed-set classification fails, while open-set evaluation warns
  and continues.
  The dead parameter was removed from the signature and both call sites.
- Fixed a `NameError` when a finetune resumed from a checkpoint with no epochs left to run.
  `range(start_epoch, train.epochs)` was empty, so the post-loop code read the unbound epoch
  variable `metrics`. `validate_resume_epochs` now rejects that configuration up front with
  an actionable message, before any training setup or checkpoint writing, and the epoch
  metrics variable is pre-initialized alongside its neighbours. Decided to fail rather than
  train zero epochs and still emit final checkpoints and a completed manifest, which would
  hide an unraised `train.epochs`.
- Fixed run-local `visualizations/index.csv` ranking with `argsort()[::-1]`, which reverses
  a stable ascending sort and therefore resolved ties in descending database-index order,
  the opposite of the documented policy used by the prediction grid and the metrics. The
  index now uses `stable_rank_1d`. Shipped runs used `visualization.top_k: 3` with distinct
  top-3 scores, so existing artifacts are unaffected and are not regenerated; the defect
  would have surfaced on tied scores, such as a matcher returning 0.0 when no confidence
  clears its threshold, or a cutoff reaching past the scored shortlist.
- Fixed identity-level probe metrics indexing the classifier's per-identity output axis
  with a per-database-image boolean mask. `linear_probe` and `efficient_probe` raised
  `IndexError` at the end of every epoch whenever the gallery held more images than
  identities, so neither method could complete a run; `reports/runs.csv` contains no probe
  rows as a result. Identity columns are now selected directly, which is what the previous
  per-image maximum was reaching for: a classifier emits one probability per identity, so
  every image of that identity carried the same value and the maximum was a no-op. Added a
  guard rejecting database label indices outside the classifier head, and regression
  coverage for the image-count/class-count mismatch that triggered the crash.
- Fixed two Vismatch preprocessing test stubs that bypass `__init__` and had not been given
  the new `preprocessing_divisor` attribute, and corrected their `FakeExtractor` to return
  one dict per batch item as native `RDD.extract` does rather than a dict of stacked
  tensors. The serial/batched parity stub now derives values from image content so an
  item/feature misalignment fails the test instead of being hidden by identical
  placeholders. Production Vismatch code was correct and is unchanged.
- Fixed probe final metric printing so string diagnostic fields such as
  `vismatch_cache_fingerprint` no longer fail a completed run during finalization.
- Added regression coverage for mixed numeric and string metric reporting.
- Fixed a probe feature-cache startup regression where the cache key was referenced
  before construction.
- Added regression coverage that verifies cache keys include model-weight identity.
- Fixed probe startup calibration using an undefined dataset variable; the original
  dataset is now passed to calibration split selection.
- Fixed failed-run index creation when the `reports/` parent directory is absent, so
  reporting errors cannot mask the original probe exception.

- Added running/completed/failed run lifecycle reporting so failed experiments retain a concise error manifest.
- Rejected unknown and misspelled Hydra override paths during configuration composition.
- Added `total_run_sec` and `total_run_min` to completed finetuning runs in the
  aggregate training CSV, while preserving incremental epoch writes.
- Fixed the missing `Sequence` typing import that prevented batched Vismatch
  probe runs from starting.
- Added an ex-reid-gated runtime smoke test that invokes batched extraction with a
  fake backend, catching runtime annotation/import failures before full benchmarks.
- Restored explicit batched Vismatch defaults in `conf/probe.yaml` while
  preserving the user-selected LoMa, Stage-A, and candidate settings.
- Prevented gradients from a final partial accumulation group from being discarded
  in finetuning, linear probing, and efficient probing.
- Ensured explicit checkpoint paths take precedence over automatic discovery.
- Ensured full resume checkpoints are not selected as model-only inference
  checkpoints.
- Replaced the stale megadescriptor shipped default with a supported identifier.
- Made best-checkpoint metrics primary in finetune reports while retaining
  final-epoch metrics and checkpoint provenance.
- Removed finite Stage-A fallback scores and the `-1e9` sentinel for non-candidate
  Vismatch positions, using shortlist-only `-inf` scores and candidate-hit/recall
  reporting instead.
- Corrected accumulation loss scaling for full and partial groups in all three
  training loops.
- Added duplicate-image-content split failures and recursive completed-run
  checkpoint discovery with model-only precedence.
- Added Hydra-selectable custom Vismatch checkpoint loading with strict component validation, default-component fallback, content-hash cache isolation, and run-manifest provenance.
- Changed Vismatch probe ranking to initialize non-candidate positions to `-inf`, matching
  WildFusion's shortlist-constrained protocol; added score-policy and pair-coverage
  reporting for fair baseline comparisons.
- Added regression coverage for invalid matcher scores, full/partial shortlist matrices,
  deterministic `-inf` ties, and content-hash-compatible batching fixtures.
- Invalidated the previous Vismatch PIL-resize and LoMa `/14` feature semantics by
  bumping the feature schema and including `lynx_finetuning_v1` in matcher profiles
  and cache fingerprints.
- Corrected Vismatch feature metadata so keypoints and LightGlue image sizes use
  processed `/32` coordinates while source dimensions remain available for provenance.
- Corrected LoMa feature extraction to use processed `/14` coordinates and matcher-specific
  cache/profile identity, preventing the LoMa DINOv2 patch-divisibility assertion seen when
  generic `/32` preprocessing produced a 512-pixel dimension.
- Corrected LoMa match visualizations to plot normalized keypoints in the displayed processed
  image coordinates, including the Vismatch half-pixel convention; raw source dimensions are
  no longer used unless raw images are displayed.
- Aligned the Hydra default-composition regression test with the shipped `vismatch`
  public method and `rdd-lightglue` matcher defaults, while preserving `wildfusion`
  and LoMa as independently selectable methods.
- Fixed WildFusion probe finalization by importing the reporting `file_identity`
  helper; this prevents a completed matching run from failing during manifest assembly.
- Added a regression test that verifies the probe runner retains this reporting import.

### Known Issues

- Decided on 2026-08-17 to record the results of a second audit pass as known issues and
  defer the fixes. Full symptoms, reproductions, and measured impact are under
  "Known issues (open)" in AGENTS.md; the actionable items are also on the future-work
  checklist. Summary of what is open:
  - `scripts/summarize_runs.py --sort-by` raises `TypeError` on the shipped
    `reports/runs.csv`, because `sort_run_rows` mixes float and string sort keys and 11 of
    23 rows have an empty `top_1`. The command documented in README fails today.
  - The same sorter leaves rows unordered when a value is `NaN` and reports no error, which
    became reachable once `mAP` started reporting `nan` for shortlist-constrained methods.
  - Both probes recompute full image-level retrieval metrics each epoch: about 1 minute
    and 2.7 GB of transient allocation per epoch, roughly 50 minutes at the shipped
    `epochs: 50`. The unused similarity-matrix allocation was removed, but this remaining
    per-epoch metric cost was previously unreachable because the probes crashed at epoch 1.
  - Probe per-epoch validation runs on the query/test split. Reported metrics are
    unaffected because they come from the post-loop final-epoch evaluation with no
    best-epoch selection, but per-epoch test curves invite selection on test.
  - Vismatch qualitative top-1 selection lacks an all-unscored guard;
    `_predict_class_probabilities` runs under grad during probe training; and
    `_to_hwc_uint8` would blacken float images if a transform ever preceded the dataset view.
- New visualization contact sheets require Pillow-readable image files and are omitted when no images are available.
- Historical `results/`, `benchmark_runs/`, and top-level visualization artifacts are intentionally not migrated.
- Probe and finetune no longer accept the legacy argparse flags or `--config`; existing scripts must use Hydra dotlist overrides.
- Hydra multirun sweeps are not supported by the project experiment contract.
- Full experiments require heavyweight dependencies, model downloads, and often
  CUDA-capable hardware.
- Custom LoMa checkpoints must match the selected LoMa architecture; the provided RDD-LightGlue checkpoint cannot be used for LoMa.
- Vismatch downloads matcher-specific weights on first use and can be expensive for
  large datasets; wrapped model licenses must be checked individually.
- Default dataset paths are environment-specific; the shared ex-reid environment must
  contain an importable pinned Vismatch installation.
- Masking and RDD improvements are dataset-dependent and must be validated per
  animal/dataset.
- Integration coverage for real backbones, CUDA, Vismatch weights, and private Lynx
  parity is intentionally outside the lightweight unit-test suite.
- The full RDD/LightGlue checkpoint and retrieval parity gate against the external
  Lynx fine-tuning implementation remains environment-gated and requires rerunning
  both default and custom checkpoints at `resize_max=512`.
- Finetuning still uses the current `test`/validation selection behavior; this
  implementation records the limitation rather than changing the protocol.
- Same-set WildFusion calibration excludes diagonal self-pairs by default and
  warns when no disjoint calibration split is configured; official all-pairs
  calibration remains an explicit compatibility mode.
- SHA-256 cache and split checks add I/O cost, and real Vismatch/WildFusion
  parity remains environment-gated.
- The 2026-08-14 code review found that configured WildFusion calibration is not yet
  truly disjoint, because one selected calibration dataset is passed to both sides.
- Mask metadata/content is not yet part of feature-cache fingerprints; changing a
  mask at the same image path can reuse stale features.
- Nested historical checkpoints without manifests are not discovered when automatic
  search starts at the repository-level `results/` root.
- Visualization and Vismatch qualitative rankings still have raw `argsort`/`argmax`
  paths that can disagree on ties with reported metrics.
- Concurrent jobs can race while updating `reports/runs.csv`, and manifests record
  checkpoint size/mtime but not SHA-256 or repository dirty-state identity.


### Added

- Added a pinned Vismatch matcher registry and feature-level backend for
  RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue.
- Added dependency-light tests for matcher validation, profile fingerprints, score
  normalization, empty matches, and Vismatch configuration defaults.
- Added canonical `FrameFeatures` shape/convention validation and an explicit
  migration error for the removed public `rdd` probe method.
- Kept the feature contract and profile registry dependency-light so base-environment
  tests do not import masking, CUDA, or downloaded matcher dependencies.
- Added Vismatch LoMa-B as the `loma` matcher profile, including its original
  14-pixel divisibility preprocessing, normalized-coordinate cache fields,
  mutual-match filtering, and profile-specific threshold default; the later
  Lynx-parity migration standardizes current Vismatch inputs on `/32` dimensions.

### Changed

- Migrated probe and Jaguar local matching from direct external RDD imports to
  configuration-driven Vismatch.
- Renamed the public probe method and Jaguar controls/artifacts from `rdd` to
  `vismatch`; `rdd-lightglue` remains the selected matcher identifier.
- Added feature-cache schema metadata and matcher-specific cache identity.
- Made feature-level matching the explicit production mode and retained pairwise
  matching only as a diagnostics API.
- Repaired the shared ex-reid Vismatch installation and recorded the pinned commit.
- Extended probe and Jaguar configuration examples so `matcher: loma` can be
  selected without changing the public `vismatch` method name.
- Made batched Vismatch execution the default with configurable extraction and
  matching batch sizes, cross-query exact shape bucketing, serial fallback, and CUDA
OOM backoff.
- Added effective/configured batch-size and batch-count timing fields without changing
  Stage-A candidate selection, candidate_k, matrix placement, or score normalization.

### Fixed

- Removed direct RDD repository/config/weight path requirements from shipped
  configurations.
- Preserved matched-confidence score normalization and Jaguar Stage-A fusion
  semantics while changing the matcher backend.
- Preserved the Lynx LoMa scoring protocol: mutual-match confidences summed and
  divided by the smaller keypoint count, with the reference 0.10 filter threshold.
- Added a clear migration error for callers that still request the removed public
  `rdd` probe method.
- Recovered from interrupted/corrupt feature-cache writes with atomic cache replacement
  and recomputation of invalid `.npz` entries.

### Known Issues

- The Vismatch backend depends on native feature/matcher components exposed by the
  pinned Vismatch commit; changing that dependency requires a parity check.
- LoMa requires its Vismatch-managed weights and the pinned LoMa dependencies;
  changing the LoMa wrapper or threshold requires a separate parity check.
- Private Lynx golden-subset parity and CUDA matcher smoke tests still require the
  external dataset, downloaded weights, and the ex-reid environment.
- Exact shape bucketing is deliberately conservative: heterogeneous feature shapes
  may fall back to serial processing, so throughput depends on the dataset and matcher.
- Batched-vs-serial CUDA parity and the 100% top-1 golden-subset gate remain environment-
  gated checks; lightweight tests do not prove numerical equivalence of downloaded models.
- Full available-split parity was measured on 2026-08-12 with 66 queries and 217
  gallery sequences: 98.48% top-1 ranking agreement, identical top-5 accuracy,
  mAP difference -0.0018, and mean absolute per-pair score difference 4.38e-05.
  One near-tie changed the top-1 result; strict 100% golden-subset parity is not
  yet established.

## Historical Features

- Modular ArcFace backbone finetuning with retrieval validation.
- Retrieval benchmarking using cosine similarity, WildFusion, local LightGlue,
  linear probes, efficient ViT probes, and RDD reranking.
- Optional COCO-RLE background removal and feature caching.
- Split-overlap and identity-coverage safety checks with saved reports.
- Optional Weights & Biases logging and qualitative retrieval, attention, and match
  visualizations.
- Standalone Kaggle Jaguar pipeline with stratified validation, alpha-mask support,
  Stage-A retrieval, optional RDD fusion, score validation, and submission export.

\n
