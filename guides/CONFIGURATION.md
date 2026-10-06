# Configuration

The Hydra configuration of `wildmatch evaluate` (probes) and `wildmatch finetune-backbone`: the
dataset contract, supported backbones, each method's settings, checkpoints, W&B and
reproducibility. Back to the [README](../README.md); running experiments and reading their
outputs is in [EXPERIMENTS.md](EXPERIMENTS.md).

- [Hydra](#hydra)
- [Dataset requirements](#dataset-requirements)
- [Supported models](#supported-models)
- [`src/wildmatch/conf/finetune.yaml`](#srcwildmatchconffinetuneyaml)
- [`src/wildmatch/conf/probe.yaml`](#srcwildmatchconfprobeyaml)
- [Safety checks](#safety-checks)
  - [Building an unseen-identity evaluation split](#building-an-unseen-identity-evaluation-split)
- [Linear probe](#linear-probe)
- [Efficient probe](#efficient-probe)
- [Vismatch](#vismatch)
- [Checkpoints and gradient accumulation](#checkpoints-and-gradient-accumulation)
- [Weights & Biases (W&B)](#weights--biases-wb)
- [Reproducibility](#reproducibility)

## Hydra

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

## Dataset requirements

Every dataset is a registry entry (`src/wildmatch/conf/dataset/<key>.yaml`): a root folder under
the profile's `data_root` and a metadata CSV with an image path (relative to the root), an
identity column and a split column with a database and a query value. Masked inputs are either
pre-masked images the CSV points at (WildlifeReID-10k, SalamanderID2025) or a COCO run-length
`mask` column applied at load time (`dataset.no_background: true`, CzechLynx).

**How to download and prepare each dataset, check the result and set up SAM 3:
[DATASET.md](DATASET.md).**

## Supported models

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

The WildlifeReID-10k registry entries cover NyalaData, WhaleSharkID, BelugaID,
ZindiTurtleRecall, ATRW, Giraffes, LeopardID2022, HyenaID2022, GiraffeZebraID, CowDataset,
StripeSpotter, and SeaStarReID2023. They read `metadata_sam3/metadata_<animal>.csv`, with
`identity` as the label column and `split` values `train` and `test`; these tables point at the
SAM 3 pre-masked `masked_images_sam3/` tree, so the entries use `image_variant: no_background`
and `no_background: false`. The tables the paper's runs read are recorded as
`registry.paper_inputs` (see [DATASET.md](DATASET.md#51-wildlifereid-10k-12-entries)).

## `src/wildmatch/conf/finetune.yaml`

Key blocks:
- `dataset`: root, metadata file, split values, `no_background`, `mask_col`, and `image_variant` (`background` or `no_background`)
- `model`: backbone type
- `train`: epochs, batch size, AMP, deterministic mode, resume checkpoint
- `loss`: ArcFace parameters
- `scheduler`: cosine settings
- `output`: experiment root, save frequency, best metric, `selection`, and legacy aggregate CSV path.
  The per-epoch evaluation split (`dataset.val_split_value`) is the test split, so `selection: final`
  (default) reports the final-epoch model; `best_on_test` reports `checkpoint-best.pth`, chosen on
  the test split, and the metrics say so (`selected_on: test`).
- `reporting`: run-index path and reporting enablement
- `benchmark`: validation retrieval metrics (`top_k`, `mAP`)
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

## `src/wildmatch/conf/probe.yaml`

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
- `benchmark.calibration`: how WildFusion and Local LightGlue fit their score calibration on
  `dataset.calibration_size` images. `mode: same_set` (default, every earlier run) uses the first
  rows of the database on both sides and drops each image's pair with itself (`exclude_self_pairs`;
  `official_same_set: true` keeps them). `mode: disjoint` draws two image sets with `seed`: identities in
  random order, two images of one identity per turn, one on each side, so the sides share identities
  but no image. `split_value` calibrates on other rows of `dataset.split_col` instead of the database;
  the query split is refused. The chosen mode, seed, side sizes and shared identities are recorded
  under `calibration` in the run manifest.
- `visualization`: optional qualitative retrieval plots
- `output`: experiment root, legacy run folder, and aggregate CSV
- `reporting`: central run-index path
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

## Safety checks

The shipped `probe.yaml` sets `safety_checks.enabled: false`, and sweeps inherit it; before
publishing results on a new dataset or split, check it once with `safety_checks.enabled=true`.
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
model construction. When safety checks are enabled (the shipped `probe.yaml` disables them),
they report the coverage gap in every case.

The classifier coverage fields include seen/unseen query images and identities plus
`classification_query_seen_coverage`. When
`benchmark.classifier_evaluation.embedding_retrieval: true`, an optional cosine
retrieval diagnostic is reported under `embedding_*`; it is separate from the
classifier-head metrics and is disabled by default. No unknown classifier class is
added.

### Building an unseen-identity evaluation split

To rebuild the paper's CzechLynx unseen-identity split, run `wildmatch prepare unseen-split`
([DATASET.md](DATASET.md#53-czechlynx-unseen-identity-split)); this section describes the
general tool behind it.

For an evaluation-only open-world test, the standalone
`wildmatch build-unseen-split` utility can derive a new metadata CSV
without changing the probe pipeline. It selects
identities that occur in the source query split but not in the source database
split, assigns the earliest encounter to `database`, and assigns later
encounters to `query` using the supplied order column. It never performs a
random image-level split. For example, for CzechLynx:

```bash
wildmatch build-unseen-split \
  --metadata <data_root>/CzechLynx_v2/CzechLynxDataset-Metadata-Real.csv \
  --output-dir /path/to/czechlynx-unseen-eval \
  --label-col unique_name \
  --source-split-col split-time_open \
  --database-value train \
  --query-value test \
  --group-col encounter \
  --order-col date \
  --root <data_root>/CzechLynx_v2
```

The output contains `metadata_unseen_eval.csv` and
`unseen_eval_manifest.json`. The manifest records source/output SHA-256 hashes,
selected and excluded identities, grouping parameters, path-overlap checks, and
cross-split duplicate-content checks. Missing files, malformed grouping/order
values, path overlap, or duplicate image content fail closed. Archive both
outputs with the resulting experiment. Consume the generated CSV through the
normal Hydra contract:

```bash
wildmatch evaluate \
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

The registry entry `czechlynx_unseen_eval` runs it in a sweep. Point
`CZECHLYNX_UNSEEN_EVAL_METADATA_FILE` at the generated CSV, list
`czechlynx_unseen_eval` under `datasets:` in a sweep spec, and inspect it with
`wildmatch sweep <spec> --list-tasks`. The sweep fails before submission if the generated
metadata file is missing or if its split settings are not `unseen_eval_split`, `database`,
and `query`.

## Linear probe

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
- `log_test_each_epoch` (default `false`): also evaluate the query (test) split after every epoch and
  log `test_*` loss and metrics; diagnostics only, never for choosing an epoch. Reported metrics
  come from the final-epoch model and are identical either way.
- `lr`, `momentum`, `weight_decay`, `eta_min_scale`
- `eval_batch_size`, `eval_num_workers`
- `resume_checkpoint`
- `save_checkpoint` (default `false`), `save_every`, `final_checkpoint_name`, `keep_last_epoch_checkpoints`
  (default `null`: keep every `*_epoch_<n>.pth`; a number keeps the newest N)
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

## Efficient probe

`efficient_probe` applies a softmax head on top of ViT patch-token outputs:
- token source: `outputs.last_hidden_state[:, -number_of_patches:, :]`
- supports train modes: `all` | `partial` | `classifier`
- logs train loss and top-k metrics with tqdm progress bars (test-split curves only with `log_test_each_epoch`)
- when `visualization.enabled: true`, also saves a single attention-overlay grid from query images

Config path:
- `benchmark.methods.efficient_probe`

Core options:
- `train_mode`, `epochs`, `log_every`
- `log_test_each_epoch` (default `false`): also evaluate the query (test) split after every epoch and
  log `test_*` loss and metrics; diagnostics only, never for choosing an epoch. Reported metrics
  come from the final-epoch model and are identical either way.
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

## Vismatch

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
wildmatch evaluate benchmark.method=vismatch benchmark.methods.vismatch.matcher=rdd-lightglue benchmark.methods.vismatch.checkpoint_source=custom benchmark.methods.vismatch.checkpoint_path=/path/to/epoch_15 benchmark.methods.vismatch.checkpoint_components=matcher_only
```

The resolver detects components from tensor schemas and optionally validates `checkpoint_manifest.json`; it does not trust filenames such as `model.safetensors` or `model_1.safetensors`. RDD-LightGlue may combine custom RDD and LightGlue files, while a LightGlue/RDD checkpoint is never accepted for LoMa. Custom component hashes are included in feature-cache identities and run manifests.

Descriptor-only fine-tuning is selected explicitly with `checkpoint_components=descriptor_only`:

```bash
wildmatch evaluate benchmark.method=vismatch \
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

Descriptor rows (`checkpoint: descriptor-fine-tuned`) need a registry entry or an explicit
checkpoint path; only CzechLynx closed has one. Cross-species tests must declare a separate
checkpoint owner and evaluation animal in `dataset_overrides`; ownership mismatches fail closed.

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
`4a743b75749a3770af59d275483ed341dea51ff0` (`[tool.uv.sources]` in `pyproject.toml`). Its matcher
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

## Checkpoints and gradient accumulation

The standard finetune-to-probe workflow uses checkpoint-final.pth. If automatic
discovery is enabled, the newest run is searched for the configured canonical
filename first and then for compatible tagged model-only checkpoints. Full
checkpoints are never selected for inference.

Checkpoints are written atomically: a crash or a full disk leaves the previous file intact.
`output.keep_last_epoch_checkpoints` keeps only the newest N `checkpoint-epoch-<n>` files of a
fine-tuning run (default `null` keeps all); final, best and latest checkpoints are never deleted.
Legacy result folders without a run manifest are found at any depth under `results/`.

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

## Reproducibility

- seed control available in both configs
- deterministic mode toggle (`deterministic: true/false`)
- finetune resume from full checkpoints via `train.resume_checkpoint`
- probe feature caching keyed by method/model/checkpoint/dataset signature and `no_background`
- Cache identities also include the resolved dataset root, metadata file, and explicit `dataset.image_variant` (`background` or `no_background`) so normal and pre-masked features cannot be reused interchangeably.
