# Plan: turn the repository into the installable `wildmatch` package (uv + pyproject)

## Context

This repository is the main codebase for the WildMatch paper and will be made public after
the notification. Today it only runs on the GMUM cluster:
- cluster paths in 16 scripts, 14 tests, both Hydra configs, the launchers and the Slurm wrappers;
- no installable package (`reid/` + top-level `models/`, 28 files patch `sys.path`);
- a heavy conda/pip setup (`torch==2.8.0+cu126`, several never-imported packages);
- dataset settings duplicated (launcher profiles, `PAPER_PROFILES`, and later the sibling
  repositories' configs);
- grids only through Slurm launchers.

Goal: a modular package that installs with uv (conda kept as a second route) and runs from
one `wildmatch` CLI, on CPU or GPU, on any machine. Mining (`rdd-parallel-benchmark`) and
matcher fine-tuning (`lynx-finetuning`) will be merged in later, so they get defined slots now.
The work happens on a separate branch, phase by phase, and every phase must reproduce the
paper code's numbers.

Decisions (32 yes/no answers + follow-ups, 2026-10-04):

| Topic | Decision |
| --- | --- |
| Baseline | Commit pending work to `main`, tag `paper-v1`, record reference outputs, phased branch with review |
| Package | `wildmatch`, `src/` layout, Hydra kept (configs inside the package), single `wildmatch` CLI |
| Environment | Python 3.12, CPU and CUDA extras, conda route kept, unused deps removed |
| Data | One dataset registry, default `./data` + env override, `prepare` downloads where licences allow, cluster `paths=gmum` profile |
| Weights | Hugging Face Hub + `wildmatch weights download`; only the paper's matcher-only LoMa/RDD checkpoints |
| Scope | ArcFace fine-tuning and classifier probes stay; paper/page tooling to `paper/`; docs and video stay; Slurm to `slurm/`; local `sweep` |
| Quality | GitHub Actions CI, pytest, ruff with one reformat, tiny CPU demo |
| Future merge | Empty `mining/` and `matcher_finetune/` slots now; later merge with history (`git subtree`) |
| Compatibility | Update docs/reproduce on the branch; caches may be recomputed once; **old `experiments/` runs must stay readable** |
| Licence | Apache-2.0 (vendored third-party code keeps its own) |
| Git | Branch `refactor/wildmatch-package`, pushed to the private GitHub repo (no deploy, repo stays private) |
| Environments | uv venv at `$UV_ENV_ROOT/wildmatch` (`/shared/results/common/kargin/projects/uv-environment`, home is small), new conda env under the shared miniconda; `ex-reid` untouched |

## Target layout

```
pyproject.toml  uv.lock  environment.yml  LICENSE (Apache-2.0)  NOTICE  THIRD_PARTY_LICENSES/
src/wildmatch/
  cli.py                     `wildmatch <subcommand>` (argparse dispatcher; Hydra subcommands compose configs)
  conf/                      Hydra configs as package data
    evaluate.yaml finetune_backbone.yaml
    paths/default.yaml paths/gmum.yaml
    datasets/<name>.yaml     the dataset registry (one file per dataset/split)
    sweep/<name>.yaml        method x k grids (replaces launcher VARIANTS/CANDIDATE_K tables)
    mining/  matcher_finetune/   empty config groups (slots)
  data/        dataset_view, safety_checks, registry.py, prepare/ (jaguar, unseen split, downloads)
  models/      model.py, objective.py
  matchers/    vismatch*, wildfusion_calibration (from reid/methods)
  evaluate/    probe_runner (+ classifier probes), metrics, ranking, candidate_scoring
  train/       finetune_runner, accumulation, checkpointing, class_weights, results
  reporting/   artifacts, summary, timing, visualizations, wandb_naming, paper_tables, plot_figures, paper_datasets
  sweep/       task table, manifest (from scripts/probe_parallel_manifest.py), log metadata (from scripts/probe_log_metadata.py, summarize_logs.py)
  weights.py   Hub download + SHA-256 verification
  mining/  matcher_finetune/   empty subpackages with the interface docstring (slots)
  utils/       io, fingerprints, cache_identity, repro, config_defaults
slurm/         submit_sweep.sh (array job calling `wildmatch sweep --task-index`), examples
paper/         page exporters, paper figures/tables drivers, analyses (import the installed package)
docs/ overrides/ mkdocs.yml video/   unchanged location
tests/         unit (CPU, no data) | integration (fixtures) | marked `data` / `gpu`
demo/          ~20 CC BY 4.0 synthetic CzechLynx renders + metadata + ATTRIBUTION
```

## Phases (each ends with: tests green, parity check where execution changed, AGENTS.md + CHANGELOG.MD updated, your review)

### Phase 0: baseline on `main`
1. Commit the pending work (JaguarReID, Kaggle removal, cleanup) in topic commits; push `main`.
2. Tag `paper-v1` on that commit and push the tag.
3. Reference outputs (you run them on GPU with commands I give, from `main`/`paper-v1` in `ex-reid`):
   SalamanderID2025 (246 queries, has fine-tuned checkpoints) at k=50: cosine, WildFusion,
   default + fine-tuned LoMa, default + fine-tuned RDD-LightGlue, weighted frozen linear probe;
   plus the existing JaguarReID `split_v2` k=10 runs. Record their run ids in `notes/parity_reference.md`.
4. Create `refactor/wildmatch-package` from the tag and push it.

### Phase 1: environment and packaging (no behaviour change)
- `pyproject.toml` (hatchling backend): `requires-python = ">=3.12,<3.13"`; core deps; extras
  `cpu` / `cu126` declared as conflicting, each mapping torch 2.8.0 / torchvision 0.23.0 to its
  PyTorch index via `[tool.uv.sources]` + `[[tool.uv.index]]`; extras `matchers` (vismatch,
  pinned), `wandb`, empty `mining`; dependency groups `dev` (pytest, ruff), `docs` (mkdocs pins
  from requirements-docs.txt), `paper` (plotting extras). Git deps pinned as today
  (wildlife-tools `e762a6c4`, wildlife-datasets `fc702c3c`, vismatch `4a743b75`).
- Dependency trim: remove packages with no direct import (zennit, pytorchwildlife, geopandas,
  scikit-learn-intelex, submitit, seaborn, iopath, termcolor, ...) only after `uv pip check`
  and importing vismatch / wildlife-tools / wildlife-datasets in the new env succeed.
- Python 3.12 forces Pillow >= 10.1 (Pillow 9.5.0 has no 3.12 wheels); opencv pin checked for
  cp312. Both are exactly what the parity check must cover.
- Env: `UV_PROJECT_ENVIRONMENT=$UV_ENV_ROOT/wildmatch uv sync --extra cu126 --group dev`
  (documented in README and AGENTS.md); `environment.yml` = python 3.12 + `pip install -e .[cu126,matchers]`,
  created as a new conda env under the shared miniconda.
- Move code with `git mv` (history kept): `reid/*` and `models/` into `src/wildmatch/` per the
  layout; `conf/` into `src/wildmatch/conf/`. Mechanical import rewrite `reid.` -> `wildmatch.`;
  delete the 28 `sys.path` inserts (scripts import the installed package).
- `train/probe.py` and `train/finetune.py` stay as thin wrappers calling the package (launchers
  keep working until Phase 3); Hydra `config_path` now points into the package.
- Tests: pytest configured in `pyproject.toml` (markers `gpu`, `data`), unittest-style tests run
  unchanged; imports updated.
- Parity: rerun the Phase-0 subset from the branch env; compare with a new
  `paper/tools/parity_check.py` (loads both runs' `scores.npz` + `metrics.json`; RDD/cosine/WildFusion
  score tolerance 1e-4 and identical Top-1 lists, LoMa 1e-2 because of bfloat16 autocast,
  probe metrics within seed noise).

### Phase 2: paths and dataset registry
- `conf/paths/default.yaml`: `data_root: ${oc.env:WILDMATCH_DATA_ROOT,./data}`, plus
  `cache_root`, `checkpoint_root`, `output_root` (experiments) and `logs_root` the same way.
  `conf/paths/gmum.yaml` = today's `/shared` locations, so cluster runs need no env vars and
  resolve to the same absolute paths (existing caches keep hitting on the cluster).
- `conf/datasets/<name>.yaml` for every current profile (13 WildlifeReID-10k animals,
  SalamanderID2025, JaguarReID `split_v2`, CzechLynx closed/open/unseen): root relative to
  `data_root`, metadata file, label/split columns and values, mask mode, image_variant,
  calibration size, checkpoint layout per matcher, `paper: true/false`, licence, source.
- `wildmatch.data.registry` loads it. `PAPER_PROFILES` / `ALL_PROFILES` in
  `reporting/paper_datasets.py` are rebuilt from the registry (same objects, same keys), so
  class balance, audit, figures and page exporters keep their API.
- Replace absolute paths in conf, scripts and tests with registry/paths lookups. Data-dependent
  tests get `@pytest.mark.data` and skip when the data root is absent; pure tests use fixtures.
- Backward compatibility: readers (`paper_tables`, `plot_figures`, `summary`, export scripts)
  keep accepting old manifests with absolute paths; a fixture with an old-format manifest pins it.

### Phase 3: CLI, sweeps, Slurm, layout
- `[project.scripts] wildmatch = "wildmatch.cli:main"` with subcommands: `evaluate` (today's
  probe, Hydra overrides passed through), `finetune-backbone`, `sweep`, `prepare`, `weights`,
  `tables`, `figures`, `summarize-runs`, `summarize-logs`, `build-unseen-split`, `class-balance`,
  `audit`, `demo`.
- `wildmatch sweep conf=<sweep>.yaml`: builds the task table in Python (replacing the bash
  `VARIANTS`/`DATASET_PROFILES`/`CANDIDATE_K_VALUES` tables) with the registry; `--local` runs
  tasks sequentially; `--emit-tasks` / `--task-index N` serve `slurm/submit_sweep.sh`. The
  immutable submission snapshot, SHA-256 checkpoint/owner validation and per-task log records
  move over unchanged (logic from `scripts/probe_parallel_manifest.py`, `probe_log_metadata.py`);
  `tests/test_probe_parallel*.py` and `test_parallel_log_reporting.py` are ported to the new task builder.
- `probe-parallel-*.sh`, `probe.sh`, `finetune.sh`, `scripts/eval_loma_epoch_curve.sh` move to
  `slurm/` (partitions/QOS as variables); `train/` wrappers removed.
- Paper/page tooling moves to `paper/` (all page exporters, `plot_*`, `export_paper_tables`,
  `analyze_background_matches`, `segment_with_sam3` (lynx-app env), `build_demo_cards`); core tools
  (`build_unseen_eval_metadata`, `prepare_jaguar_metadata`, `export_class_balance`,
  `audit_image_quality`, `summarize_*`) become package modules behind the CLI.
- Empty `mining/` and `matcher_finetune/` subpackages + config groups with a short interface
  note (inputs: registry dataset + cached features; outputs: pair index / checkpoint directory
  with protocol json under `checkpoint_root`), matching how the sibling repos' outputs are consumed today.

### Phase 4: data preparation and weights
- `wildmatch prepare <dataset>`: JaguarReID (existing `prepare/embed/split` steps, user supplies
  Kaggle files), unseen split (existing builder), WildlifeReID-10k and CzechLynx download (verify
  the wildlife-datasets / Zenodo routes first; fall back to "place files here" instructions).
  Salamander and Jaguar are Kaggle data: no redistribution, user-supplied.
- `wildmatch weights download --dataset X --matcher loma|rdd-lightglue`: Hub repo + filename +
  SHA-256 per entry in the registry; paper checkpoints only (8 datasets x 2 matchers, epoch 299,
  matcher-only; Nyala uses the SHA of `model__actual_nyala.safetensors`). Upload is your action;
  the Hub repo stays private until the notification (token supported).

### Phase 5: quality, demo, docs
- ruff config; one formatting-only commit, separate from logic changes.
- GitHub Actions: `uv sync --extra cpu --group dev`, `ruff check`, `pytest -m "not gpu and not data"`,
  `mkdocs build --strict` (docs group). No deploy step.
- `demo/`: ~20 synthetic CzechLynx renders (CC BY 4.0, already exported for the page) +
  metadata; `wildmatch demo` runs cosine and default LoMa on CPU in minutes.
- README quickstart (uv, conda, CPU/GPU, data, weights, evaluate, sweep); AGENTS.md rewritten
  for the new layout; CHANGELOG entries per phase; `docs/reproduce/*` and their tests updated to
  the new commands; LICENSE (Apache-2.0), NOTICE, third-party licence files (RDD, LoMa, vismatch).

## Out of scope

Merging the sibling repositories (later, with `git subtree`), retraining, uploading weights,
making the repository or Hub public, deploying the page, re-running the full paper grid.

## Critical files

`train/probe.py`, `train/finetune.py`, `conf/*.yaml`, `reid/**`, `models/**`,
`reid/reporting/paper_datasets.py`, `reid/utils/cache_identity.py`, `probe-parallel-*.sh`,
`scripts/probe_parallel_manifest.py`, `scripts/probe_log_metadata.py`, `scripts/summarize_logs.py`,
`requirements*.txt`, `environment.yml`, `tests/test_hydra_config.py`, `tests/test_probe_parallel*.py`,
`docs/reproduce/*.md`, `README.md`, `AGENTS.md`, `CHANGELOG.MD`.

## Verification (per phase)

- `uv sync --frozen --extra cpu --group dev` and `--extra cu126` both resolve in `$UV_ENV_ROOT/wildmatch`;
  the conda route installs from `environment.yml`.
- `pytest -m "not gpu and not data"` green on CPU; full suite green on the cluster (`paths=gmum`).
- Parity (Phases 1-3): you run the Phase-0 subset from the branch (`wildmatch sweep ... paths=gmum`
  on Slurm, commands provided); `paper/tools/parity_check.py` passes against the `paper-v1` reference runs.
- Old runs: `wildmatch tables` and `wildmatch figures` regenerate today's `reports/paper_tables`
  and figures from the existing 669 runs byte-for-byte or with documented differences only.
- `wildmatch sweep --local` and `slurm/submit_sweep.sh --list-tasks` produce the same task table.
- `wildmatch demo` runs on a CPU-only machine from a fresh clone.
- `mkdocs build --strict` passes; `tests/test_project_page*.py` green.
