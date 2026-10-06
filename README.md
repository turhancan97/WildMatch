<p align="center">
  <img src="docs/assets/logo/wildmatch-fullname.png" alt="WildMatch" width="360">
</p>

<p align="center">
  <a href="https://github.com/turhancan97/WildMatch/actions/workflows/ci.yml"><img src="https://github.com/turhancan97/WildMatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-Apache--2.0-green" alt="Licence: Apache-2.0"></a>
</p>

# WildMatch

Code for *WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification*.
WildMatch identifies individual animals: given a query photo, it ranks the gallery photos by how
likely they show the same individual. Starting from a fixed MegaDescriptor-L candidate list, a
pretrained keypoint matcher (LoMa or RDD-LightGlue) scores each query against its candidates by
feature matching. WildMatch adapts that matcher to a wildlife domain with identity labels only: it
mines matching and non-matching image pairs with the pretrained matcher and fine-tunes the matcher
on them, without any keypoint or correspondence annotation.

## Overview

The code is the installable `wildmatch` package (`src/wildmatch/`), run through one `wildmatch`
command (`wildmatch --help` lists every subcommand):

- **Data**: `wildmatch prepare` downloads and prepares the 17 datasets of the dataset registry
  (WildlifeReID-10k animals, SalamanderID2025, CzechLynx and its unseen-identity protocol), including
  SAM 3 background masks.
- **Pair mining**: `wildmatch mine` matches every training image against the others with the
  pretrained matcher and builds pools of positive and negative pairs from the strong matches.
- **Matcher fine-tuning**: `wildmatch finetune-matcher` fine-tunes LoMa or RDD-LightGlue on the mined
  pairs.
- **Evaluation**: `wildmatch evaluate` (one run) and `wildmatch sweep` (a grid on Slurm or locally)
  run the default and fine-tuned matchers and the baselines: cosine retrieval, WildFusion, and
  linear and efficient classifier probes; `wildmatch finetune-backbone` fine-tunes a backbone with
  ArcFace.
- **Checkpoints and results**: `wildmatch weights` fetches the paper's fine-tuned matchers;
  `wildmatch tables` and `wildmatch figures` build tables and accuracy-versus-budget figures from
  completed runs.

The sources of the project page are in `docs/` (MkDocs Material); its
[Reproduce](docs/reproduce/index.md) pages walk through mining, fine-tuning and evaluation.

## Quick start

```bash
# 1. Install (Python 3.12; pick one PyTorch build: --extra cu126 or --extra cpu;
#    add --extra train for mining and matcher fine-tuning)
uv sync --extra cu126 --extra matchers --group dev
source .venv/bin/activate                     # or prefix the commands below with `uv run`

# 2. Check the installation on any machine (CPU, about a minute, no dataset needed)
wildmatch demo                                # cosine on 24 bundled synthetic lynx renders
wildmatch demo --method loma                  # default LoMa over the cosine candidates (~2 min)

# 3. Data: see what each registry dataset needs, then download or rebuild it
wildmatch prepare status
wildmatch prepare download zindi              # raw WildlifeReID-10k (Kaggle credentials)
wildmatch prepare build zindi                 # split table; prints the SAM 3 masking command
wildmatch prepare finish zindi                # pre-masked metadata, after the masks exist

# 4. The paper's fine-tuned matcher checkpoints (Hugging Face Hub)
wildmatch weights download --dataset zindi

# 5. One evaluation (Hydra overrides)
wildmatch evaluate dataset=zindi benchmark.method=vismatch \
    benchmark.methods.vismatch.matcher=loma benchmark.candidate_k=250

# 6. A grid of evaluations (sweep spec: datasets x budgets x methods)
wildmatch sweep example --list-tasks          # src/wildmatch/conf/sweep/example.yaml
wildmatch sweep my_sweep.yaml --local         # or --submit on Slurm

# 7. Mine pairs and fine-tune a matcher (needs --extra train)
wildmatch mine plan --dataset salamander --backend loma
wildmatch finetune-matcher dataset=salamander matcher_finetune=loma

# 8. Results
wildmatch summarize-runs --format markdown
wildmatch tables && wildmatch figures
```

## Documentation

| Guide | Contents |
|---|---|
| [Installation](guides/INSTALLATION.md) | uv and conda setup, data locations (path profiles), paper checkpoints, tests |
| [Datasets](guides/DATASET.md) | Downloading and preparing all 17 registry datasets, SAM 3 masking, expected counts |
| [Configuration](guides/CONFIGURATION.md) | Hydra settings: datasets, backbones, probes, Vismatch matchers, safety checks, W&B |
| [Experiments](guides/EXPERIMENTS.md) | Single runs, sweeps, outputs, tables and figures, research-validity rules, other tools |
| [Reproduce](docs/reproduce/index.md) | Mining, fine-tuning and evaluation as run for the paper (project page) |
| [Third-party licences](THIRD_PARTY_LICENSES.md) | Licences of dependencies, model weights and datasets |
| [Changelog](CHANGELOG.MD) | Chronological record of changes and decisions |

## Repository structure

```text
.
├── src/wildmatch/          the package
│   ├── cli.py             the `wildmatch` command
│   ├── conf/              Hydra configs, path profiles, dataset registry, sweep specs
│   ├── data/              dataset views, masking, split checks, dataset preparation
│   ├── mining/            pair mining (`wildmatch mine`)
│   ├── matcher_finetune/  matcher fine-tuning (`wildmatch finetune-matcher`)
│   ├── evaluate/          probe runner and metrics
│   ├── matchers/          Vismatch matchers and WildFusion
│   ├── models/            backbone factory and training objectives
│   ├── train/             backbone fine-tuning and checkpointing
│   ├── sweep/             sweeps and immutable submissions
│   ├── reporting/         run manifests, run index, tables and figures
│   ├── utils/             I/O, fingerprints, cache identities
│   └── vendor/rdd/        vendored RDD code
├── guides/                installation, datasets, configuration and experiment guides
├── slurm/                 Slurm scripts for sweeps, evaluation, mining and fine-tuning
├── paper/                 paper figures, project-page exporters and tools
├── tests/                 unit tests (pytest)
├── docs/, mkdocs.yml, overrides/   project page (MkDocs Material)
├── notebooks/             dataset annotation viewer
├── video/explainer/       explainer video sources (separate environment)
├── notes/                 narrative history and parity references
├── pyproject.toml, uv.lock, requirements/, environment.yml   environment
└── AGENTS.md, CHANGELOG.MD   operating guide and change record
```

Generated and ignored: `experiments/` (one directory per run), `reports/` (run index, tables,
figures), `logs/`, `benchmark_runs/`, `wandb/` and `site/` (page build).

## Troubleshooting

- `ModuleNotFoundError: wildmatch`: install the package (`uv sync ...` or the conda route in
  [Installation](guides/INSTALLATION.md)); the code does not patch `sys.path`.
- `wildmatch weights download` is refused: the checkpoint repository is private for now; set
  `HF_TOKEN` or run `hf auth login`.
- Mask decoding errors with `dataset.no_background=true`: the metadata's `mask` column must hold
  valid COCO-RLE (a JSON string or dict) matching the image size.
- CUDA mismatch or availability issues: install the matching PyTorch build (`--extra cu126` or
  `--extra cpu`) and adjust the device and AMP settings in the configuration.

## Licence

The code is licensed under Apache-2.0 (`LICENSE`, `NOTICE`). Dependencies, model weights and
datasets keep their own licences; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
MegaDescriptor-L weights are CC BY-NC 4.0 (non-commercial), and work that uses the SAM 3 masks must
acknowledge SAM.
