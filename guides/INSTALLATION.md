# Installation

How to install the `wildmatch` package, where it looks for data and checkpoints, how to fetch the
paper's checkpoints and how to run the tests. Back to the [README](../README.md).

- [Requirements](#requirements)
- [uv (recommended)](#uv-recommended)
- [Data locations](#data-locations)
- [Conda](#conda)
- [Paper checkpoints](#paper-checkpoints)
- [Testing](#testing)

## Requirements

Python 3.12. Pick exactly one torch build: `cu126` (CUDA 12.6) or `cpu`.

## uv (recommended)

```bash
uv sync --extra cu126 --extra matchers --group dev   # or --extra cpu
uv run wildmatch evaluate --help
```

`--extra matchers` adds Vismatch (RDD-LightGlue, LoMa and the other local matchers);
`--extra train` adds what pair mining and matcher fine-tuning need (accelerate, LoMa, PoseLib);
`--extra wandb` adds Weights & Biases logging. `uv.lock` pins every package, including the
git dependencies (wildlife-tools, wildlife-datasets, Vismatch, glue-factory, LightGlue).
uv creates `.venv/` in the repository unless `UV_PROJECT_ENVIRONMENT` points elsewhere,
for example to keep large environments off a small home directory:

```bash
export UV_PROJECT_ENVIRONMENT=/path/with/space/wildmatch
```

## Data locations

Paths come from a profile in `src/wildmatch/conf/paths/`: `default` expects datasets under
`./data`, writes caches to `./cache` and looks for checkpoints in `./checkpoints` (override with
`WILDMATCH_DATA_ROOT`, `WILDMATCH_CACHE_ROOT`, `WILDMATCH_CHECKPOINT_ROOT`); `gmum` is the GMUM
cluster layout. Choose one per run with `paths=<name>`, per shell with `WILDMATCH_PATHS`, or per
checkout with a gitignored `wildmatch.local.yaml` containing `paths: <name>`. Datasets are
selected from the registry in `src/wildmatch/conf/dataset/` with `dataset=<key>`, for example
`wildmatch evaluate dataset=salamander benchmark.method=cosine`.

## Conda

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

## Paper checkpoints

The fine-tuned matcher checkpoints of the paper are on the Hugging Face Hub. Datasets are
prepared separately with `wildmatch prepare`; see [DATASET.md](DATASET.md).

```bash
wildmatch weights list                        # the paper's fine-tuned matcher checkpoints
wildmatch weights download --dataset salamander   # from turhancan97/wildmatch-checkpoints (public; no token needed)
wildmatch weights verify
```

Downloaded checkpoints land where the dataset registry expects them (under the profile's
`checkpoint_root`), so fine-tuned sweep rows find them without extra settings.

## Testing

Run the tests (pytest; the tests are unittest-style classes):

```bash
uv run pytest                       # or: python -m pytest, inside the conda env
uv run pytest -m "not gpu and not data"    # what runs on any machine (no datasets, no GPU)
uv run ruff check src paper tests          # lint
uv run ruff format --check src paper tests # formatting
```

GitHub Actions ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) runs the lint, format and
CPU test steps with the CPU PyTorch build, and builds the project page with `mkdocs build --strict`, on every push.
