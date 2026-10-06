# Preparing the datasets

This guide takes you from a fresh clone to every dataset WildMatch evaluates on, ready for
`wildmatch evaluate`, `wildmatch sweep`, mining and fine-tuning. It covers all 17 entries of the
dataset registry: the eight datasets of the paper, the CzechLynx unseen-identity protocol, and the
other WildlifeReID-10k animals the pipeline supports.

You need the `wildmatch` package installed (see the [README](README.md#installation)), Kaggle
API access and, for the masked datasets, a GPU that can run SAM 3. Downloading and preparing data
does not need the paper checkpoints; those are described in the README
([Paper checkpoints](README.md#paper-checkpoints)).

**Contents**

1. [How the data side works](#1-how-the-data-side-works)
2. [The datasets at a glance](#2-the-datasets-at-a-glance)
3. [Prerequisites](#3-prerequisites)
4. [Setting up SAM 3](#4-setting-up-sam-3)
5. [Recipes](#5-recipes)
6. [Checking the result](#6-checking-the-result)
7. [Troubleshooting](#7-troubleshooting)

## 1. How the data side works

**Where data lives.** All locations come from a *path profile*
(`src/wildmatch/conf/paths/<name>.yaml`). The `default` profile keeps datasets under `./data`
(the `data_root`); set `WILDMATCH_DATA_ROOT` to put them elsewhere. A profile can also be chosen
per command (`--paths <name>` for `wildmatch prepare`, `paths=<name>` for Hydra commands), per
shell (`WILDMATCH_PATHS`) or per checkout (a gitignored `wildmatch.local.yaml` with
`paths: <name>`).

**What each dataset is.** Every dataset is an entry of the registry,
`src/wildmatch/conf/dataset/<key>.yaml`, selected with `dataset=<key>`. An entry names a root
folder under `data_root`, a metadata CSV (image path relative to the root, an identity column and
a split column with a database and a query value), how the images are masked, and where the data
comes from. Its `registry.download` block records the raw source, the derived files and whether
this repository can rebuild them.

**What you have.** `wildmatch prepare status` checks every entry (root, metadata, columns, split
values, a sample of images, the mask column where masks are applied at load time) and prints what
is missing and where it comes from. Start and finish with it:

```bash
wildmatch prepare status                     # all entries
wildmatch prepare status --dataset nyala     # one entry (repeatable)
```

On a fresh machine every entry reports `MISSING  <key>  root missing; metadata missing`, followed
by its `raw:` source and `derived:` steps.

**Layout under `data_root`** after the recipes below:

```text
<data_root>/
├── WildlifeReID-10k/        # 12 entries: Kaggle release + wildmatch_prepare/, masked_images_sam3/, metadata_sam3/
├── CzechLynx_v2/            # czechlynx_closed, czechlynx_open, czechlynx_unseen_eval
├── SalamanderID2025/        # salamander
└── jaguar/                  # jaguar (paused, see 5.5)
```

`CzechLynx_v2` is the folder name the experiment paths and caches use; it holds the Kaggle
CzechLynx release.

## 2. The datasets at a glance

| Key | Dataset | Paper | Source | Route | Steps |
|---|---|---|---|---|---|
| `hyenaid2022` | HyenaID2022 | yes | WildlifeReID-10k (Kaggle) | download + SAM 3 | [5.1](#51-wildlifereid-10k-12-entries) |
| `leopardid2022` | LeopardID2022 | yes | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `nyala` | NyalaData | yes | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `seastarreid2023` | SeaStarReID2023 | yes | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `whaleshark` | WhaleSharkID | yes | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `zindi` | ZindiTurtleRecall | yes | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `beluga` | BelugaID | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `atrw` | ATRW | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `cowdataset` | CowDataset | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `giraffes` | Giraffes | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `giraffezebraid` | GiraffeZebraID | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `stripespotter` | StripeSpotter | no | WildlifeReID-10k | download + SAM 3 | 5.1 |
| `czechlynx_closed` | CzechLynx, time-closed split | yes | CzechLynx (Kaggle) | download | [5.2](#52-czechlynx-closed-and-open) |
| `czechlynx_open` | CzechLynx, time-open split | source of the unseen split | CzechLynx (Kaggle) | download | 5.2 |
| `czechlynx_unseen_eval` | CzechLynx unseen identities | yes (unseen-identity protocol) | derived from `czechlynx_open` | built locally | [5.3](#53-czechlynx-unseen-identity-split) |
| `salamander` | SalamanderID2025 | yes | AnimalCLEF2025 (Kaggle competition) | manual download + SAM 3 | [5.4](#54-salamanderid2025) |
| `jaguar` | JaguarReID | no | Jaguar Re-ID (Kaggle competition) | **paused** | [5.5](#55-jaguarreid-paused) |

**Licences and terms.** Each dataset keeps its own terms; read them before you download, and see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the full table. In short:

- **WildlifeReID-10k**: no commercial use, no re-uploading, and attribution of WildlifeReID-10k and
  of each source dataset, on top of the source licence. Source licences:
  - CDLA-Permissive-1.0: HyenaID2022, LeopardID2022, SeaStarReID2023, WhaleSharkID, BelugaID,
    GiraffeZebraID;
  - CC BY-SA 4.0: ZindiTurtleRecall;
  - CC BY-NC-SA 4.0: ATRW;
  - CC BY-SA 3.0: StripeSpotter;
  - CC BY 4.0: CowDataset;
  - none stated by the source: NyalaData, Giraffes.
- **CzechLynx**: CC BY 4.0.
- **SalamanderID2025**: the AnimalCLEF2025 competition rules, which allow non-commercial use,
  including academic research and education, and forbid redistributing the data.
- **JaguarReID**: the competition rules allow competition use only.

Because of these terms this repository ships no images, masks or prepared tables. The recipes
rebuild them from your own downloads.

## 3. Prerequisites

1. **The package**, installed as in the README. `wildmatch prepare` runs on CPU.
2. **Kaggle API credentials.** Downloads go through the `kaggle` command-line tool, installed with
   the package. Create an API token on your Kaggle account page and put it in
   `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. **Competition rules.** SalamanderID2025 comes from a Kaggle competition. Open the
   `animal-clef-2025` competition page and accept its rules before downloading, or the download
   is refused.
4. **A SAM 3 environment and a GPU** for the 13 masked entries (the WildlifeReID-10k animals and
   SalamanderID2025). See [section 4](#4-setting-up-sam-3). CzechLynx needs no masking step.
5. **Disk space.** Masking writes a masked copy of every image (`masked_images_sam3/` or
   `masked_images/`) next to the originals, so plan for roughly twice the size of the raw images.

## 4. Setting up SAM 3

SAM 3 (Meta's text-promptable segmentation model) removes the image backgrounds. It runs in its
**own environment**: it needs its own PyTorch and CUDA build, and it must never be installed into
the `wildmatch` environment.

**Requirements (from the SAM 3 README).** Python 3.12 or higher, PyTorch 2.7 or higher, CUDA 12.6
or higher with a compatible GPU. Our masking runs used a build with kernels for compute capability
7.5 and newer, so V100 GPUs (7.0) could not run it; RTX 4090, A100 and H100 work.

**1. Create the environment and install SAM 3** (commands from the SAM 3 README; pick the PyTorch
wheel that matches your CUDA):

```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e . && cd ..
```

**2. Add what our masking script needs:**

```bash
pip install pycocotools pandas pillow omegaconf
```

`omegaconf` lets the script read the `wildmatch` path profile from the repository's `src/`
without installing `wildmatch` here.

**3. Download the checkpoint.** The weights are gated: request access on the Hugging Face model
page `facebook/sam3`, then log in and download `sam3.pt`:

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir <sam3-checkpoint-dir>
```

SAM 3 is released under the SAM License, which asks you to acknowledge SAM in publications.

**4. Tell `wildmatch` where SAM 3 is.** Export these in the shell that runs `wildmatch prepare` and
the masking jobs:

```bash
export WILDMATCH_SAM3_REPO=<path>/sam3                          # the cloned repository
export WILDMATCH_SAM3_CHECKPOINT=<sam3-checkpoint-dir>/sam3.pt
export SAM3_PYTHON=<conda envs>/sam3/bin/python                  # used by slurm/sam3_masks.sbatch
```

The masking script also accepts `--sam3-dir` and `--checkpoint` directly.

**5. Running a masking command.** `wildmatch prepare build` (and `retry-empty`) prints the exact
command, of the form `python src/wildmatch/data/prepare/sam3_masks.py <arguments> --segment`.
Run it from the repository root, either:

- **directly, on a GPU machine**, with the SAM 3 environment's Python:

  ```bash
  env -u VIRTUAL_ENV -u PYTHONPATH "$SAM3_PYTHON" src/wildmatch/data/prepare/sam3_masks.py <arguments> --segment
  ```

- **or through Slurm**, passing the printed arguments to the wrapper (adjust the partition and
  QOS, which default to our cluster's):

  ```bash
  sbatch -p <partition> --qos=<qos> slurm/sam3_masks.sbatch <arguments> --segment
  ```

The script checks all of its outputs before loading the model and refuses to replace existing
files unless you pass `--overwrite`.

## 5. Recipes

Run every command from the repository root.

### 5.1 WildlifeReID-10k (12 entries)

The twelve entries `atrw`, `beluga`, `cowdataset`, `giraffes`, `giraffezebraid`, `hyenaid2022`,
`leopardid2022`, `nyala`, `seastarreid2023`, `stripespotter`, `whaleshark` and `zindi` share one
Kaggle release, `wildlifedatasets/wildlifereid-10k`, under `<data_root>/WildlifeReID-10k/`.

**Download once** (any of the twelve keys downloads the whole release):

```bash
wildmatch prepare download nyala
```

Then, **for each entry you need** (here `nyala`):

```bash
# 1. The split table: wildmatch_prepare/NyalaData_split.csv (CPU, seconds)
wildmatch prepare build nyala
#    ...it prints the SAM 3 command; run it as in section 4 (GPU)

# 2. Images where SAM 3 found nothing: retry with extra prompts, else keep the whole photo
wildmatch prepare retry-empty nyala
#    ...prints "KEY: no empty masks", or a second SAM 3 command; run it

# 3. The metadata the registry reads: metadata_sam3/metadata_NyalaData.csv
wildmatch prepare finish nyala

wildmatch prepare status --dataset nyala
```

What each step writes, relative to `<data_root>/WildlifeReID-10k/`:

| Step | Writes |
|---|---|
| `build` | `wildmatch_prepare/<Animal>_split.csv` |
| SAM 3 | `masked_images_sam3/<image path>` and `wildmatch_prepare/masks_<Animal>.csv` |
| `retry-empty` | `wildmatch_prepare/<Animal>_split_empty.csv`; its SAM 3 run writes `wildmatch_prepare/masks_<Animal>_retry.csv` |
| `finish` | `metadata_sam3/metadata_<Animal>.csv`, with the retry masks merged over the empty rows |

`<Animal>` is the entry's sub-dataset name (`NyalaData`, `ZindiTurtleRecall`, ...).

**The split.** `build` reproduces the closed-set split of WildFusion (arXiv:2408.12934), which
comes from the WildlifeDatasets toolkit: every individual appears on both sides, 80 % of its
images go to the database (`train`) and 20 % to the query (`test`), with seed 666. The Kaggle
release has its own `split` column, which differs from this one on about 30 % of rows; the paper
uses this split. The table rebuilds exactly; check its SHA-256 in
[section 6](#6-checking-the-result). BelugaID uses only the images under its `beluga/` folder.

**The masks.** SAM 3 is prompted with each image's species, keeps the largest detection (one
animal per photo) and blackens the background. Images with no detection are retried with extra
prompts and finally `Animal`; if still nothing is found, the whole photo is kept and flagged in
the `sam3_full_frame` column.

| Key | First prompt | Retry prompts (then `Animal`) |
|---|---|---|
| `atrw` | species | big cat |
| `beluga` | "beluga whale" | whale, beluga |
| `cowdataset` | species | cattle |
| `giraffes` | species | animal neck |
| `giraffezebraid` | species | horse |
| `hyenaid2022` | species | dog |
| `leopardid2022` | species | big cat |
| `nyala` | species | antelope, deer |
| `seastarreid2023` | species | starfish |
| `stripespotter` | species | horse |
| `whaleshark` | "whale shark" | shark, fish |
| `zindi` | species | turtle |

"species" is the release's `species` value for the image.

**Masks and the paper.** The paper's WildlifeReID-10k runs used masked images that the team made
before this pipeline existed, and that method was not recorded. The masks this recipe produces
are new. They agree closely with the paper's (median IoU 0.98 to 0.999 per dataset), so results
can differ slightly from the published numbers. To measure the agreement on your own masks:

```bash
wildmatch prepare compare-masks nyala --masks-csv <data_root>/WildlifeReID-10k/wildmatch_prepare/masks_NyalaData.csv
```

### 5.2 CzechLynx (closed and open)

```bash
wildmatch prepare download czechlynx_closed      # Kaggle picekl/czechlynx into <data_root>/CzechLynx_v2/
wildmatch prepare status --dataset czechlynx_closed --dataset czechlynx_open
```

Both entries read the release's `CzechLynxDataset-Metadata-Real.csv` as published: the
time-closed (`split-time_closed`) and time-open (`split-time_open`) split columns, values
`train`/`test`. The release ships a COCO run-length mask per image, which is applied when an image
is loaded, so there is no masking step. The paper reports the closed split; the open split is the
source of the unseen-identity protocol below.

### 5.3 CzechLynx unseen-identity split

The paper's unseen-identity protocol evaluates on individuals that never appear in training. It
is rebuilt from `czechlynx_open` (download it first, 5.2):

```bash
wildmatch prepare unseen-split
```

It takes the query individuals absent from the open split's training side, groups their photos
by encounter, gives the earliest encounter of each individual to the database and the later ones
to the query, and writes `metadata/czechlynx-unseen-eval/metadata_unseen_eval.csv` and
`unseen_eval_manifest.json` under `<data_root>/CzechLynx_v2/`. It fails if the files exist; pass
`--force` to rebuild. The output is byte-identical to the paper's (SHA-256 in section 6).

### 5.4 SalamanderID2025

SalamanderID2025 is part of the AnimalCLEF2025 competition data, so there is no automatic
download. Accept the competition rules on Kaggle first, then:

```bash
kaggle competitions download -c animal-clef-2025
unzip animal-clef-2025.zip -d <animal-clef-2025 folder>      # holds metadata.csv and images/
```

Build the entry from that folder:

```bash
# 1. Copies the salamander photos into <data_root>/SalamanderID2025/{database,query}/images/
#    and writes split_time_closed.csv (CPU)
wildmatch prepare build salamander --source <animal-clef-2025 folder>
#    ...it prints the SAM 3 command; run it as in section 4 (GPU)

# 2. Only if SAM 3 reports empty masks
wildmatch prepare retry-empty salamander

# 3. split_time_closed_no_background.csv, the metadata the registry reads
wildmatch prepare finish salamander
wildmatch prepare status --dataset salamander
```

**The split.** Only labelled, dated photos are used (four undated ones are dropped). For every
salamander photographed on two or more dates, all photos of its latest date become `query`, the
rest `database`. The table rebuilds byte for byte (section 6).

**The masks.** SAM 3 is prompted with "Salamander" and **all** detections are merged, because a
handler's finger often splits one animal into several pieces. One image
(`query/images/9d1fc96e28c0058e_1277.jpg`) uses a lower threshold of 0.10. The masked images go to
`masked_images/`, the masks to `masks.csv`.

### 5.5 JaguarReID (paused)

> **Do not prepare or use this dataset for now.** The Jaguar Re-ID competition rules allow
> competition use only. Research use, publications, training models outside the competition and
> derived datasets need written authorization from the competition sponsors. That authorization
> has been requested; until it is granted, this entry must not be run.

For reference once research use is authorized: the entry reads the labelled training photos of
the competition (`train.csv` and `train/train/`) from `<data_root>/jaguar/` and prepares them in
three CPU steps,

```bash
wildmatch prepare jaguar prepare    # masked images (from the PNG alpha channel) + base table
wildmatch prepare jaguar embed      # DINOv2-small embeddings used to find photo bursts
wildmatch prepare jaguar split      # burst-aware database/query split: jaguar_reid_v2_no_background.csv
```

Unlike the other steps, these overwrite existing outputs without asking.

## 6. Checking the result

`wildmatch prepare status` should report every entry you prepared as ready. The tables below give
the expected sizes and, for files that rebuild exactly, their SHA-256 (`sha256sum <file>`).

**Expected sizes** (rows of the metadata file, identities by side):

| Key | Split column (database / query) | Database images | Query images | Database identities | Query identities |
|---|---|---:|---:|---:|---:|
| `atrw` | `split` (`train` / `test`) | 4,340 | 1,075 | 182 | 182 |
| `beluga` | `split` (`train` / `test`) | 4,625 | 1,277 | 788 | 633 |
| `cowdataset` | `split` (`train` / `test`) | 1,188 | 297 | 13 | 13 |
| `giraffes` | `split` (`train` / `test`) | 1,105 | 263 | 178 | 178 |
| `giraffezebraid` | `split` (`train` / `test`) | 5,398 | 1,500 | 2,051 | 1,142 |
| `hyenaid2022` | `split` (`train` / `test`) | 2,499 | 630 | 256 | 256 |
| `leopardid2022` | `split` (`train` / `test`) | 5,373 | 1,433 | 430 | 380 |
| `nyala` | `split` (`train` / `test`) | 1,514 | 428 | 237 | 236 |
| `seastarreid2023` | `split` (`train` / `test`) | 1,759 | 428 | 95 | 95 |
| `stripespotter` | `split` (`train` / `test`) | 656 | 164 | 45 | 44 |
| `whaleshark` | `split` (`train` / `test`) | 6,108 | 1,585 | 543 | 512 |
| `zindi` | `split` (`train` / `test`) | 9,721 | 3,082 | 2,265 | 2,251 |
| `czechlynx_closed` | `split-time_closed` (`train` / `test`) | 27,836 | 11,924 | 319 | 319 |
| `czechlynx_open` | `split-time_open` (`train` / `test`) | 27,587 | 12,173 | 275 | 126 |
| `czechlynx_unseen_eval` | `unseen_eval_split` (`database` / `query`) | 160 | 2,081 | 44 | 44 |
| `salamander` | `split` (`database` / `query`) | 1,138 | 246 | 584 | 221 |
| `jaguar` | `split_v2` (`database` / `query`) | 1,408 | 487 | 31 | 31 |

**Files that rebuild exactly:**

| File (under its dataset root) | SHA-256 |
|---|---|
| `WildlifeReID-10k/metadata.csv` (the Kaggle release, as a source check) | `da83480cae34bbd3ec275461b560ce9343fe88b526844afbc88857d7c751d566` |
| `wildmatch_prepare/ATRW_split.csv` | `51c736ff77970797de9d99ecea8262cf0bffbe173ce1b31c9e3e2eb20cc4d350` |
| `wildmatch_prepare/BelugaID_split.csv` | `31924b7cb3fb1bf5758f85ac76b9dbd28d8ab04b7abfc245396c092052170cdd` |
| `wildmatch_prepare/CowDataset_split.csv` | `76b5277395dd38cfa680248f3031e1123535fa603318b4ed6b1d1fee92267ebf` |
| `wildmatch_prepare/Giraffes_split.csv` | `ffb9cbac170e5d519d7e002ab598b058870de8dc7b3cb43e8eebe7f90d19134e` |
| `wildmatch_prepare/GiraffeZebraID_split.csv` | `20890e3bf1deff0e65f5b7696b46922bb58da5e5febf9e52a8c8f81b87a046dd` |
| `wildmatch_prepare/HyenaID2022_split.csv` | `33d8503434d01fcf2dbf2e8fb3eb9059569258ca77d74a4539facef27620f600` |
| `wildmatch_prepare/LeopardID2022_split.csv` | `e428cb83911a64f8e6cdf58cf50dde2ae106664e03257b03bcc6844b8326ab76` |
| `wildmatch_prepare/NyalaData_split.csv` | `ab047d2c89da4dcb00e517bffd70dbddb83bb67edbe9e6cf11721c522b9df7ce` |
| `wildmatch_prepare/SeaStarReID2023_split.csv` | `d0a80022f10bca8e276b9ef21302c783b415528533eb2c3d3447693bba6d1dd4` |
| `wildmatch_prepare/StripeSpotter_split.csv` | `074bd61caef62ecf4f8567e0d0a232e1c0f9ed8cddb1512895b5fe79e25f0945` |
| `wildmatch_prepare/WhaleSharkID_split.csv` | `372cf4e659c93bf142fd5723864fb5a6bf66e9f79210d7216b5f4a513652e0d5` |
| `wildmatch_prepare/ZindiTurtleRecall_split.csv` | `0293d9a15934ae798142b4ecbee0bb3fd1c114b653e1a0152a141a2fea084fac` |
| `SalamanderID2025/split_time_closed.csv` | `0b832fc81d651c3439f421c2108ab85ec430f7714fb60c66b6afbc4bdf1744f6` |
| `CzechLynx_v2/metadata/czechlynx-unseen-eval/metadata_unseen_eval.csv` | `eef513b5dd328bcf095eb772f92e662dd382d985d416026a9744af5bc3158b5a` |

The `wildmatch_prepare/` files are under `WildlifeReID-10k/`. The metadata written by `finish`
(`metadata_sam3/...`, `split_time_closed_no_background.csv`) holds the SAM 3 masks, which can
vary slightly with the GPU and the SAM 3 version, so check those by their sizes above, not by
checksum.

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `kaggle` reports missing credentials, or `401` / `403` | Put your API token in `~/.kaggle/kaggle.json` (or export `KAGGLE_USERNAME` and `KAGGLE_KEY`). For `animal-clef-2025`, accept the competition rules on Kaggle first. |
| `no automatic download for salamander` (or `jaguar`) | Expected: competition data is downloaded by hand (5.4). |
| `SAM 3 location unknown: pass --sam3-dir and --checkpoint ...` | Set `WILDMATCH_SAM3_REPO` and `WILDMATCH_SAM3_CHECKPOINT` (section 4, step 4). |
| `sam3` cannot be imported although SAM 3 is installed | The command ran with the `wildmatch` environment's Python. Use `$SAM3_PYTHON` and clear `VIRTUAL_ENV` and `PYTHONPATH` (section 4, step 5); the Slurm wrapper does this for you. |
| `no kernel image is available for execution on the device` | The GPU is too old for the PyTorch build (for example a V100). Use a newer GPU. |
| `... already exists; pass --overwrite or choose --output-dir` | Nothing is replaced by accident. Pass `--overwrite` to rebuild in place, or `--output-dir <dir>` to try elsewhere. |
| Many empty masks after the first SAM 3 run | Expected for some animals (top-down whale backs, full-frame close-ups). Run `wildmatch prepare retry-empty <key>`; what is still empty keeps the whole photo. |
| `status` still says `metadata missing` after masking | `finish` has not run yet. |
