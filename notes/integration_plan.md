# Integration plan: mining and matcher fine-tuning into `wildmatch`

Status: **planned, not started**; decisions D1-D8 in section 5 (user, 2026-10-05). The survey
(2026-10-05) was read-only: nothing in either sibling repository was changed. **Reference (user decision 2026-10-05): the `feat/wildlife-reid-pipeline`
branch of both repositories, as pushed by the user that day** (mining `4f29292`, fine-tuning
`319477e`); the adaptation starts from those commits. Piotr's `piotr-wip` branches (and mining's
`piotr/czechlynx`) are not part of the reference. Fills the slots agreed in the refactor plan
(`notes/refactor_plan.md`, "Future merge"): `src/wildmatch/mining/` from `rdd-parallel-benchmark`
and `src/wildmatch/matcher_finetune/` from `lynx-finetuning`, merged with history.

Data flow today: `rdd-parallel-benchmark` mines triplet indices -> `lynx-finetuning` trains matchers
on them -> checkpoints listed in `src/wildmatch/conf/weights.yaml` -> `wildmatch evaluate`.

## 1. What is there

| | rdd-parallel-benchmark (mining) | lynx-finetuning (fine-tuning) |
| --- | --- | --- |
| Remote | GitHub `PiotrKubaty/rdd-parallel-benchmark` | GitHub `PiotrKubaty/lynx-finetuning` |
| Reference branch | `feat/wildlife-reid-pipeline` `4f29292` (2026-10-05; all commits by Turhan) | `feat/wildlife-reid-pipeline` `319477e` (2026-10-05; all commits by Turhan; 39 ahead of `main`) |
| Not in the reference | `origin/...-piotr-wip` eb04345 (+1: `MIGRATION.md`, env vars for paths, pip freezes, few-shot scripts, 4 tests); `origin/piotr/czechlynx` (+4) | `origin/...-piotr-wip` 20b3cc1 (+1: `env.example.sh`, `slurm_scripts/czechlynx_protocol.sh`) |
| Uncommitted | 2 notebooks only | none (the relaxed objective, joint presets, resume, keep-every were committed in `319477e`) |
| Python | 55 files, ~9.6k lines, incl. vendored `RDD/` (27 files, Apache-2.0, xtcpete/rdd) | 28 files, ~11.1k lines, incl. `rdd_patch/` (modified LightGlue, Apache-2.0) |
| Tests | 7 (+1 untracked), CPU, pytest | 10 files, ~60 tests, partly CPU, partly CUDA + weights |
| Environment | conda `rdd` (py3.10) / `loma` (py3.12), torch 2.13 + cu130, unpinned | same envs; `rdd` checkout via symlink (`../rdd`, HEAD 86f0e38 with local edits), LoMa via pip (unpinned) |
| Licence | vendored RDD Apache-2.0; repo itself: none stated | **no LICENSE** |

Pipeline (mining): `*_dataset.py` builds symlinked split views -> `lynx_build_cache.py` /
`lynx_build_loma_cache.py` write per-frame `.npz` features -> `*_mine.py` (one query collection per
Slurm array task, masked LightGlue or LoMa against the train gallery) -> `*_aggregate.py` writes
`strong-matches_<split>_combined.json` (`[{query_frame, positives, negatives}]`) -> `*_evaluate.py`.

Pipeline (fine-tuning): `train_by_lg_matches.py` (RDD-LightGlue, 3,008 lines) and
`train_loma_matches.py` (LoMa) read those indices; margin-0.5 hinge on the relaxed score
`s = (mean_i max_j P_ij + mean_j max_i P_ij)/2` (`relaxed_v1`), AdamW 1e-5, cosine over 300 epochs;
joint presets train descriptor + matcher. Outputs: accelerate `epoch_NNN/` (RDD) or LoMa bundles,
plus `czechlynx_protocol.json` / `wildlife_protocol.json`, under
`/shared/sets/datasets/vision/czechlynx/checkpoints/...` - the layout
`src/wildmatch/matchers/vismatch_checkpoints.py` already reads.

## 2. Findings that shape the plan

1. **Code behind the paper's relaxed and joint checkpoints is now committed** (`319477e`,
   2026-10-05; before that it existed only as uncommitted changes in the user's clone). One file
   is still missing from the reference branch: `slurm_scripts/czechlynx_protocol.sh` (66 lines, the
   legacy/strict protocol resolver) is on disk but matched by the repository's `.gitignore`;
   the four CzechLynx cache and training scripts source it and `tests/test_czechlynx_protocol.py`
   tests it, so a fresh clone of the reference cannot run CzechLynx training. It exists in git only
   on `piotr-wip` (20b3cc1). Add it to the reference (`git add -f`) or carry it in from disk.
   The `rdd` checkout that fine-tuning imports (`../rdd`, GitHub `turhancan97/rdd`, branch
   `lynx_analysis`, 86f0e38) has two uncommitted edits in `RDD/`: absolute paths for the config and
   LightGlue weights, and `torch.cuda.amp.custom_fwd` -> `torch.amp.custom_fwd(device_type="cuda")`
   (no behaviour change).
2. **Most checkpoints cannot be tied to a commit.** None records a git commit; only LoMa
   `metadata.json` records a seed. The ~2026-09-05 wildlife RDD checkpoints used the earlier
   filtered/Adam recipe, which the current code no longer reproduces (README); the commit is
   likely <= 742b170 but unconfirmed. Several checkpoint folders were renamed after training.
   Parity for those can only be "evaluation of the published file is unchanged", not "retraining
   reproduces it".
3. **Two different RDD copies.** Mining vendors `RDD/` (stripped down, plus `lightglue_masked.py`);
   fine-tuning imports the `rdd` checkout. They differ in `RDD.py`, `models/backbone.py`,
   `matchers/lightglue.py`, `matchers/__init__.py` and `utils/misc.py`; the checkout also has
   `dense_matcher.py`, `dual_softmax_matcher.py`, `RDD_helper.py`, `dataset/` and a compiled
   deformable-attention op for Python 3.10. A third LightGlue variant is `rdd_patch/` in
   fine-tuning, and `wildmatch` evaluates through vismatch's own RDD. The package needs one
   decision on which RDD code mining and training use, checked against the indices and checkpoints
   they produced.
4. **Packaging blockers:** top-level `scripts.` package (mining) and `import rdd.RDD...` + CWD-relative
   `rdd/configs`, `rdd/weights` paths (fine-tuning) do not work inside a package; ~80 env vars and
   hard-coded `/shared/...`, `/home/kargin/...` paths; lynx-finetuning's `.gitignore` ignores
   `*.md *.sh *.json *.yaml ...` (would silently hide files under the new prefix);
   `czechlynx_protocol.sh` is required but untracked.
5. **Environment conflict:** sibling envs are torch 2.13/cu130 (py3.10 for `rdd`), `wildmatch` pins
   torch 2.8.0/cu126, py3.12. Mining and training have never run on the `wildmatch` env. Missing
   from `wildmatch`: `accelerate`, `lomatch` (LoMa), `ninja`; `hloc` is listed but apparently unused.
6. **Duplicated logic that must stay bit-identical:** the trainers' `resize_long_side` (/32 RDD,
   /14 LoMa) vs `wildmatch.matchers.vismatch_preprocessing.resize_long_side_divisible`
   (`lynx_finetuning_v1`, `lynx_loma_finetuning_v1`); checkpoint key names (`model_1.safetensors`,
   `*_train_component`, protocol JSON fields) read by `vismatch_checkpoints.py`; mining's RDD/LoMa
   scoring vs vismatch's. The mining configs (`configs/wildlife/*.json`) are a second dataset registry.
7. **History size:** the mining repo's history carries ~30 MB of deleted upstream RDD assets
   (`paper.pdf`, `supp.pdf`, images). Filter them before the subtree (or `--squash`).
8. **No secrets found;** personal e-mail addresses appear in commit history (normal for history merges).

## 3. Proposed phases (each reviewed before the next, like the refactor)

**Phase 0 - preserve and decide (no code moves).**
- ~~Commit the uncommitted lynx-finetuning work~~ done 2026-10-05 (`319477e`); ~~commit
  `czechlynx_protocol.sh`~~ done 2026-10-05 (`6958d3d`).
- Licence for both repos (Apache-2.0 to match `wildmatch`?); attribution (Piotr's commits on other
  branches, the remote's owner).
- Decide the open questions in section 4.

**Phase 1 - import fine-tuning with history.** `git subtree add --prefix=vendor/lynx-finetuning`
from the agreed commit (staging prefix, so history arrives unchanged), then in separate commits
`git mv` the code into `src/wildmatch/matcher_finetune/`, Slurm wrappers into `slurm/`, docs into
`docs/` or `notes/`; drop the repo-level `.gitignore`, `logs/` placeholders and debug scripts that
are not wanted. Add `rdd_patch` to `THIRD_PARTY_LICENSES.md`/`NOTICE`.

**Phase 2 - make it a package module.** Replace `rdd` symlink imports (pinned git dependency or
vendored, per decision); paths through `wildmatch.paths` / Hydra config instead of env vars and
absolute paths; resize helpers import `vismatch_preprocessing` (bit-identity test, run always, not
only with `RUN_VISMATCH_LYNX_PARITY=1`); a `wildmatch finetune-matcher` CLI; tests run in CI on CPU
(GPU tests optional).

**Phase 3 - import mining with history** (filtered history), same staging-prefix route into
`src/wildmatch/mining/`; `scripts.` -> `wildmatch.mining.`; datasets from the registry instead of
`configs/wildlife/*.json` (or a thin adapter first); `wildmatch mine` CLI; indices under
`paths.output_root` with `external.mining_outputs` still able to read the existing ones.

**Phase 4 - environment.** Either move mining and training onto the `wildmatch` env (torch 2.8;
needs proof that results do not change) or keep a separate `train` extra / env with its own lock.
Decide by a parity test, not by preference.

**Phase 5 - parity and provenance.**
- Evaluation of every checkpoint in `weights.yaml` unchanged (same scores within the existing
  tolerances in `notes/parity_reference.md`) - mandatory.
- Mining one small dataset (e.g. SalamanderID2025) gives identical `*_combined.json`.
- Fine-tuning a few epochs on that index matches the old code's losses step by step (same seed).
- From then on every checkpoint records the `wildmatch` commit, dirty state and seed (the
  manifest code-identity block from `fix/manifest-code-identity` already exists for runs).

## 4. Open questions

1. ~~Canonical source per repo~~ decided 2026-10-05: `feat/wildlife-reid-pipeline` as pushed
   (`4f29292`, `319477e`). Still open: is anything from `piotr-wip` wanted (its env-var path layer,
   few-shot scripts, `czechlynx_protocol.sh`)? Is `319477e` exactly the code of the 2026-09-29
   relaxed runs and the joint runs?
2. Licence of both repositories (none stated) and attribution in `wildmatch`.
3. Scope: only the wildlife and CzechLynx pipelines, or also the confidential-lynx `lynx_*`
   scripts, `debug/`, `helios_scripts/`, notebooks, older training strategies?
4. RDD and LoMa: keep vendored `RDD/` + own backends, pin `rdd` (86f0e38 has local edits) and LoMa
   (which commit/branch, "train-dev"?) as git dependencies, or go through vismatch? If mining
   switches implementation, must the mined indices stay bit-identical?
5. One environment or a separate training environment (section 3, phase 4)?
6. Replace the mining JSON configs by the `wildmatch` dataset registry?
7. Existing indices and checkpoints: stay where they are (read through `external.*` and
   `checkpoint_root`), or move under the `wildmatch` path layout?
8. Which commit produced the ~2026-09-05 wildlife checkpoints and Nyala's
   `model__actual_nyala.safetensors`?

Questions 1-5 were answered on 2026-10-05 (section 5). Still open: 6-8, and whether `319477e` is
exactly the code of the relaxed and joint runs (checked by parity level L5, not by asking).

## 5. Decisions and design (user, 2026-10-05)

Section 3 gives the order of the phases; this section takes precedence where they differ.

**Decisions.**

| # | Question | Decision |
| --- | --- | --- |
| D1 | Merge mechanics | `git filter-repo` on a throwaway clone of each reference branch: move paths to their final place (`src/wildmatch/{mining,matcher_finetune}/`, `slurm/`, `tests/`, `notes/`), remove the paths that stay behind and the ~30 MB of deleted RDD assets from history, then `git merge --allow-unrelated-histories` into a phase branch. This replaces `git subtree` + staging prefix (same goal: history preserved, and `git blame`/`log` work on the new paths). The sibling repositories and their remotes are not touched. |
| D2 | `czechlynx_protocol.sh` | The user commits it to lynx-finetuning `feat/wildlife-reid-pipeline` (`git add -f`). Done 2026-10-05: `6958d3d` (on top of `319477e`; file SHA-256 `1d047ac9...`, the on-disk version the training scripts source, not piotr-wip's older one; its 7 tests pass). **The fine-tuning reference is now `6958d3d`.** |
| D3 | Environment | A spike first (about one day): mining and ~50 training steps in the `wildmatch` uv env (torch 2.8/cu126) against the old conda envs (torch 2.13/cu130) on the same GPU type, measured with parity levels L3-L5. If it fails, a separate, locked training environment. Decided by the measurement. |
| D4 | Stays behind | The confidential-lynx `lynx_*` pipeline (except the three helpers that are still imported: `FrameFeat`, `load_cached_feat`, `sample_frames`), mining's `*_evaluate.py` (duplicates `wildmatch evaluate`), `helios_scripts/` and `debug/`, and the older training strategies (the pre-relaxed/filtered recipe paths). They stay readable in the sibling repositories' history. |
| D5 | From `piotr-wip` | **Only the few-shot scripts** (mining `eb04345`: `scripts/wildlife_fewshot.py`, `scripts/czechlynx_fewshot.py`, `slurm_scripts/fewshot/*`, `tests/test_wildlife_fewshot.py`). The env-var path layer, `MIGRATION.md`, the pip freezes and the migration pack scripts are not taken. See the caveats below. |
| D6 | Licence | (Premise corrected 2026-10-05: Piotr wrote 30 of the 48 fine-tuning commits and ~52 % of its surviving lines, plus 158 of the moved mining lines, so his consent covers the fine-tuning code as a whole.) Apache-2.0, like `wildmatch`, with `NOTICE`/`THIRD_PARTY_LICENSES.md` entries for the vendored RDD (xtcpete/rdd) and `rdd_patch` (modified cvg/LightGlue), both Apache-2.0. Piotr is credited as the owner of the remotes and as co-developer. |
| D7 | RDD and LoMa | Vendor the RDD checkout that training used (`turhancan97/rdd`, `lynx_analysis`, `86f0e38`, with its two harmless local edits) once inside the package, and check mining's results against it (mining's own `RDD/` copy differs in five files). LoMa becomes a git dependency pinned to the commit that training used. Installed on 2026-10-05: `lomatch @ 5e541b8` in the `loma` env (pip from git), and an editable checkout at `7043bac` in the `rdd` env. The commit is chosen by parity L5, not assumed. |
| D8 | This plan | Written here, then `main` pushed after the user's review. |

**Caveats found while recording D5 and D6.** The few-shot code sits in one commit by Piotr Kubaty
(`eb04345 migrate`, 2026-09-22, 79 files) and is not the user's. (1) It is mixed with changes to
shared files (`wildlife_mine.py`, `czechlynx_mine.py`, chunked mining, `.gitignore`, the Slurm
wrappers), and it branched from `1afcb73` (2026-09-08), before the reference `4f29292`. So the
few-shot files have to be ported onto the reference by hand and may depend on its mining changes.
This is a port, not a cherry-pick, and comes in a phase of its own after mining. (2) Under D6 the
reference code is all the user's, but these files are Piotr's. Before they are relicensed as
Apache-2.0 Piotr has to agree (or the few-shot part is left out).

**Design.**
- **Scope moved:** fine-tuning trainers (`train_by_lg_matches.py`, `train_loma_matches.py`),
  `train_common`, `loma_backend`, loading, caches, `rdd_patch/`, tests, the WILDLIFE/CZECHLYNX docs;
  mining's dataset view, cache, mine and aggregate scripts plus the three `lynx_benchmark.py` helpers.
- **Formats stay byte-compatible:** the mining view layout and the index JSON
  (`strong-matches_<split>_combined.json`, with view-relative paths that the paper checkpoints were
  trained on), the checkpoint folder layout under `checkpoint_root`, and the protocol-JSON keys read
  by `src/wildmatch/matchers/vismatch_checkpoints.py`. Provenance fields (commit, dirty, seed) are
  only added, after checking that the loader accepts extra protocol keys. Existing indices and
  checkpoints stay where they are.
- **Configuration and CLI:** Hydra groups `conf/matcher_finetune/{rdd,loma}.yaml` (presets
  `matcher`, `descriptor`, `joint`; relaxed objective by default) and `conf/mining/{rdd,loma}.yaml`;
  `wildmatch mine <view|cache|run|aggregate>` and `wildmatch finetune-matcher`; Slurm scripts in
  `slurm/`, with mining arrays using the sweep's frozen-submission pattern. Datasets come from the
  registry. Generated views are tested against the existing ones before the mining JSON configs are
  removed.
- **Resize helpers** import `wildmatch.matchers.vismatch_preprocessing`, with a bit-identity test
  that always runs.

**Parity ladder** (old code in the old env vs new code, on the **same GPU type**; Vismatch features
differ between H100 and RTX 4090, see AGENTS.md "Known issues"):

| Level | Check |
| --- | --- |
| L0 | The sibling repositories' tests pass inside `wildmatch` on CPU. |
| L1 | Training resize == `resize_long_side_divisible`, bit for bit (always run). |
| L2 | Registry-built views == the existing views for the eight paper datasets. |
| L3 | Extracted mining features == the old caches. |
| L4 | SalamanderID2025 LoMa mining gives an identical `strong-matches_*_combined.json`. RDD mining only matters for Salamander, WhaleShark and Nyala (the rest are LoMa-mined). |
| L5 | ~50 training steps at seed 0 give the same loss trace, and tensors within tolerance (this also pins the LoMa commit, D7). |
| L6 | **Mandatory:** every checkpoint in `conf/weights.yaml` evaluates unchanged (existing sweep machinery, tolerances in `notes/parity_reference.md`). |

**Process.** One branch per phase, in a worktree on `/shared` (home has a 61,440 MB quota; keep
about 1 GB free). Never edit the main checkout while Slurm sweeps run from it. Training outputs,
caches and W&B stay off home. Never touch the Hub checkpoints (`turhancan97/wildmatch-checkpoints`)
or the paper snapshot. The paper is frozen; nothing here changes it.

**Phase order (revised).** 0: ~~D2 commit~~ (done, `6958d3d`); Piotr's consent for the few-shot files.
1: fine-tuning via filter-repo + merge. 2: make it a package module (vendored RDD, pinned LoMa,
paths, CLI). 3: mining, same route. 4: environment spike (D3). 5: parity L0-L6 and provenance.
6: few-shot port (if Piotr agrees).

Effort estimate: phases 1-3 a few days of mostly mechanical work; phases 4-5 dominate (GPU parity
runs on the cluster). Nothing here affects the submitted paper.
