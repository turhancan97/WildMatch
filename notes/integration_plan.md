# Integration plan (draft): mining and matcher fine-tuning into `wildmatch`

Status: **draft, not started** (survey 2026-10-05, read-only; nothing in either sibling repository
was changed by the survey). **Reference (user decision 2026-10-05): the `feat/wildlife-reid-pipeline`
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
- ~~Commit the uncommitted lynx-finetuning work~~ done 2026-10-05 (`319477e`). Still to do: commit
  `czechlynx_protocol.sh` to the reference branch.
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

Effort estimate: phases 1-3 a few days of mostly mechanical work; phases 4-5 dominate (GPU parity
runs on the cluster). Nothing here affects the submitted paper.
