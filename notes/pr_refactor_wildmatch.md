# PR draft: `refactor/wildmatch-package` into `main`

Draft written 2026-10-04; open the pull request only on the user's instruction. The repository is
private (review embargo until 2026-12-07); nothing in this PR deploys the page or publishes weights.

## Summary

Turns the paper code (tag `paper-v1`, commit `4bad498`) into the installable `wildmatch` package,
run through one `wildmatch` command, without changing results: every execution-changing phase
was checked against reference runs and reproduced them bit for bit on the same GPU type.

- **Install:** `uv sync --extra cu126 --extra matchers --group dev` (or `--extra cpu`), Python
  3.12, locked in `uv.lock`; a conda route installs the same pins (`environment.yml`,
  `requirements/*.txt`).
- **Layout:** `src/wildmatch/` (data, evaluate, matchers, models, train, reporting, sweep,
  utils, conf), `paper/` (page exporters, paper figures, tools), `slurm/`, `tests/`.
- **Commands:** `wildmatch evaluate | finetune-backbone | sweep | sweep-task | prepare | weights |
  demo | tables | figures | summarize-runs | summarize-logs | class-balance | audit |
  build-unseen-split`.
- **Portable paths:** profiles `default` (relative `./data`, `./cache`, `./checkpoints`, environment
  overrides) and `gmum` (the cluster layout, so cluster paths and cache keys are unchanged).
- **Dataset registry:** one Hydra config per dataset/split (`conf/dataset/<key>.yaml`, 17 entries)
  with checkpoints, data sources and preparation rules; `PAPER_PROFILES` is built from it.
- **Sweeps:** `wildmatch sweep <spec>` (YAML: datasets, budgets, method rows) replaces the two bash
  launchers, with the same immutable submissions (frozen config tree, checkpoint SHA-256 and owner
  checks) and per-task logs; `--list-tasks`, `--dry-run`, `--submit` (Slurm array), `--local`.
- **Data preparation:** `wildmatch prepare status | download | build | finish | compare-masks |
  retry-empty | jaguar | unseen-split`. Recovered and implemented the rules behind the prepared
  tables: WildlifeReID-10k closed-set split (WildFusion's, `ClosedSetSplit(0.8, seed=666)`; all
  twelve tables reproduced), SalamanderID2025 time split (byte-identical; matches the original
  `results/make_salamander_split.py` of the paper repository), CzechLynx unseen-identity split
  (byte-identical).
- **Weights:** `wildmatch weights list | verify | download | stage` for the 18 matcher-only paper
  checkpoints (located by the SHA-256 the paper's 395 runs recorded) through the Hugging Face Hub.
- **Quality:** ruff (one formatting-only commit, AST-identical; one lint-fix commit), GitHub
  Actions CI (lint, format, CPU tests, strict page build; rehearsed on a clean export), Apache-2.0
  `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md`, README quickstart, `wildmatch demo` (24 bundled
  synthetic lynx renders, CPU, about 1 minute).

## Evidence that results did not change

Reference: SalamanderID2025, k=50, seven methods (cosine, WildFusion, default and fine-tuned LoMa
and RDD-LightGlue, frozen weighted linear probe), `notes/parity_reference.md`.

| Phase | Check | Result |
| --- | --- | --- |
| 1 Packaging, environment | runs 1-2 (arrays 524163, 524170) and two controls | bit-identical to `paper-v1` on the same GPU type |
| 2 Paths, registry | run 3 (array 524188) | 7/7 bit-identical to run 2, cache keys unchanged |
| 3 CLI, sweeps | run 4 (array 524200) | 7/7 bit-identical to run 2, cache fingerprints unchanged |
| 3 Sweeps | task tables, manifests, checkpoint hashes, Hydra overrides | equal to the bash launchers' |
| 3 Reports | `wildmatch tables`, `wildmatch figures` on the 669 existing runs | tables byte-identical, figures pixel-identical |
| 5 Formatting | every reformatted file | same syntax tree |

Old `experiments/` runs stay readable (641 records identical to `paper-v1`'s reader; a fixture
pins a pre-refactor manifest).

## Behaviour changes

- WildlifeReID-10k entries read new SAM 3 masked inputs (`metadata_sam3/`, `masked_images_sam3/`)
  by default; the team's masked files the paper used stay untouched and are reachable as
  `registry.paper_inputs` and `inputs: paper` in sweep specs. On the six paper datasets at k=250
  (sweep `wildlife_sam3`, array 524372, all 42 runs) results stay close: Top-1 changes by a mean
  of +0.73 points (median +0.5, range -2.0 to +3.7; 27 of 42 within one point, 14 higher, one
  lower), balanced Top-1 by +0.52; Sea star gains about 2.4 (old masks were specks on close-ups),
  Zindi WildFusion loses 2.0 (near-tie flips, masks unchanged). Table: `reports/sam3_vs_paper.csv`
  (`python paper/tools/compare_with_paper.py --job 524372`).
- Registry checkpoint paths point at the files the paper's runs used (renamed folders, Nyala's
  `model__actual_nyala.safetensors`, the relaxed RDD checkpoints for CzechLynx closed and Nyala,
  joint epoch 100).
- Sweep task logs always include a split folder; the launchers' checkpoint environment variables
  are replaced by `dataset_overrides` in the spec.
- `train/`, `scripts/`, `probe.sh`, `finetune.sh` and the bash launchers are gone (all in
  `paper-v1`); Slurm wrappers live in `slurm/`.

## Bugs found and fixed on the way

- Launchers passed the frozen config with `--config-dir`, which Hydra ignores as primary config
  (snapshots were never used); fixed on both branches (`--config-path`).
- `finetune_runner.py` used `file_identity` without importing it since 2026-08-13 (every backbone
  fine-tuning run would crash after training; none existed); fixed on both branches.
- `wildmatch prepare` hardening found during the SAM 3 runs: concurrent folder creation on the
  shared filesystem, the SAM 3 Python resolved from the submitting shell, a misplaced registry key.

## Merging

`main` has three commits since the branch point (`d676d40` config-path fix, `9ffb767` parity
notes, `eb13881` file_identity fix). The branch contains equivalent changes, but `reid/` moved to
`src/wildmatch/`, so expect conflicts in the launchers, `reid/engine/finetune_runner.py`,
AGENTS.md and CHANGELOG.MD; resolve them in favour of the branch.

## Open items (not blocking the merge)

- Before the public release: GPU type in the Vismatch cache key and run manifest; checkpoint
  fingerprints without absolute paths (both invalidate caches once).
- Hugging Face repository for the weights (name to decide; upload with `wildmatch weights stage`).
- Licences: default LoMa/RDD weights, the release licence of the WildMatch checkpoints, datasets.
- Committed page exports under `docs/` still say `generated_by: scripts/...` until rerun.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01SMdxjbXRiNQq2NGr54XVd7
