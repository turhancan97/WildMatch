# Parity reference for the `wildmatch` package refactor

The refactor on branch `refactor/wildmatch-package` must not change any result. Every phase
that touches execution reruns the reference set below from the branch and compares it with
these reference runs, made from the tagged paper code.

- Reference code: tag `paper-v1` (commit `4bad498`), environment `ex-reid`.
- Comparison: `scores.npz` and `metrics.json` of each pair of runs. Cosine, WildFusion and
  RDD-LightGlue scores within 1e-4 and identical Top-1 lists; LoMa within 1e-2 (it matches
  under bfloat16 autocast); the linear probe within seed noise.

## Reference set

SalamanderID2025 (1,138 database / 246 query photos, fine-tuned checkpoints available),
`candidate_k=50`, seven tasks:

| Task | Method | Checkpoint |
| --- | --- | --- |
| 0 | cosine | default |
| 1 | wildfusion | default |
| 2 | vismatch / loma | default |
| 3 | vismatch / loma | fine-tuned, matcher only (`legacy-loma-mined/epoch_299`) |
| 4 | vismatch / rdd-lightglue | default |
| 5 | vismatch / rdd-lightglue | fine-tuned, matcher only (`legacy-rdd-mined/epoch_299`) |
| 6 | linear_probe | frozen backbone, class-weighted |

JaguarReID `split_v2`, `candidate_k=10`, already run on 2026-10-03 with the same code:

| Method | Run |
| --- | --- |
| cosine | `20261003T221722Z_0f6830d2` |
| wildfusion | `20261003T221724Z_609d8fd8` |
| vismatch / loma (default) | `20261003T221724Z_ae12917c` |
| vismatch / rdd-lightglue (default) | `20261003T221724Z_b3daf60f` |

## How the Salamander reference was submitted

From the repository root on `main` at `paper-v1`, with a temporary copy of
`probe-parallel-wildlife.sh` (`probe-parallel-parity.sh`, untracked) whose tables were set to:
only the `salamander` profile active, `CANDIDATE_K_VALUES=(50)`, and the seven `VARIANTS` rows
above (`cosine`, `wildfusion`, `vismatch|loma|default`, `vismatch|loma|custom|${LOMA_CUSTOM_CHECKPOINT_PATH}|matcher_only`,
`vismatch|rdd-lightglue|default`, `vismatch|rdd-lightglue|custom|${RDD_CUSTOM_CHECKPOINT_PATH}|matcher_only`,
`linear_probe|-|default|-|-|classifier|weighted`):

    bash probe-parallel-parity.sh --list-tasks   # 7 tasks
    bash probe-parallel-parity.sh                # submits; the array runs from the frozen manifest
    rm probe-parallel-parity.sh                  # not needed after submission

Salamander reference runs (array 524155, 2026-10-04, code at `69fc403` = `paper-v1` code, `ex-reid`):

| Task | Run | Top-1 | Top-5 | Balanced Top-1 |
| --- | --- | --- | --- | --- |
| 0 cosine | `20261004T075043Z_6032a6f6` | 3.66 | 9.35 | 3.62 |
| 1 wildfusion | `20261004T075043Z_a4ae34a5` | 26.83 | 27.24 | 27.68 |
| 2 loma default | `20261004T075035Z_fc961e4d` | 30.89 | 31.71 | 32.20 |
| 3 loma fine-tuned | `20261004T075035Z_57276f29` | 32.11 | 32.11 | 33.56 |
| 4 rdd default | `20261004T075035Z_7d93fc25` | 27.64 | 27.64 | 28.58 |
| 5 rdd fine-tuned | `20261004T075035Z_41712b9f` | 28.05 | 28.86 | 29.03 |
| 6 linear probe (frozen, weighted) | `20261004T075044Z_3b9240fb` | 1.22 | 2.03 | 1.06 |

The refactor's parity runs use a fresh feature cache (`parity_v2`), so feature extraction is
re-run in the new environment instead of reusing these runs' cached features.

## Parity results on `refactor/wildmatch-package`

| Run | Array | Scope | Result |
| --- | --- | --- | --- |
| 1 (Phase 1) | 524163 | features from the shared cache (the launcher still passed `--config-dir`, so the `parity_v2` snapshot was ignored) | 7/7 pairs bit-identical (cosine, WildFusion, default and fine-tuned LoMa and RDD-LightGlue, frozen weighted linear probe), no Top-1 change |
| 2 (Phase 1) | 524170 | fixed launcher (`--config-path`), fresh `parity_v2` cache: feature extraction in the new environment | cosine (8.6e-07) and WildFusion (9.2e-06) pass with no Top-1 change; the four Vismatch pairs keep identical Top-1/Top-5/balanced Top-1 and mAP@k within 0.03 points, but pair scores drift (LoMa default 0.032 and 25 Top-1 photo changes, LoMa fine-tuned 0.50 and 96, RDD default 0.0075 and 1, RDD fine-tuned 0.015 and 0). The reference read Vismatch features cached on 2026-09-30, so two controls follow |
| control A | 524178 | `paper-v1` code in `ex-reid`, fresh cache (`parity_ref_fresh`), outputs in `experiments/parity-control` (`logs/parity/parity_control_main.sbatch` in the main checkout) | run 2 vs control A: 4/4 Vismatch pairs bit-identical (max score difference 0, no Top-1 change) |
| control B | 524179 | branch twin of run 2, fresh cache (`parity_v3`), outputs in `experiments/parity-twin` (`logs/parity/parity_control_branch.sbatch`) | control B vs run 2: 4/4 bit-identical (extraction is deterministic on one GPU type) |

Control A against the original reference shows the same drift as run 2, so the drift is not
caused by the refactor: the reference read Vismatch features that run `20260930T063532Z_9ecdc517`
(job 521339, `dgxh100`, H100) had cached, while every run on 2026-10-04 extracted on
`rtx4090_batch` (RTX 4090). Vismatch keypoint extraction depends on the GPU type.

**Verdict for Phase 1: parity holds.** With the same GPU type and cache state the branch
reproduces `paper-v1` bit for bit for all seven methods (run 1; run 2 against control A), and the
linear probe is bit-identical in run 2 as well. Parity runs must therefore fix the GPU type
(`rtx4090_batch`) and compare against a reference extracted on that GPU type: from Phase 2 on,
compare against run 2 (`20261004T0848*`, branch, fresh `parity_v2` cache on RTX 4090).
| 3 (Phase 2) | 524188 | paths and registry: `paths=gmum dataset=salamander`, config groups frozen in the snapshot, caches `parity_v2` (hits expected: unchanged cache keys), outputs in `experiments/parity-phase2`; compared with run 2 | 7/7 pairs bit-identical, no Top-1 change; every feature lookup hit the `parity_v2` cache (cache keys unchanged); every run recorded registry key `salamander` and the `gmum` data root |
| 4 (Phase 3) | 524200 | CLI and sweeps: `wildmatch sweep parity --submit --config logs/parity/probe_parity_phase3.yaml` from the worktree root (array element = `wildmatch sweep-task`, `python -m wildmatch evaluate`), caches `parity_v2` (hits expected), outputs in `experiments/parity-phase3`; compare with run 2 | 6/6 retrieval pairs bit-identical (max score difference 0, no Top-1 change); Vismatch cache fingerprint equal to run 2's (`5613207d...` for default LoMa), so cache keys are unchanged; all task records `completed`/`validated` with run directories; the linear probe (task 6) was still running on 2026-10-04 |
