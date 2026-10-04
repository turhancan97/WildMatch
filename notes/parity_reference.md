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

Salamander reference runs (fill in after the array finishes):

| Task | Run |
| --- | --- |
| 0 cosine | |
| 1 wildfusion | |
| 2 loma default | |
| 3 loma fine-tuned | |
| 4 rdd default | |
| 5 rdd fine-tuned | |
| 6 linear probe | |
