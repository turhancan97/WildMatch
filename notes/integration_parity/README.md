# Integration parity runs (2026-10-05)

Scripts behind parity levels L2, L3 and L5 and the environment spike (decision D3) of
`notes/integration_plan.md`. They are kept as run; paths point at the cluster and at the integration
worktrees under `/shared/results/common/kargin/projects/wildmatch-integration/`.

| Level | Script | Job | Result |
| --- | --- | --- | --- |
| L2 views | `compare_views.py` | CPU | Rebuilt views equal the live training views for all eight paper datasets and CzechLynx open (every directory and symlink; only `manifest.json`'s `output_root` differs). |
| L3 features | `spike_features.sbatch`, `compare_features.py`, `match_features.py` | 525366, RTX 4090 | Old code in the old env is deterministic (two runs bit-identical, LoMa and RDD). New code + torch 2.8 is not bit-identical but as close to the old run as the old run is to the paper's caches: LoMa 96.8 % of keypoints within 0.5 px, median descriptor cosine 0.99998 (old vs paper 96.8 %, 0.99994); RDD 99.98 %, 1.000000 (old vs paper 99.95 %, 0.999999). |
| L5 training | `spike_train.sbatch`, `compare.py` | 525263, RTX 4090 | One epoch of SalamanderID2025 with the paper runs' arguments (1 GPU). LoMa: pretrained baseline metrics identical; old vs old differs by 34.5 % of the epoch's weight update (GPU nondeterminism), old vs new by 34.5 %, loss 0.3352 between the old runs' 0.3362 and 0.3343. RDD-LightGlue: old vs old bit-identical; old vs new differs by 24.5 % of the update (max 4.5e-4). |

Reading: with torch 2.8 LoMa training cannot be told apart from the old environment; RDD training
is deterministic in the old environment and diverges in the new one by less than LoMa's own
run-to-run spread, so RDD checkpoints cannot be reproduced bit for bit with torch 2.8.
