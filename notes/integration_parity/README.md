# Integration parity runs (2026-10-05)

Scripts behind parity levels L2, L3 and L5 and the environment spike (decision D3) of
`notes/integration_plan.md`. They are kept as run; paths point at the cluster and at the integration
worktrees under `/shared/results/common/kargin/projects/wildmatch-integration/`.

| Level | Script | Job | Result |
| --- | --- | --- | --- |
| L2 views | `compare_views.py` | CPU | Rebuilt views equal the live training views for all eight paper datasets and CzechLynx open (every directory and symlink; only `manifest.json`'s `output_root` differs). |
| L3 features | `spike_features.sbatch`, `compare_features.py`, `match_features.py` | 525366, RTX 4090 | Old code in the old env is deterministic (two runs bit-identical, LoMa and RDD). New code + torch 2.8 is not bit-identical but as close to the old run as the old run is to the paper's caches: LoMa 96.8 % of keypoints within 0.5 px, median descriptor cosine 0.99998 (old vs paper 96.8 %, 0.99994); RDD 99.98 %, 1.000000 (old vs paper 99.95 %, 0.999999). |
| L4 mining | `spike_mining.sbatch`, `compare_mining.py`, `query_ids.txt` | 525384, RTX 4090 | 24 seeded-random SalamanderID2025 train queries, both miners reading the paper's feature caches: the merged code under torch 2.8 selects exactly the positives and negatives of the old code and of the paper's per-query files (24/24 for LoMa and for RDD-LightGlue; scores within 3e-8; the old code reproduces the paper files bit for bit). |
| L5 training | `spike_train.sbatch`, `compare.py` | 525263, RTX 4090 | One epoch of SalamanderID2025 with the paper runs' arguments (1 GPU). LoMa: pretrained baseline metrics identical; old vs old differs by 34.5 % of the epoch's weight update (GPU nondeterminism), old vs new by 34.5 %, loss 0.3352 between the old runs' 0.3362 and 0.3343. RDD-LightGlue: old vs old bit-identical; old vs new differs by 24.5 % of the update (max 4.5e-4). |
| L6 checkpoints | `l6.sbatch`, `compare_l6.py`, `tasks.tsv` | 525388, RTX 4090 | Each checkpoint of `conf/weights.yaml` evaluated at k=50 by the main env with main's code and by the integration env, same GPU, fresh caches each: 13 of 18 finished (2026-10-05), all with bit-identical score matrices and identical metrics (hyenaid2022 loma, hyenaid2022 rdd-lightglue, leopardid2022 loma, leopardid2022 rdd-lightglue, nyala loma, nyala rdd-lightglue, salamander loma, salamander rdd-lightglue, seastarreid2023 loma, seastarreid2023 rdd-lightglue, whaleshark loma, whaleshark rdd-lightglue, zindi loma). The CzechLynx tasks (about 2 h each: two full extractions of ~28k images) were still running when this was written; their result is in `runs/525388/<task>/compare.txt` under the spike folder. |
| D3 outcome, RDD | `full_rdd_eval.sbatch` (training: `wildmatch finetune-matcher`, job 525382) | 525382 + 525383, RTX 4090 | Full 300-epoch SalamanderID2025 RDD-LightGlue run in the new env (4 GPUs x 8, paper arguments), evaluated with the paper checkpoint on one GPU. Top-1 / Top-10 / balanced Top-1, paper vs new: k=50 28.0/29.3/29.0 vs 27.2/28.9/28.1; k=250 41.9/46.7/43.3 vs 39.8/44.7/41.0. Per query (246): k=50 3 vs 1 flips (exact McNemar p=0.63), k=250 9 vs 4 (p=0.27). Not significant; accepted as run-to-run variation (user decision 2026-10-05, no control runs). |

Reading: with torch 2.8 LoMa training cannot be told apart from the old environment; RDD training
is deterministic in the old environment and diverges in the new one by less than LoMa's own
run-to-run spread, so RDD checkpoints cannot be reproduced bit for bit with torch 2.8.
