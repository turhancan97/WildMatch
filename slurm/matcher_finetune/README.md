# Original lynx-finetuning wrappers (reference only)

These are the Slurm wrappers of `lynx-finetuning` `6958d3d`, merged with their history. They no
longer run here (they activate the old `loma`/`rdd` conda environments and call
`contrastive_finetuning.*`). Train with `wildmatch finetune-matcher` (`slurm/finetune_matcher.sbatch`),
which ports them; `tests/matcher_finetune/test_launch.py` checks the port against
`czechlynx_protocol.sh` and against the arguments and protocol files the paper's runs recorded.
The feature caches the matcher-only runs read are built by `wildmatch.matcher_finetune.build_keypoint_cache`
(RDD) and, until mining is merged, by the mining repository's LoMa cache builder.
