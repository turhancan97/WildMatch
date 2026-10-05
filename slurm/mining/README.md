# Original rdd-parallel-benchmark mining wrappers (reference only)

The Slurm wrappers of `rdd-parallel-benchmark` `4f29292`, merged with their history. They no longer
run here (they activate the old conda environments and call `scripts.*`). Mine with
`wildmatch mine <plan|view|cache|check|task|aggregate|submit>` (`src/wildmatch/mining/launch.py`,
Slurm script `slurm/mine.sbatch`), which ports them together with the cache wrappers in
`slurm/matcher_finetune/build_*_cache.sh`.
