# CzechLynx time-closed workflow

The integration uses masked real CzechLynx images and the metadata
`split-time_closed` protocol. The converter creates a symlink view with
`train`, `val`, and `test` directories; validation is a deterministic 20%
encounter holdout from metadata training encounters, while metadata test
frames are reserved for final evaluation.

## Run order

From the benchmark repository:

```bash
sbatch slurm_scripts/prepare_czechlynx.sh
```

Then build the RDD cache and mine the shared train/validation indices:

```bash
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/build_czechlynx_rdd_cache.sh
bash slurm_scripts/spawn_czechlynx_mining.sh
```

After aggregation, run RDD first and LoMa second:

```bash
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_rdd.sh
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/build_czechlynx_loma_cache.sh
sbatch /home/kargin/Projects/repositories/lynx-finetuning/slurm_scripts/train_czechlynx_loma.sh
```

The benchmark evaluator is launched with `CZECHLYNX_BACKEND=rdd` or
`CZECHLYNX_BACKEND=loma`, and writes separate full-gallery or top-15 JSON
reports. Existing original-Lynx scripts and artifact directories are not
changed.
