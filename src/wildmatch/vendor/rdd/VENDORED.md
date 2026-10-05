# Vendored RDD

- Source: `turhancan97/rdd` (fork of `xtcpete/rdd`), branch `lynx_analysis`, commit `86f0e38`, the
  checkout `lynx-finetuning` imported through its `rdd` symlink when the paper's matchers were
  fine-tuned. Licence: Apache-2.0 (`LICENSE`, copied unchanged).
- Copied byte for byte from the commit (not from the working tree, whose two uncommitted edits were
  absolute paths for the config and LightGlue weights and a `torch.amp.custom_fwd` spelling in
  `RDD/matchers/lightglue.py`): `RDD/RDD.py`, `RDD/utils/`, `RDD/models/` (the deformable-attention
  Python code included) and `configs/default.yaml`. Added: empty `__init__.py` files (the
  original folders are namespace packages) and this note.
- Left out: `RDD/matchers/` (fine-tuning uses `wildmatch.matcher_finetune.rdd_patch`),
  `RDD/dataset/`, `RDD/RDD_helper.py`, the CUDA op sources and build files (`models/ops/src/`,
  `setup.py`, `make.sh`, `test.py`) and the compiled `MultiScaleDeformableAttention` library, plus
  everything outside `RDD/` (weights, benchmarks, training, notebooks).
- Deformable attention runs the PyTorch fallback (`ms_deform_attn_core_pytorch`): the compiled
  module was importable in neither the `rdd` nor the `loma` conda environment (checked
  2026-10-05), so training and mining used the fallback too. Do not add the compiled op without a
  parity check; it is numerically different.
- Weights are not vendored (173 MB); they come from `paths.external.rdd_weights_dir`
  (`RDD-v2.pth`, `RDD_lg-v2.pth`) or an explicit `--rdd_weights`/`--lg_weights`.
