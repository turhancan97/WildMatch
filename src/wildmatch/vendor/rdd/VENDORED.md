# Vendored RDD

- Source: `turhancan97/rdd` (fork of `xtcpete/rdd`), branch `lynx_analysis`, commit `86f0e38`, the
  checkout `lynx-finetuning` imported through its `rdd` symlink when the paper's matchers were
  fine-tuned. Licence: Apache-2.0 (`LICENSE`, copied unchanged).
- Copied byte for byte from the commit (not from the working tree, whose two uncommitted edits were
  absolute paths for the config and LightGlue weights and a `torch.amp.custom_fwd` spelling in
  `RDD/matchers/lightglue.py`): `RDD/RDD.py`, `RDD/utils/`, `RDD/models/` (the deformable-attention
  Python code included), `RDD/matchers/` (LightGlue, used by mining) and `configs/default.yaml`.
  Added: empty `__init__.py` files (the original folders are namespace packages), this note, and in
  the package `__init__.py` the config path and the weights resolver.
- Mining's own copy (`rdd-parallel-benchmark` `RDD/`, not imported) is a strict subset of this one:
  ignoring whitespace, its only line not found here is a shorter import in `models/backbone.py`;
  the checkout adds an unused ALIKED path, distributed-training helpers and two matchers. Its
  `matchers/lightglue.py` and `configs/default.yaml` are identical. Its own addition,
  `matchers/lightglue_masked.py`, lives in `wildmatch.mining.lightglue_masked`.
- Left out: `RDD/dataset/`, `RDD/RDD_helper.py`, the CUDA op sources and build files (`models/ops/src/`,
  `setup.py`, `make.sh`, `test.py`) and the compiled `MultiScaleDeformableAttention` library, plus
  everything outside `RDD/` (weights, benchmarks, training, notebooks).
- Deformable attention runs the PyTorch fallback (`ms_deform_attn_core_pytorch`): the compiled
  module was importable in neither the `rdd` nor the `loma` conda environment (checked
  2026-10-05), so training and mining used the fallback too. Do not add the compiled op without a
  parity check; it is numerically different.
- Weights are not vendored (173 MB); they come from `paths.external.rdd_weights_dir`
  (`RDD-v2.pth`, `RDD_lg-v2.pth`) or an explicit `--rdd_weights`/`--lg_weights`.
