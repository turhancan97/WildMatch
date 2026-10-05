from pathlib import Path

import torch

from wildmatch.matcher_finetune.rdd_patch.lightglue_masked_training import LightGlueForTraining
from wildmatch.vendor.rdd import CONFIG_PATH as RDD_CONFIG_PATH
from wildmatch.vendor.rdd import LG_WEIGHTS, RDD_WEIGHTS, resolve_weights  # noqa: F401 (re-exported)
from wildmatch.vendor.rdd.RDD.RDD import build
from wildmatch.vendor.rdd.RDD.utils import read_config

resolve_rdd_weights = resolve_weights


def build_rdd(weights: Path, device: torch.device, top_k: int):
    rdd_conf = read_config(str(RDD_CONFIG_PATH))
    model = build(rdd_conf, weights=str(weights))
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    model.to(device)
    model.eval()
    return model


def build_masked_lg(
    device: torch.device,
    weights=None,
    init_threshold=0.01,
    detach_descriptors=True,
):
    """Builds LightGlueForTraining, not the LightGlueMasked it started out as.

    The two are numerically identical (same architecture, same weights, verified
    pair-by-pair in tests/test_dense_matching.py); LightGlueForTraining just
    additionally exposes the dense `matching_scores0` / `valid0` that
    train_common._lg_scores now reads, which — unlike the ragged `scores` list —
    keep a grad_fn when a pair produces no matches at all. Everything here goes
    through this builder, training and eval alike, so both sides score pairs the
    exact same way.
    """
    weights = resolve_rdd_weights(weights, LG_WEIGHTS)
    lg_conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": init_threshold,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": weights,
        "detach_descriptors": detach_descriptors,
    }
    lg = LightGlueForTraining("rdd", **lg_conf).to(device).eval()
    return lg
