from pathlib import Path

import torch

from rdd_patch.lightglue_masked_training import LightGlueForTraining
from rdd.RDD.RDD import build
from rdd.RDD.utils import read_config

def build_rdd(weights: Path, device: torch.device, top_k: int):
    rdd_conf = read_config("rdd/configs/default.yaml")
    model = build(rdd_conf, weights=str(weights))
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    model.to(device)
    model.eval()
    return model

def build_masked_lg(
    device: torch.device, weights="rdd/weights/RDD_lg-v2.pth", init_threshold=0.01,
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
