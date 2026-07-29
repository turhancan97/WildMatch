from pathlib import Path

import torch

from rdd_patch.lightglue_masked import LightGlueMasked
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
    lg = LightGlueMasked("rdd", **lg_conf).to(device).eval()
    return lg
