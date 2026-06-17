from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from rdd.RDD.RDD import RDD
from rdd_patch.lightglue_masked import LightGlueMasked

@dataclass
class FrameFeat:
    keypoints: np.ndarray  # [K, 2]
    descriptors: np.ndarray  # [K, 256]
    scores: np.ndarray  # [K]
    image_size: np.ndarray  # [2] (H, W)

def extract_frame(model: RDD, img: torch.Tensor, device: torch.device, top_k: int) -> FrameFeat:
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    img = img.to(device)
    out = model.extract(img)[0]
    return FrameFeat(
        keypoints=out["keypoints"].cpu().numpy(),
        descriptors=out["descriptors"].cpu().numpy(),
        scores=out["scores"].cpu().numpy(),
        image_size=np.array(img.shape[-2:], dtype=np.int32),  # (H, W)
    )

def pad_to_length(x: torch.Tensor, length: int) -> Tuple[torch.Tensor]:
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(
        *x.shape[:-2], length - x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
    )
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., : x.shape[-2], :] = True
    return y, mask


def align_tensors_to_max_length(ts: List[torch.Tensor]):
    B = max(len(t) for t in ts)

    pts, ms = [], []
    for t in ts:
        pt, m = pad_to_length(t, B)
        pts.append(pt)
        ms.append(m)

    pts = torch.stack(pts)
    ms = torch.stack(ms)

    return pts, ms

def get_batched_data(feats: List[FrameFeat], device) -> Dict[str, torch.Tensor]:
    ks, ds, sizes = [], [], []

    for feat in feats:
        k = torch.from_numpy(feat.keypoints).to(device)
        d = torch.from_numpy(feat.descriptors).to(device)
        size = torch.tensor(feat.image_size[::-1].copy(), device=device)

        ks.append(k)
        ds.append(d)
        sizes.append(size)

    ks, masks = align_tensors_to_max_length(ks)
    ds, _ = align_tensors_to_max_length(ds)
    sizes = torch.stack(sizes)

    return {
        "keypoints": ks,
        "descriptors": ds,
        "image_size": sizes,
        "masks": masks.unsqueeze(1),
    }

def get_matches_lightglue(
    lg: LightGlueMasked,
    q_data: Dict[str, torch.Tensor],
    g_data: Dict[str, torch.Tensor],
) -> Tuple[float, int]:
    pred = lg(
        {
            "image0": q_data,
            "image1": g_data,
        }
    )
    return pred
