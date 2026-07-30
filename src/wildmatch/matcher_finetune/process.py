from typing import List, Tuple

import torch


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

