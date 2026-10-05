from typing import Dict, List, Tuple

import torch

from wildmatch.mining.lightglue_masked import LightGlueMasked, pad_to_length
from wildmatch.mining.lynx_benchmark import FrameFeat


def get_query_data(
    feat: FrameFeat, device, repeats: int = 1
) -> Dict[str, torch.Tensor]:
    ks, ds, sizes = [], [], []

    k = torch.from_numpy(feat.keypoints).to(device)
    d = torch.from_numpy(feat.descriptors).to(device)
    size = torch.tensor(feat.image_size[::-1].copy(), device=device)

    mask = torch.ones(k.shape[0], 1, dtype=torch.bool, device=device)

    def repeat_tensor(t, repeats):
        return t.unsqueeze(0).expand(repeats, *t.shape)

    ks, masks = repeat_tensor(k, repeats), repeat_tensor(mask, repeats)
    ds = repeat_tensor(d, repeats)
    sizes = repeat_tensor(size, repeats)

    return {
        "keypoints": ks,
        "descriptors": ds,
        "image_size": sizes,
        "masks": masks.unsqueeze(1),
    }


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


def score_batch_lightglue(
    lg: LightGlueMasked,
    q_data: Dict[str, torch.Tensor],
    g_data: Dict[str, torch.Tensor],
    device,
) -> Tuple[float, int]:
    pred = lg(
        {
            "image0": q_data,
            "image1": g_data,
        }
    )

    B = q_data["keypoints"].shape[0]

    scores_out = torch.zeros(B, device=device)
    matches_out = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(B):
        conf = pred["scores"][i]

        if conf.numel() == 0:
            continue

        sum_conf = conf.sum()
        norm = min(
            max(1, q_data["masks"][i].squeeze(0).count_nonzero()),
            max(1, g_data["masks"][i].squeeze(0).count_nonzero()),
        )

        scores_out[i] = sum_conf / norm
        
        matches_out[i] = (conf > 0).sum()

    return scores_out, matches_out


@torch.inference_mode()
def sequence_score_per_video_and_per_frame(
    lg,
    q_feats: List[FrameFeat],
    g_feats: List[FrameFeat],
    device: torch.device,
) -> torch.Tensor:
    if len(q_feats) == 0 or len(g_feats) == 0:
        return torch.empty(len(q_feats), len(g_feats))

    g_data = get_batched_data(g_feats, device)

    all_scores = []

    for qf in q_feats:
        q_data = get_query_data(qf, device, len(g_feats))

        scores, _ = score_batch_lightglue(lg, q_data, g_data, device)

        all_scores.append(scores)

    scores_matrix = torch.stack(all_scores, dim=0)

    return scores_matrix

def sequence_metrics_from_scores(
    scores_matrix: torch.Tensor,
    top_m: int,
) -> Dict[str, float]:

    if scores_matrix.numel() == 0:
        return {"score": 0.0, "avg_matches": None, "frame_scores": []}

    # best score dla każdego query frame
    per_frame_best, _ = torch.max(scores_matrix, dim=1)

    per_frame_scores = per_frame_best.tolist()

    # top-m
    m = min(top_m, len(per_frame_scores))
    top_vals = torch.topk(per_frame_best, m).values
    seq_score = float(top_vals.mean())

    return {
        "score": seq_score,
        "avg_matches": None,
        "frame_scores": per_frame_scores,
    }
