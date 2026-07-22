from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from torch.utils.data import Subset

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.process import align_tensors_to_max_length


# ── CLI ───────────────────────────────────────────────────────────────────────
def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by all training scripts.

    Loss-specific hyperparameters (e.g. --lg_margin) are added by each
    script's own parse_args().
    """
    p.add_argument("--train_index",   type=Path, required=True, help="JSON triplet index for training")
    p.add_argument("--val_index",     type=Path, required=True, help="JSON triplet index for validation")
    p.add_argument("--data_root",     type=Path, default=None,  help="Root prepended to relative paths in the index")
    p.add_argument("--rdd_weights",   type=str,  default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights",    type=str,  default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--output_dir",    type=Path, default=Path("checkpoints"))
    p.add_argument("--project",       type=str,  default=None,  help="wandb project name")
    p.add_argument("--run_name",      type=str,  default=None)
    p.add_argument("--epochs",        type=int,  default=10)
    p.add_argument("--batch_size",    type=int,  default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--resize",        type=int,  default=512)
    p.add_argument("--top_k",         type=int,  default=512)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--num_workers",   type=int,  default=4)
    p.add_argument("--eval_fraction", type=float, default=0.1, help="Fraction of dataset used for post-epoch eval")
    p.add_argument(
        "--wandb_tags", type=str, default="",
        help="Comma-separated wandb tags for this run",
    )


# ── utils ─────────────────────────────────────────────────────────────────────
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return base model regardless of DDP / Accelerate wrapping."""
    return model.module if hasattr(model, "module") else model


def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize so the long side == size and dims are div-by-32."""
    _, _, H, W = images.shape
    scale = size / max(H, W)
    new_H = int(H * scale) // 32 * 32
    new_W = int(W * scale) // 32 * 32
    return F.interpolate(images.float(), (new_H, new_W), mode="bilinear", align_corners=False)


def build_wandb_tags(args: argparse.Namespace) -> list[str]:
    """`--wandb_tags` (comma-separated) as a list; falls back to ["run"] if empty."""
    extra = getattr(args, "wandb_tags", "") or ""
    tags = [t.strip() for t in extra.split(",") if t.strip()]
    return tags or ["run"]


def batch_features(feats: list[dict], image_h: int, image_w: int) -> dict:
    """
    Pack variable-length feature dicts into tensors for LightGlueMasked.

    Args:
        feats: list of B dicts with 'keypoints' (N_i, 2) and 'descriptors' (N_i, D)
        image_h, image_w: image dimensions for keypoint normalisation

    Returns dict with keypoints, descriptors, masks, image_size ready for LG.
    """
    ks = [f["keypoints"]   for f in feats]
    ds = [f["descriptors"] for f in feats]
    device = ks[0].device

    ks_pad, masks = align_tensors_to_max_length(ks)   # (B, M, 2), (B, M, 1)
    ds_pad, _     = align_tensors_to_max_length(ds)   # (B, M, D)

    # image_size as [W, H] — LightGlue normalize_keypoints convention
    sizes = torch.tensor(
        [image_w, image_h], device=device
    ).unsqueeze(0).expand(len(feats), -1).contiguous()

    return {
        "keypoints":   ks_pad,
        "descriptors": ds_pad,
        "image_size":  sizes,
        "masks":       masks.unsqueeze(1),  # (B, 1, M, 1) for masked attention
    }


# ── training-time feature extraction ─────────────────────────────────────────
def extract_train(rdd: torch.nn.Module, images: torch.Tensor) -> list[dict]:
    """
    One RDD forward pass. RDD is always frozen for LG-only training (see
    train_lg_matching_loss.py), so gradients never flow through it here, but
    this stays a plain forward — the caller wraps it in torch.no_grad() when
    it wants to skip building the graph.

    images must be div-by-32 aligned (use resize_long_side first).

    Returns list of B dicts: {keypoints: Tensor(N,2), descriptors: Tensor(N,D)}
    """
    raw = _unwrap(rdd)
    B, _, H, W = images.shape

    # preprocess_tensor: dtype/device cast + div-by-32 resize (no-op since
    # images are already div-by-32)
    images_prep, rh, rw = raw.preprocess_tensor(images)
    _, _, H_p, W_p = images_prep.shape

    M1, K1, _ = rdd(images_prep)
    M1 = F.normalize(M1, dim=1)

    with torch.no_grad():
        kpts, kscores, _ = raw.softdetect(K1)
        kpts    = torch.vstack([kpts[b].unsqueeze(0)    for b in range(B)])
        kscores = torch.vstack([kscores[b].unsqueeze(0) for b in range(B)])
        kpts_px     = to_pixel_coords(kpts, H_p, W_p)
        kpts_scaled = kpts_px * torch.tensor([rw, rh], device=images.device).view(1, -1)
        valid = kscores > raw.detection_threshold

    # Interpolate descriptor map at keypoints — F.grid_sample is differentiable
    descs = raw.interpolator(M1, kpts_px, H=H_p, W=W_p)
    descs = F.normalize(descs, dim=-1)

    return [
        {
            "keypoints":   kpts_scaled[b][valid[b]].detach(),
            "descriptors": descs[b][valid[b]],
        }
        for b in range(B)
    ]


# ── LG matching ────────────────────────────────────────────────────────────────
def run_lg_matching_grad(
    lg: torch.nn.Module,
    feats_a: list[dict],
    feats_p: list[dict],
    feats_n: list[dict],
    image_h: int,
    image_w: int,
) -> tuple[dict, dict]:
    """
    Run LightGlue WITHOUT a no_grad wrapper, so LG's own parameters receive
    gradient from any loss computed on the returned `scores` /
    `matching_scores0`. Returns the full prediction dicts (not just match
    indices) — used by train_lg_matching_loss.py.

    Note LightGlueMasked always internally `.detach()`s its descriptor
    *inputs* (see rdd_patch/lightglue_masked.py), so this can never send
    gradient back into the network that produced feats_* — only into LG's
    own weights.
    """
    data_a = batch_features(feats_a, image_h, image_w)
    data_p = batch_features(feats_p, image_h, image_w)
    data_n = batch_features(feats_n, image_h, image_w)
    pred_pos = lg({"image0": data_a, "image1": data_p})
    pred_neg = lg({"image0": data_a, "image1": data_n})
    return pred_pos, pred_neg


# ── validation epoch ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_epoch(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    prefix: str,
) -> dict:
    """
    Compute mean number of LightGlue matches for positive and negative pairs.
    A good matcher should give many matches on positive pairs and few on
    negative pairs.  `prefix` is prepended to every returned metric key.
    """
    device = accelerator.device
    _unwrap(rdd).eval()
    total_pos = torch.zeros(1, device=device)
    total_neg = torch.zeros(1, device=device)
    n         = torch.zeros(1, device=device)

    for anchors, positives, negatives in loader:
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)
        H_r, W_r = anchors_r.shape[-2:]

        feats_a = extract_train(_unwrap(rdd), anchors_r)
        feats_p = extract_train(_unwrap(rdd), positives_r)
        feats_n = extract_train(_unwrap(rdd), negatives_r)

        data_a = batch_features(feats_a, H_r, W_r)
        data_p = batch_features(feats_p, H_r, W_r)
        data_n = batch_features(feats_n, H_r, W_r)

        pred_pos = lg({"image0": data_a, "image1": data_p})
        pred_neg = lg({"image0": data_a, "image1": data_n})

        total_pos += sum(len(m) for m in pred_pos["matches"])
        total_neg += sum(len(m) for m in pred_neg["matches"])
        n += len(feats_a)

    # Sum counts across all processes before computing means
    total_pos = accelerator.reduce(total_pos, reduction="sum")
    total_neg = accelerator.reduce(total_neg, reduction="sum")
    n         = accelerator.reduce(n,         reduction="sum")

    mean_pos = (total_pos / n.clamp(min=1)).item()
    mean_neg = (total_neg / n.clamp(min=1)).item()
    return {
        f"{prefix}/mean_matches_pos": mean_pos,
        f"{prefix}/mean_matches_neg": mean_neg,
    }


# ── pseudo-accuracy eval ──────────────────────────────────────────────────────
@torch.no_grad()
def eval_pseudo_accuracy(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    dataset_subset,
    args: argparse.Namespace,
    prefix: str,
) -> dict:
    """
    For each query in the subset, run LG against every positive and every
    negative candidate listed in the JSON index.  The candidate with the most
    matches wins; the prediction is correct when that winner is a positive.

    Also returns mean match counts over all pos/neg pairs as a byproduct.
    """
    device = accelerator.device
    _unwrap(rdd).eval()

    if isinstance(dataset_subset, Subset):
        base_ds = dataset_subset.dataset
        entries = [base_ds._entries[i] for i in dataset_subset.indices]
    else:
        base_ds = dataset_subset
        entries = base_ds._entries

    accuracies: list[float] = []
    best_pos_scores: list[float] = []
    best_neg_scores: list[float] = []

    for entry in tqdm(entries, desc=f"{prefix}", leave=False, disable=not accelerator.is_main_process):
        query_img = base_ds._loader(base_ds._full_path(entry["query_frame"]))
        if base_ds.query_transform is not None:
            query_img = base_ds.query_transform(query_img)
        query_r = resize_long_side(query_img.unsqueeze(0).to(device), args.resize)
        H_q, W_q = query_r.shape[-2:]
        feats_q = extract_train(_unwrap(rdd), query_r)

        def _score(rel_path: str) -> float:
            cand_img = base_ds._loader(base_ds._full_path(rel_path))
            if base_ds.transform is not None:
                cand_img = base_ds.transform(cand_img)
            cand_r = resize_long_side(cand_img.unsqueeze(0).to(device), args.resize)
            H_c, W_c = cand_r.shape[-2:]
            feats_c = extract_train(_unwrap(rdd), cand_r)
            pred = lg({
                "image0": batch_features(feats_q, H_q, W_q),
                "image1": batch_features(feats_c, H_c, W_c),
            })
            s = pred["scores"][0]
            return s.mean().item() if s.numel() > 0 else 0.0

        score_pos = max(_score(p) for p in entry["positives"])
        score_neg = max(_score(n) for n in entry["negatives"])

        if score_pos > score_neg:
            accuracies.append(1.0)
        elif score_pos == score_neg:
            accuracies.append(0.5)
        else:
            accuracies.append(0.0)

        best_pos_scores.append(score_pos)
        best_neg_scores.append(score_neg)

    n = len(entries)
    return {
        f"{prefix}/pseudo_accuracy": sum(accuracies)    / max(n, 1),
        f"{prefix}/mean_score_pos":  sum(best_pos_scores) / max(n, 1),
        f"{prefix}/mean_score_neg":  sum(best_neg_scores) / max(n, 1),
    }
