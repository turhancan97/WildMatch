from __future__ import annotations

import argparse
import copy
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms

from torch.utils.data import Subset

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.process import align_tensors_to_max_length

LossFn = Callable[..., tuple[torch.Tensor, dict]]
"""
loss_fn(feats_a, feats_p, feats_n, matches_pos, matches_neg, loss_hyperparam,
        neg_grad_scale=1.0, match_boost_weight=0.0, match_boost_temperature=0.1)
    -> (loss, stats)

`feats_*` carry the *training* (gradient-bearing) descriptors; `matches_pos`/
`matches_neg` are LG correspondence indices computed upstream by
run_lg_matching (possibly from an EMA teacher's descriptors, decoupled from
`feats_*`) — loss_fn no longer calls LightGlue itself.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────
def add_common_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by all matched-descriptor training scripts.

    Loss-specific hyperparameters (e.g. --margin, --temperature) are added by
    each script's own parse_args().
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

    # ── anti-collapse mechanisms (independent, combinable — see train_epoch) ──
    p.add_argument(
        "--ema_decay", type=float, default=0.0,
        help="EMA decay for a teacher descriptor network used only to generate LG "
             "correspondence labels, decoupling the matching oracle from the network "
             "being trained (0 disables; typical 0.999)",
    )
    p.add_argument(
        "--neg_grad_scale", type=float, default=1.0,
        help="Gradient scale applied to the negative-similarity term (1.0 = unchanged; "
             "<1 weakens the push-apart signal on LG-matched negatives without changing "
             "the forward loss value)",
    )
    p.add_argument(
        "--match_boost_weight", type=float, default=0.0,
        help="Max weight of the auxiliary full-candidate-set retrieval loss that "
             "encourages unique/confident matches; activated automatically (ramped "
             "0..weight) when matches/mean_pos trends down (0 disables)",
    )
    p.add_argument(
        "--match_boost_window", type=int, default=50,
        help="Number of steps averaged when tracking the matches/mean_pos trend",
    )
    p.add_argument(
        "--match_boost_drop_frac", type=float, default=0.15,
        help="Relative drop from the best windowed matches/mean_pos average that "
             "fully activates the match-boost term",
    )
    p.add_argument(
        "--match_boost_temperature", type=float, default=0.1,
        help="Softmax temperature for the auxiliary retrieval loss",
    )
    p.add_argument(
        "--match_boost_freq_decay", type=float, default=0.98,
        help="EMA decay for matches/boost_activation_freq, the logged fraction of "
             "steps where the match-boost term is active",
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


def scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Identity in the forward pass; scales the gradient flowing through `x` by `scale`.

    Used to weaken (scale < 1) the negative-similarity term's push-apart signal
    without touching the logged/forward loss value — see --neg_grad_scale.
    """
    if scale == 1.0:
        return x
    return scale * x + (1 - scale) * x.detach()


def retrieval_boost_loss(
    da: torch.Tensor, dp: torch.Tensor, m_pos: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    Full-candidate-set retrieval cross-entropy: for each LG-matched anchor point,
    push it to be the *uniquely* closest point to its true correspondent among ALL
    descriptors in the other image, not just the one it is already matched to.

    Unlike the pos_sims term (raw similarity magnitude), this directly targets
    match *uniqueness*, which is what LightGlue's mutual-NN + threshold filtering
    requires to keep a match — meant as a counter-pressure when matches/mean_pos
    is trending down (see MatchTrendMonitor / --match_boost_weight).
    """
    anchor_idx = m_pos[:, 0]
    pos_idx    = m_pos[:, 1]
    sim = da[anchor_idx] @ dp.t() / temperature  # (M, Np)
    return F.cross_entropy(sim, pos_idx)


class MatchTrendMonitor:
    """
    Tracks a windowed running mean of matches/mean_pos and compares it to the best
    windowed mean seen so far. `update()` returns a smooth [0, 1] activation that
    ramps up as the current window falls below the best-ever window by more than
    `drop_frac` — meant to scale --match_boost_weight dynamically rather than
    toggling it on/off abruptly.

    `best` (the best windowed mean seen so far, None until the window first
    fills) and `activation_freq` (an EMA, decay=`freq_decay`, of the fraction
    of steps where activation > 0) are kept as public attributes so the caller
    can log them — e.g. to see when/how often the boost term is kicking in.
    """

    def __init__(self, window: int, drop_frac: float, freq_decay: float = 0.98) -> None:
        self.window = window
        self.drop_frac = max(drop_frac, 1e-6)
        self.freq_decay = freq_decay
        self._buf: deque[float] = deque(maxlen=window)
        self.best: float | None = None
        self.activation_freq: float = 0.0

    def update(self, mean_pos_matches: float) -> float:
        self._buf.append(mean_pos_matches)
        if len(self._buf) < self.window:
            activation = 0.0
        else:
            current = sum(self._buf) / len(self._buf)
            if self.best is None or current > self.best:
                self.best = current
                activation = 0.0
            else:
                drop = (self.best - current) / max(self.best, 1e-6)
                activation = float(min(max(drop / self.drop_frac, 0.0), 1.0))

        is_active = 1.0 if activation > 0.0 else 0.0
        self.activation_freq = (
            self.freq_decay * self.activation_freq + (1 - self.freq_decay) * is_active
        )
        return activation


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
def extract_train(rdd: torch.nn.Module, images: torch.Tensor, return_dense: bool = False):
    """
    One RDD forward pass with gradients flowing only through the descriptor.

    images must be div-by-32 aligned (use resize_long_side first).  Keypoint
    positions come from the frozen detector and are detached; descriptors at
    those positions retain their computation graph.

    Returns list of B dicts: {keypoints: Tensor(N,2), descriptors: Tensor(N,D)}

    If `return_dense`, also returns (images_prep, kpts_px, valid) — the
    pre-preprocessed image tensor, the dense (B, top_k, 2) pre-filter keypoint
    grid (in the interpolator's coordinate frame), and the dense (B, top_k)
    validity mask. These let a second network (e.g. an EMA teacher) describe
    the *exact same* keypoint locations via extract_teacher_descriptors,
    without re-running detection.
    """
    raw = _unwrap(rdd)
    B, _, H, W = images.shape

    # preprocess_tensor: dtype/device cast + div-by-32 resize (no-op since
    # images are already div-by-32)
    images_prep, rh, rw = raw.preprocess_tensor(images)
    _, _, H_p, W_p = images_prep.shape

    # Forward — gradients flow through descriptor (requires_grad=True),
    # not through detector (requires_grad=False)
    M1, K1, _ = rdd(images_prep)
    M1 = F.normalize(M1, dim=1)

    # Keypoint detection: frozen detector, no gradient needed
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

    feats = [
        {
            "keypoints":   kpts_scaled[b][valid[b]].detach(),
            "descriptors": descs[b][valid[b]],       # retains gradient
        }
        for b in range(B)
    ]
    if return_dense:
        return feats, images_prep.detach(), kpts_px.detach(), valid
    return feats


# ── EMA teacher (decouples the LG correspondence oracle from the trained net) ─
def build_ema_teacher(rdd: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Deep-copy the (unwrapped) RDD model into a frozen, eval-mode teacher."""
    teacher = copy.deepcopy(_unwrap(rdd)).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    """In-place EMA update of `teacher` params towards `student` params.

    Buffers (e.g. BatchNorm running stats) are hard-copied rather than EMA'd —
    the teacher's detector is never actually run (see extract_teacher_descriptors),
    so only its descriptor-branch buffers matter, and keeping them exactly in
    sync avoids any normalization drift relative to the live student.
    """
    student = _unwrap(student)
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.mul_(decay).add_(s_p.detach(), alpha=1 - decay)
    for t_b, s_b in zip(teacher.buffers(), student.buffers()):
        t_b.copy_(s_b)


def extract_teacher_descriptors(
    teacher: torch.nn.Module,
    images_prep: torch.Tensor,
    kpts_px: torch.Tensor,
    valid: torch.Tensor,
) -> list[torch.Tensor]:
    """
    Describe the SAME keypoint locations the (frozen-detector) student already
    detected, but using the EMA teacher's descriptor weights. Reusing the
    student's locations/valid-mask guarantees index alignment without re-running
    detection (which could otherwise drift out of sync, e.g. via BatchNorm
    running stats in the detector, between two independently-run forward passes).
    """
    raw = _unwrap(teacher)
    _, _, H_p, W_p = images_prep.shape
    with torch.no_grad():
        M1_t, _ = raw.descriptor(images_prep)
        M1_t = F.normalize(M1_t, dim=1)
        descs_t = raw.interpolator(M1_t, kpts_px, H=H_p, W=W_p)
        descs_t = F.normalize(descs_t, dim=-1)
    B = descs_t.shape[0]
    return [descs_t[b][valid[b]] for b in range(B)]


# ── LG matching (shared by all loss_fn's — may be sourced from student or teacher) ─
def run_lg_matching(
    lg: torch.nn.Module,
    feats_match_a: list[dict],
    feats_match_p: list[dict],
    feats_match_n: list[dict],
    image_h: int,
    image_w: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Run LightGlue to get correspondence indices, always under no_grad — matching
    itself never receives gradient. `feats_match_*` supplies the descriptors LG
    sees; loss functions then index the (possibly different) *training*
    descriptors with the returned indices, so the labels and the gradient
    source can be decoupled (see build_ema_teacher).
    """
    with torch.no_grad():
        data_a = batch_features(feats_match_a, image_h, image_w)
        data_p = batch_features(feats_match_p, image_h, image_w)
        data_n = batch_features(feats_match_n, image_h, image_w)
        pred_pos = lg({"image0": data_a, "image1": data_p})
        pred_neg = lg({"image0": data_a, "image1": data_n})
    return pred_pos["matches"], pred_neg["matches"]


# ── training epoch ────────────────────────────────────────────────────────────
def train_epoch(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    mini_train_loader,
    mini_val_loader,
    epoch: int,
    total_epochs: int,
    args: argparse.Namespace,
    global_step: int,
    loss_fn: LossFn,
    loss_hyperparam: float,
    teacher: torch.nn.Module | None = None,
    match_monitor: MatchTrendMonitor | None = None,
) -> tuple[float, int]:
    """Shared training loop; `loss_fn` supplies the per-batch loss + stats.

    See the LossFn type comment above for the expected loss_fn signature;
    `stats` must contain at least "n_skipped", "mean_na", "mean_np", "mean_nn",
    "mean_pos_matches", "mean_neg_matches", "mean_pos_sim", "mean_neg_sim".
    Extra keys (e.g. "retrieval_acc") are logged automatically when present.

    `teacher` (built via build_ema_teacher when --ema_decay > 0) decouples the
    LG correspondence oracle from the network receiving gradient updates.
    `match_monitor` (built when --match_boost_weight > 0) tracks the
    matches/mean_pos trend across steps and dynamically scales the auxiliary
    match-boost loss term when it drops. Both are independent and combinable
    with each other and with --neg_grad_scale (passed straight from `args`).
    """
    rdd.train()
    epoch_loss     = 0.0
    epoch_skipped  = 0
    epoch_images   = 0
    mini_eval_time = 0.0
    steps_per_epoch = len(loader)
    t_epoch_start = time.perf_counter()

    pbar = tqdm(
        enumerate(loader),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:02d}",
        disable=not accelerator.is_main_process,
    )
    for step, (anchors, positives, negatives) in pbar:
        device = accelerator.device
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)

        H_r, W_r = anchors_r.shape[-2:]

        if teacher is not None:
            feats_a, prep_a, kpx_a, val_a = extract_train(rdd, anchors_r,   return_dense=True)
            feats_p, prep_p, kpx_p, val_p = extract_train(rdd, positives_r, return_dense=True)
            feats_n, prep_n, kpx_n, val_n = extract_train(rdd, negatives_r, return_dense=True)
            teacher_descs_a = extract_teacher_descriptors(teacher, prep_a, kpx_a, val_a)
            teacher_descs_p = extract_teacher_descriptors(teacher, prep_p, kpx_p, val_p)
            teacher_descs_n = extract_teacher_descriptors(teacher, prep_n, kpx_n, val_n)
            match_feats_a = [{"keypoints": fa["keypoints"], "descriptors": d} for fa, d in zip(feats_a, teacher_descs_a)]
            match_feats_p = [{"keypoints": fp["keypoints"], "descriptors": d} for fp, d in zip(feats_p, teacher_descs_p)]
            match_feats_n = [{"keypoints": fn["keypoints"], "descriptors": d} for fn, d in zip(feats_n, teacher_descs_n)]
        else:
            feats_a = extract_train(rdd, anchors_r)
            feats_p = extract_train(rdd, positives_r)
            feats_n = extract_train(rdd, negatives_r)
            match_feats_a, match_feats_p, match_feats_n = feats_a, feats_p, feats_n

        matches_pos, matches_neg = run_lg_matching(
            lg, match_feats_a, match_feats_p, match_feats_n, H_r, W_r
        )

        boost_activation = 0.0
        if match_monitor is not None:
            mean_pos_now = sum(len(m) for m in matches_pos) / max(len(matches_pos), 1)
            boost_activation = match_monitor.update(mean_pos_now)
        effective_boost_weight = args.match_boost_weight * boost_activation

        loss, stats = loss_fn(
            feats_a, feats_p, feats_n, matches_pos, matches_neg, loss_hyperparam,
            neg_grad_scale=args.neg_grad_scale,
            match_boost_weight=effective_boost_weight,
            match_boost_temperature=args.match_boost_temperature,
        )
        epoch_skipped += stats["n_skipped"]
        epoch_images  += len(feats_a)

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(
            (p for p in _unwrap(rdd).parameters() if p.requires_grad),
            args.grad_clip,
        )
        optimizer.step()

        if teacher is not None:
            update_ema(teacher, rdd, args.ema_decay)

        loss_val    = loss.item()
        epoch_loss += loss_val
        global_step += 1

        progress = (epoch * steps_per_epoch + step + 1) / (total_epochs * steps_per_epoch)
        postfix = {"loss": f"{loss_val:.4f}", "skip": stats["n_skipped"]}
        if "retrieval_acc" in stats:
            postfix["acc"] = f"{stats['retrieval_acc']:.2f}"
        pbar.set_postfix(**postfix)

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss":    loss_val,
                    "train/mean_na": stats["mean_na"],
                    "train/mean_np": stats["mean_np"],
                    "train/mean_nn": stats["mean_nn"],
                    "progress":      progress,
                },
                step=global_step,
            )
            match_log = {
                "matches/mean_pos":       stats["mean_pos_matches"],
                "matches/mean_neg":       stats["mean_neg_matches"],
                "matches/mean_pos_sim":   stats["mean_pos_sim"],
                "matches/mean_neg_sim":   stats["mean_neg_sim"],
                "matches/boost_weight":   effective_boost_weight,
                "matches/boost_activation": boost_activation,
            }
            if match_monitor is not None:
                match_log["matches/boost_activation_freq"] = match_monitor.activation_freq
                if match_monitor.best is not None:
                    match_log["matches/boost_best"] = match_monitor.best
            if "retrieval_acc" in stats:
                match_log["matches/retrieval_acc"] = stats["retrieval_acc"]
            accelerator.log(match_log, step=global_step)

        if (step + 1) % 100 == 0:
            t_mini_start = time.perf_counter()
            mini_train_m = eval_epoch(accelerator, rdd, lg, mini_train_loader, args, prefix="mini_train")
            mini_val_m   = eval_epoch(accelerator, rdd, lg, mini_val_loader,   args, prefix="mini_val")
            mini_eval_time += time.perf_counter() - t_mini_start
            rdd.train()
            if accelerator.is_main_process:
                accelerator.log({**mini_train_m, **mini_val_m, "progress": progress}, step=global_step)

    epoch_total_time = time.perf_counter() - t_epoch_start
    epoch_train_time = epoch_total_time - mini_eval_time
    skip_rate = epoch_skipped / max(epoch_images, 1)

    if accelerator.is_main_process:
        accelerator.log(
            {
                "train/skip_rate":    skip_rate,
                "train/n_skipped":    epoch_skipped,
                "time/train_s":       epoch_train_time,
                "time/mini_eval_s":   mini_eval_time,
            },
            step=global_step,
        )

    return epoch_loss / max(steps_per_epoch, 1), global_step


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
    A good descriptor should give many matches on positive pairs and few on
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


# ── full training run ─────────────────────────────────────────────────────────
def run_training(args: argparse.Namespace, loss_fn: LossFn, loss_hyperparam: float) -> None:
    """
    Shared setup + epoch loop for matched-descriptor contrastive training.

    `loss_fn` is the loss (e.g. margin triplet or InfoNCE) applied every step;
    `loss_hyperparam` is its scalar hyperparameter (margin, temperature, ...).
    """
    seed_all(args.seed)

    accelerator = Accelerator(log_with="wandb" if args.project else None)
    device = accelerator.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    transform = transforms.ToTensor()
    train_ds = IndexAssignedTripletDataset(
        args.train_index, root=args.data_root, transform=transform,
    )
    val_ds = IndexAssignedTripletDataset(
        args.val_index, root=args.data_root, transform=transform,
    )

    train_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
    )

    _rng = random.Random(args.seed)

    def _fixed_subset(ds, fraction):
        n = max(1, int(len(ds) * fraction))
        return Subset(ds, _rng.sample(range(len(ds)), min(n, len(ds))))

    mini_train_loader = get_loader(
        _fixed_subset(train_ds, 10 * args.batch_size / max(len(train_ds), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    mini_val_loader = get_loader(
        _fixed_subset(val_ds, 10 * args.batch_size / max(len(val_ds), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    eval_train_subset = _fixed_subset(train_ds, args.eval_fraction)
    eval_val_subset   = _fixed_subset(val_ds,   args.eval_fraction)

    # ── models ──
    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    lg  = build_masked_lg(device, weights=args.lg_weights)
    lg.eval()

    # EMA teacher: decouples the LG correspondence oracle from the network
    # receiving gradient updates (see train_epoch docstring). Built from rdd's
    # initial weights, before accelerator.prepare wraps rdd for DDP.
    teacher = build_ema_teacher(rdd, device) if args.ema_decay > 0 else None

    # Match-count trend monitor: dynamically activates the match-boost loss
    # term (independent of, and combinable with, the EMA teacher above).
    match_monitor = (
        MatchTrendMonitor(
            args.match_boost_window, args.match_boost_drop_frac,
            freq_decay=args.match_boost_freq_decay,
        )
        if args.match_boost_weight > 0 else None
    )

    # Only descriptor parameters are trainable (detector frozen in build_rdd)
    optimizer = torch.optim.Adam(
        [p for p in rdd.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    rdd, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
        rdd, optimizer, train_loader, mini_train_loader, mini_val_loader
    )

    if args.project:
        accelerator.init_trackers(
            args.project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )

    # ── baseline eval (before any training) ──
    global_step = 0
    baseline_train = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
    baseline_val   = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
    rdd.train()
    if accelerator.is_main_process:
        accelerator.log({**baseline_train, **baseline_val, "epoch": -1}, step=global_step)

    # ── loop ──
    for epoch in range(args.epochs):
        epoch_loss, global_step = train_epoch(
            accelerator, rdd, lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
            loss_fn, loss_hyperparam,
            teacher=teacher, match_monitor=match_monitor,
        )

        t_eval_start = time.perf_counter()
        train_eval_metrics = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
        val_metrics        = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
        epoch_eval_time = time.perf_counter() - t_eval_start
        rdd.train()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":              epoch,
            "train/epoch_loss":   epoch_loss,
            "train/lr":           lr,
            "time/epoch_eval_s":  epoch_eval_time,
            **train_eval_metrics,
            **val_metrics,
        }

        if accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)
            accelerator.save_state(str(args.output_dir / f"epoch_{epoch:02d}"))

    if args.project:
        accelerator.end_training()
