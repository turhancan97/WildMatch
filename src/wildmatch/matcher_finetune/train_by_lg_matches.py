from __future__ import annotations

import argparse
import contextlib
import copy
import math
import random
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms

from torch.utils.data import Subset

from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    _lg_scores, _unwrap, add_common_args, batch_features, build_wandb_tags, eval_epoch,
    eval_pseudo_accuracy, extract_train, resize_long_side, resolve_trained_models,
    run_lg_matching_grad, seed_all,
)

"""
Trains via a margin loss on LightGlue's OWN matching confidence (`scores`,
the per-match probability LG assigns to the pairs it selects) rather than on
raw descriptor dot products — i.e. treating LG's own matches as the training
signal instead of a fixed, external oracle for evaluating matches (as a
descriptor-training setup would).

`--trained_model` (lg / rdd / lg+rdd) decides which model(s) are unfrozen and
receive gradient; the other stays frozen. LightGlueMasked detaches its
descriptor *inputs* by default (see `detach_descriptors` in
rdd_patch/lightglue_masked.py), which is what normally keeps this loss from
backpropagating into RDD; when `--trained_model` includes 'rdd', LG is built
with `detach_descriptors=False` so gradient can reach RDD too.

Three independent, combinable anti-overfitting mechanisms, each off by default:
  --augment    photometric-only data augmentation on the training images
               (never on val — keypoint geometry is untouched, detection runs
               after augmentation, so correspondences stay valid).
  --lora / --lora_rank
               freeze LightGlue's own pretrained weights and train only
               low-rank adapters injected into its attention/assignment
               Linear layers — drastically fewer trainable params than full
               fine-tuning. The rotary positional encoding (posenc.Wr) is
               deliberately left un-adapted.
  --ema_decay  exponential moving average of LightGlue's weights, used for all
               evaluation/checkpointing instead of the raw (noisier) live
               weights — smooths over late-training variance/overfitting.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train via a margin loss on LightGlue's own matching confidence; "
                     "--trained_model picks which of LG/RDD are unfrozen."
    )
    add_common_args(p)
    p.add_argument("--lg_margin", type=float, default=0.5, help="Margin for the LightGlue match-confidence loss")
    p.add_argument(
        "--augment", action="store_true",
        help="Apply photometric-only augmentation (color jitter, light blur) to training images",
    )
    p.add_argument(
        "--lora", action="store_true",
        help="Freeze LightGlue's own weights and train only injected LoRA adapters (see --lora_rank)",
    )
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA adapter rank (only used with --lora)")
    p.add_argument(
        "--ema_decay", type=float, default=0.0,
        help="EMA decay for LightGlue's weights (0 disables); the EMA copy is used for all "
             "eval/checkpointing instead of the raw live weights (typical 0.999)",
    )
    p.add_argument(
        "--random_negative_prob", type=float, default=0.0,
        help="Probability of replacing the index-mined training negative with a random "
             "image of a different lynx drawn from the whole candidate pool, instead of "
             "just the entry's top_m negatives. 0 (default) keeps negatives index-only.",
    )
    p.add_argument(
        "--freeze_confidence_head", action="store_true",
        help="Freeze LightGlue's log_assignment and token_confidence submodules (the "
             "matchability/confidence heads) so only the attention/transformer backbone "
             "trains. Independent of --lora — applies to full fine-tuning too, and also "
             "freezes any LoRA adapters injected into those submodules when combined "
             "with --lora.",
    )
    p.add_argument(
        "--moving_negative_prob", type=float, default=None,
        help="Enables an adaptive schedule for the random-negative mixing probability, "
             "starting from this value (requires --random_negative_prob > 0, which acts "
             "as the schedule's target/reference probability). Before training, a "
             "forward-only warmup pass measures the baseline ratio of mean LG confidence "
             "between index-mined and random negatives; each epoch after that, the same "
             "ratio is measured opportunistically from real training batches and the live "
             "probability is remapped between --moving_negative_min_prob and "
             "--moving_negative_max_prob as the ratio moves relative to that baseline "
             "(see compute_moving_prob). Both confidences are logged to wandb every epoch "
             "under moving_neg/*.",
    )
    p.add_argument(
        "--moving_negative_warmup_batches", type=int, default=20,
        help="Batches used by the pre-training warmup pass for --moving_negative_prob. "
             "Negatives are temporarily forced to a ~50/50 index/random mix during this "
             "pass only, for balanced baseline statistics regardless of the configured "
             "probabilities.",
    )
    p.add_argument(
        "--moving_negative_min_prob", type=float, default=0.05,
        help="Floor for the adaptive probability driven by --moving_negative_prob.",
    )
    p.add_argument(
        "--moving_negative_max_prob", type=float, default=0.95,
        help="Ceiling for the adaptive probability driven by --moving_negative_prob.",
    )
    p.add_argument(
        "--negative_mining", action="store_true",
        help="When drawing a random negative (--random_negative_prob > 0, required), bias "
             "which lynx it's drawn from toward candidates that have produced high LG "
             "confidence for that query lynx in the past, instead of picking uniformly. "
             "Maintains a per-(query_lynx, candidate_lynx) EMA difficulty matrix, updated "
             "once per epoch from that epoch's observed negatives.",
    )
    p.add_argument(
        "--negative_mining_temperature", type=float, default=1.0,
        help="Softmax temperature over the difficulty matrix row when sampling a negative "
             "lynx under --negative_mining; lower concentrates sampling on the single "
             "hardest known candidate.",
    )
    p.add_argument(
        "--negative_mining_decay", type=float, default=0.9,
        help="EMA decay for --negative_mining's difficulty matrix (closer to 1 remembers "
             "older observations longer).",
    )
    args = p.parse_args()
    if args.moving_negative_prob is not None:
        if args.random_negative_prob <= 0:
            p.error("--moving_negative_prob requires --random_negative_prob > 0")
        if not (0.0 <= args.moving_negative_prob <= 1.0):
            p.error("--moving_negative_prob must be in [0, 1]")
    if args.negative_mining and args.random_negative_prob <= 0:
        p.error("--negative_mining requires --random_negative_prob > 0")
    return args


# ── loss ──────────────────────────────────────────────────────────────────────
def lg_confidence_loss(
    pred_pos: dict, pred_neg: dict, margin: float, device: torch.device,
    data_a: dict, data_p: dict, data_n: dict,
) -> tuple[torch.Tensor, dict]:
    """
    Margin loss on LightGlue's OWN matching confidence (`scores`):
      loss = relu(margin - pos_conf + neg_conf)

    pos_conf/neg_conf are sum(confidence) / min(valid keypoints on each
    side) — the same normalization eval_pseudo_accuracy uses (see
    _lg_scores in train_common.py) — which keeps a couple of lucky
    high-confidence matches from dominating the loss for an otherwise
    poorly-matched pair.
    """
    pos_conf_all = _lg_scores(pred_pos, data_a, data_p, device)  # (B,)
    neg_conf_all = _lg_scores(pred_neg, data_a, data_n, device)  # (B,)

    losses = []
    n_skipped = 0
    pos_match_list, neg_match_list = [], []
    pos_conf_list, neg_conf_list = [], []

    for i, (s_pos, s_neg) in enumerate(zip(pred_pos["scores"], pred_neg["scores"])):
        if s_pos.shape[0] == 0:
            n_skipped += 1
            continue

        pos_match_list.append(s_pos.shape[0])
        neg_match_list.append(s_neg.shape[0])

        pos_conf = pos_conf_all[i]
        pos_conf_list.append(pos_conf.item())

        if s_neg.shape[0] > 0:
            neg_conf = neg_conf_all[i]
            neg_conf_list.append(neg_conf.item())
            losses.append(F.relu(margin - pos_conf + neg_conf))
        else:
            losses.append(F.relu(margin - pos_conf))

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "n_skipped":        n_skipped,
        "mean_pos_matches": _mean(pos_match_list),
        "mean_neg_matches": _mean(neg_match_list),
        "mean_pos_conf":    _mean(pos_conf_list),
        "mean_neg_conf":    _mean(neg_conf_list),
    }
    if not losses:
        stats["n_skipped"] += len(pred_pos["scores"])
        return torch.zeros(1, device=device, requires_grad=True).squeeze(), stats
    return torch.stack(losses).mean(), stats


# ── augmentation ──────────────────────────────────────────────────────────────
def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform). eval_transform is always plain
    ToTensor (val/eval must stay clean). When augment, train_transform adds
    photometric-only jitter — no crop/flip/rotate, since keypoints are detected
    *after* this transform, so geometric augmentation would just shrink the
    true anchor/positive keypoint overlap rather than help.
    """
    eval_transform = transforms.ToTensor()
    if not augment:
        return eval_transform, eval_transform
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
    ])
    return train_transform, eval_transform


# ── adaptive random-negative mixing ─────────────────────────────────────────────
# Denominator floor for the index/random confidence ratio (see compute_moving_prob).
NEG_GAP_EPS = 1e-6


def compute_moving_prob(
    ratio: float, baseline_ratio: float, base_prob: float, min_prob: float, max_prob: float,
) -> float:
    """
    Maps the current index/random LG-confidence ratio to a target
    random_negative_prob, anchored at three points in log-ratio space:
      ratio <= 1                  -> max_prob   (random negatives are now at
                                      least as hard as index-mined ones)
      ratio == baseline_ratio     -> base_prob   (gap unchanged from before training)
      ratio >= baseline_ratio**2  -> min_prob    (gap has ~doubled, log-wise, past baseline)
    linearly interpolated between those anchors, clamped to [min_prob, max_prob].
    """
    if ratio <= 1.0:
        return max_prob
    if baseline_ratio <= 1.0:
        # Degenerate baseline (index negatives weren't harder than random to
        # begin with) — nothing meaningful to scale against.
        return base_prob
    x, x0 = math.log(ratio), math.log(baseline_ratio)
    if x <= x0:
        frac = (x0 - x) / x0
        return base_prob + frac * (max_prob - base_prob)
    frac = min((x - x0) / x0, 1.0)
    return base_prob - frac * (base_prob - min_prob)


@torch.no_grad()
def measure_negative_gap(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    dataset: IndexAssignedTripletDataset,
    args: argparse.Namespace,
    n_batches: int,
    force_prob: float = 0.5,
) -> tuple[float, float]:
    """
    Forward-only pass over up to `n_batches` freshly-sampled batches from
    `dataset`, with `dataset.random_negative_prob` temporarily forced to
    `force_prob` so both index-mined and random negatives show up in roughly
    equal numbers regardless of the configured/current schedule — used to
    establish --moving_negative_prob's pre-training baseline ratio. Restores
    the dataset's original probability (and return_meta flag) before
    returning. Uses a throwaway, non-persistent-worker loader so it never
    interferes with the main train_loader's worker pool.

    Returns (mean_index_conf, mean_random_conf); either is 0.0 if that source
    didn't come up in the sampled batches.
    """
    prev_prob = dataset.random_negative_prob
    prev_meta = dataset.return_meta
    dataset.random_negative_prob = force_prob
    dataset.return_meta = True
    try:
        loader = get_loader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, persistent_workers=False,
        )
        device = accelerator.device
        index_confs: list[float] = []
        random_confs: list[float] = []
        for step, (anchors, positives, negatives, neg_meta) in enumerate(loader):
            if step >= n_batches:
                break
            anchors_r   = resize_long_side(anchors,   args.resize).to(device)
            negatives_r = resize_long_side(negatives, args.resize).to(device)
            H_r, W_r = anchors_r.shape[-2:]

            feats_a = extract_train(rdd, anchors_r)
            feats_n = extract_train(rdd, negatives_r)
            data_a = batch_features(feats_a, H_r, W_r)
            data_n = batch_features(feats_n, H_r, W_r)
            pred_neg = lg({"image0": data_a, "image1": data_n})
            neg_conf = _lg_scores(pred_neg, data_a, data_n, device).tolist()

            for src, conf in zip(neg_meta["neg_source"], neg_conf):
                (index_confs if src == "index" else random_confs).append(conf)
    finally:
        dataset.random_negative_prob = prev_prob
        dataset.return_meta = prev_meta

    mean_index  = sum(index_confs)  / len(index_confs)  if index_confs  else 0.0
    mean_random = sum(random_confs) / len(random_confs) if random_confs else 0.0
    return mean_index, mean_random


# ── LoRA ──────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter (Hu et al. 2021).

    `lora_B` is zero-initialized so the wrapped layer starts out numerically
    identical to the original — training begins from LightGlue's pretrained
    behavior, not a perturbed one.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features,  device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return self.base(x) + delta.to(x.dtype)


def _apply_lora_recursive(module: nn.Module, rank: int) -> int:
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
            count += 1
        else:
            count += _apply_lora_recursive(child, rank)
    return count


def apply_lora(lg: nn.Module, rank: int) -> int:
    """
    Freezes LightGlue's own weights and injects LoRA adapters into the
    attention/FFN/assignment Linear layers (self_attn, cross_attn,
    log_assignment, token_confidence). Deliberately excludes
    `posenc.Wr` (the rotary positional encoding basis) and `input_proj` when
    it's the identity (true for the "rdd" feature config) — those aren't part
    of the actual matching computation being adapted.

    Returns the number of Linear layers wrapped.
    """
    targets = [lg.transformers, lg.log_assignment, lg.token_confidence]
    if not isinstance(lg.input_proj, nn.Identity):
        targets.append(lg.input_proj)
    return sum(_apply_lora_recursive(root, rank) for root in targets)


# ── EMA ───────────────────────────────────────────────────────────────────────
def build_ema(model: nn.Module, device: torch.device) -> nn.Module:
    """Deep-copy `model` into a frozen, eval-mode EMA shadow."""
    ema = copy.deepcopy(_unwrap(model)).to(device)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model = _unwrap(model)
    for e_p, m_p in zip(ema.parameters(), model.parameters()):
        e_p.mul_(decay).add_(m_p.detach(), alpha=1 - decay)
    for e_b, m_b in zip(ema.buffers(), model.buffers()):
        e_b.copy_(m_b)


# ── training epoch ────────────────────────────────────────────────────────────
def train_epoch_lg(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    eval_lg: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader,
    mini_train_loader,
    mini_val_loader,
    epoch: int,
    total_epochs: int,
    args: argparse.Namespace,
    global_step: int,
    ema_lg: torch.nn.Module | None = None,
) -> tuple[float, int, dict | None]:
    """
    `eval_lg` is what mini-evals run against (the EMA shadow when --ema_decay >
    0, else `lg` itself); `lg` is always what receives gradient. `ema_lg` (same
    object as `eval_lg` when EMA is on, else None) is updated after every step.

    When --moving_negative_prob or --negative_mining is active, `loader`'s
    dataset was built with return_meta=True, so each batch carries a 4th
    `neg_meta` element; this opportunistically buckets that epoch's negatives'
    LG confidence by `neg_source` ("index" vs "random") into the returned
    dict, for run_training_lg to act on after the epoch — see NEG_GAP_EPS /
    compute_moving_prob / IndexAssignedTripletDataset.update_mining_stats.
    Returns None instead when neither flag is set.
    """
    train_rdd, train_lg = resolve_trained_models(args.trained_model)
    _unwrap(rdd).train(train_rdd)
    lg.train(train_lg)

    track_negatives = args.moving_negative_prob is not None or args.negative_mining
    mining_active   = args.negative_mining
    epoch_index_confs:  list[float] = []
    epoch_random_confs: list[float] = []
    epoch_mining_obs:   list[tuple[str, str, float]] = []

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
    for step, batch in pbar:
        if track_negatives:
            anchors, positives, negatives, neg_meta = batch
        else:
            anchors, positives, negatives = batch
        device = accelerator.device
        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)
        H_r, W_r = anchors_r.shape[-2:]

        # When RDD isn't being trained, no_grad purely skips building an unused graph.
        with contextlib.nullcontext() if train_rdd else torch.no_grad():
            feats_a = extract_train(rdd, anchors_r)
            feats_p = extract_train(rdd, positives_r)
            feats_n = extract_train(rdd, negatives_r)

        pred_pos, pred_neg, data_a, data_p, data_n = run_lg_matching_grad(lg, feats_a, feats_p, feats_n, H_r, W_r)

        loss, stats = lg_confidence_loss(
            pred_pos, pred_neg, args.lg_margin, device,
            data_a=data_a, data_p=data_p, data_n=data_n,
        )

        epoch_skipped += stats["n_skipped"]
        epoch_images  += len(feats_a)

        if track_negatives:
            with torch.no_grad():
                neg_conf_all = _lg_scores(pred_neg, data_a, data_n, device).tolist()
            for src, q_lynx, n_lynx, conf in zip(
                neg_meta["neg_source"], neg_meta["query_lynx"], neg_meta["neg_lynx"], neg_conf_all,
            ):
                if src == "index":
                    epoch_index_confs.append(conf)
                else:
                    epoch_random_confs.append(conf)
                    if mining_active:
                        epoch_mining_obs.append((q_lynx, n_lynx, conf))

        trainable_params = [p for p in _unwrap(lg).parameters() if p.requires_grad]
        if train_rdd:
            trainable_params += [p for p in _unwrap(rdd).parameters() if p.requires_grad]

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        if ema_lg is not None:
            update_ema(ema_lg, lg, args.ema_decay)

        loss_val    = loss.item()
        epoch_loss += loss_val
        global_step += 1

        progress = (epoch * steps_per_epoch + step + 1) / (total_epochs * steps_per_epoch)
        pbar.set_postfix(loss=f"{loss_val:.4f}", skip=stats["n_skipped"])

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss":                 loss_val,
                    "matches/mean_pos":           stats["mean_pos_matches"],
                    "matches/mean_neg":           stats["mean_neg_matches"],
                    "lg_confidence/mean_pos_conf": stats["mean_pos_conf"],
                    "lg_confidence/mean_neg_conf": stats["mean_neg_conf"],
                    "progress":                    progress,
                },
                step=global_step,
            )

        if (step + 1) % 100 == 0:
            t_mini_start = time.perf_counter()
            mini_train_m = eval_epoch(accelerator, rdd, eval_lg, mini_train_loader, args, prefix="mini_train")
            mini_val_m   = eval_epoch(accelerator, rdd, eval_lg, mini_val_loader,   args, prefix="mini_val")
            mini_eval_time += time.perf_counter() - t_mini_start
            _unwrap(rdd).train(train_rdd)
            lg.train(train_lg)
            if accelerator.is_main_process:
                accelerator.log({**mini_train_m, **mini_val_m, "progress": progress}, step=global_step)

    epoch_total_time = time.perf_counter() - t_epoch_start
    epoch_train_time = epoch_total_time - mini_eval_time
    skip_rate = epoch_skipped / max(epoch_images, 1)

    if accelerator.is_main_process:
        accelerator.log(
            {
                "train/skip_rate":  skip_rate,
                "train/n_skipped":  epoch_skipped,
                "time/train_s":     epoch_train_time,
                "time/mini_eval_s": mini_eval_time,
            },
            step=global_step,
        )

    neg_gap_stats = None
    if track_negatives:
        neg_gap_stats = {
            "mean_index_conf":  sum(epoch_index_confs)  / len(epoch_index_confs)  if epoch_index_confs  else 0.0,
            "mean_random_conf": sum(epoch_random_confs) / len(epoch_random_confs) if epoch_random_confs else 0.0,
            "n_index":  len(epoch_index_confs),
            "n_random": len(epoch_random_confs),
            "mining_observations": epoch_mining_obs,
        }

    return epoch_loss / max(steps_per_epoch, 1), global_step, neg_gap_stats


# ── full training run ─────────────────────────────────────────────────────────
def run_training_lg(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    accelerator = Accelerator(log_with="wandb" if args.project else None)
    device = accelerator.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    moving_active   = args.moving_negative_prob is not None
    mining_active   = args.negative_mining
    track_negatives = moving_active or mining_active

    # ── data ──
    train_transform, eval_transform = build_transforms(args.augment)
    train_ds = IndexAssignedTripletDataset(
        args.train_index, root=args.data_root, transform=train_transform,
        random_negative_prob=args.random_negative_prob,
        negative_mining=mining_active,
        negative_mining_temperature=args.negative_mining_temperature,
        negative_mining_decay=args.negative_mining_decay,
        return_meta=track_negatives,
    )
    # Diagnostics/eval on the training split must stay on clean, index-only
    # negatives, even when the actual training loader is augmented and/or
    # mixes in random negatives — otherwise train_eval/mini_train metrics
    # would be noisier than val's and not comparable across epochs.
    train_ds_eval = (
        IndexAssignedTripletDataset(args.train_index, root=args.data_root, transform=eval_transform)
        if (args.augment or args.random_negative_prob > 0) else train_ds
    )
    val_ds = IndexAssignedTripletDataset(args.val_index, root=args.data_root, transform=eval_transform)

    # persistent_workers=True (get_loader's default) would pickle train_ds into
    # long-lived worker processes once and never see it again — fatal for
    # --moving_negative_prob/--negative_mining, which mutate train_ds
    # (random_negative_prob, the mining EMA matrix) from the main process
    # between epochs. Disabling it means each epoch's fresh worker spawn
    # re-pickles the current state instead.
    train_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
        persistent_workers=(not track_negatives) and args.num_workers > 0,
    )

    _rng = random.Random(args.seed)

    def _fixed_subset(ds, fraction):
        n = max(1, int(len(ds) * fraction))
        return Subset(ds, _rng.sample(range(len(ds)), min(n, len(ds))))

    mini_train_loader = get_loader(
        _fixed_subset(train_ds_eval, 10 * args.batch_size / max(len(train_ds_eval), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    mini_val_loader = get_loader(
        _fixed_subset(val_ds, 10 * args.batch_size / max(len(val_ds), 1)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    # Video-level pseudo-accuracy needs every video's full set of query
    # frames present, so no subsetting here (unlike the mini-loaders above).
    eval_train_subset = train_ds_eval
    eval_val_subset   = val_ds

    # ── models ──
    train_rdd, train_lg = resolve_trained_models(args.trained_model)

    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    # detach_descriptors=False lets gradient reach RDD through LG's forward
    # pass (see rdd_patch/lightglue_masked.py); irrelevant when RDD is frozen.
    lg  = build_masked_lg(device, weights=args.lg_weights, detach_descriptors=not train_rdd)

    for p in rdd.parameters():
        p.requires_grad_(train_rdd)

    if args.lora:
        if not train_lg:
            raise ValueError("--lora requires --trained_model to include 'lg'")
        n_wrapped = apply_lora(_unwrap(lg), args.lora_rank)
        accelerator.print(f"[lora] wrapped {n_wrapped} Linear layers with rank-{args.lora_rank} adapters")

    if args.freeze_confidence_head:
        # After apply_lora (if used), so this also freezes any LoRA adapters
        # injected into these submodules — not just their frozen base weights.
        lg_unwrapped = _unwrap(lg)
        n_frozen = 0
        for submodule in (lg_unwrapped.log_assignment, lg_unwrapped.token_confidence):
            for p in submodule.parameters():
                p.requires_grad_(False)
                n_frozen += 1
        accelerator.print(f"[freeze_confidence_head] froze {n_frozen} parameters in log_assignment + token_confidence")

    if not train_lg:
        for p in _unwrap(lg).parameters():
            p.requires_grad_(False)

    # EMA shadow of LG's weights, built from its current (possibly LoRA-wrapped)
    # state, before accelerator.prepare wraps lg for DDP. Used for all eval below.
    ema_lg = build_ema(lg, device) if args.ema_decay > 0 else None

    trainable_params = [p for p in lg.parameters() if p.requires_grad]
    if train_rdd:
        trainable_params += [p for p in rdd.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # DDP-wrapping a module with no trainable parameters is unnecessary and can
    # error out under some Accelerate/PyTorch versions, so a frozen rdd is left
    # unprepared; it's only included here when --trained_model unfreezes it.
    if train_rdd:
        rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    else:
        lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    eval_lg = ema_lg if ema_lg is not None else lg

    if args.project:
        accelerator.init_trackers(
            args.project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name, "tags": build_wandb_tags(args)}},
        )

    # ── moving-negative baseline (before any training) ──
    baseline_ratio = None
    if moving_active:
        accelerator.print(
            f"[moving_negative] measuring pre-training index/random confidence gap "
            f"over {args.moving_negative_warmup_batches} batches..."
        )
        warmup_index_conf, warmup_random_conf = measure_negative_gap(
            accelerator, rdd, eval_lg, train_ds, args,
            n_batches=args.moving_negative_warmup_batches,
        )
        baseline_ratio = warmup_index_conf / (warmup_random_conf + NEG_GAP_EPS)
        train_ds.random_negative_prob = args.moving_negative_prob
        accelerator.print(
            f"[moving_negative] baseline nc_index={warmup_index_conf:.4f} "
            f"nc_random={warmup_random_conf:.4f} ratio={baseline_ratio:.4f} -> "
            f"random_negative_prob={train_ds.random_negative_prob}"
        )
        if accelerator.is_main_process:
            accelerator.log(
                {
                    "moving_neg/nc_index":  warmup_index_conf,
                    "moving_neg/nc_random": warmup_random_conf,
                    "moving_neg/ratio":     baseline_ratio,
                    "moving_neg/prob":      train_ds.random_negative_prob,
                },
                step=0,
            )

    # ── baseline eval (before any training) ──
    global_step = 0
    baseline_train = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_train_subset, args, prefix="train_eval")
    baseline_val   = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_val_subset,   args, prefix="val")
    _unwrap(rdd).train(train_rdd)
    lg.train(train_lg)
    if accelerator.is_main_process:
        accelerator.log({**baseline_train, **baseline_val, "epoch": -1}, step=global_step)

    # ── loop ──
    for epoch in range(args.epochs):
        epoch_loss, global_step, neg_gap_stats = train_epoch_lg(
            accelerator, rdd, lg, eval_lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
            ema_lg=ema_lg,
        )

        if moving_active and neg_gap_stats is not None and neg_gap_stats["n_random"] > 0:
            nc_index  = neg_gap_stats["mean_index_conf"]
            nc_random = neg_gap_stats["mean_random_conf"]
            ratio = nc_index / (nc_random + NEG_GAP_EPS)
            train_ds.random_negative_prob = compute_moving_prob(
                ratio, baseline_ratio, args.random_negative_prob,
                args.moving_negative_min_prob, args.moving_negative_max_prob,
            )
            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "moving_neg/nc_index":  nc_index,
                        "moving_neg/nc_random": nc_random,
                        "moving_neg/ratio":     ratio,
                        "moving_neg/prob":      train_ds.random_negative_prob,
                    },
                    step=global_step,
                )

        if mining_active and neg_gap_stats is not None and neg_gap_stats["mining_observations"]:
            train_ds.update_mining_stats(neg_gap_stats["mining_observations"])

        t_eval_start = time.perf_counter()
        do_eval = epoch % args.eval_every_epochs == args.eval_every_epochs - 1
        if do_eval:
            train_eval_metrics = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_train_subset, args, prefix="train_eval")
            val_metrics        = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_val_subset,   args, prefix="val")
        epoch_eval_time = time.perf_counter() - t_eval_start
        _unwrap(rdd).train(train_rdd)
        lg.train(train_lg)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":             epoch,
            "train/epoch_loss":  epoch_loss,
            "train/lr":          lr,
            "time/epoch_eval_s": epoch_eval_time,
        }
        if do_eval:
            metrics.update(train_eval_metrics)
            metrics.update(val_metrics)

        if accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)
            ckpt_dir = args.output_dir / f"epoch_{epoch:02d}"
            accelerator.save_state(str(ckpt_dir))
            if ema_lg is not None:
                torch.save(ema_lg.state_dict(), ckpt_dir / "ema_lg.pt")

    if args.project:
        accelerator.end_training()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_training_lg(args)


if __name__ == "__main__":
    main()
