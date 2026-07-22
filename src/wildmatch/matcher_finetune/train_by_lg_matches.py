from __future__ import annotations

import argparse
import random
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm
from torchvision import transforms

from torch.utils.data import Subset

from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_by_strong_matches_triplet_loss import matched_descriptor_loss
from contrastive_finetuning.train_common import (
    MatchTrendMonitor, _unwrap, add_common_args, build_wandb_tags, eval_epoch,
    eval_pseudo_accuracy, extract_train, resize_long_side, run_lg_matching_grad, seed_all,
)

"""
Alternative to train_by_strong_matches_{triplet,simclr}_loss.py: those scripts
treat LightGlue as a fixed, external oracle for evaluating matches — its
weights come from a separate training run and never adapt to the descriptor
distribution RDD is currently producing, which is itself a plausible
contributor to matches/mean_pos decay (on top of the descriptor-collapse
feedback loop the --ema_decay / --neg_grad_scale / --match_boost_weight
mechanisms address).

This script instead makes LightGlue itself trainable, via a margin loss on its
OWN matching confidence (`scores`, the per-match probability LG assigns to the
pairs it selects) rather than on raw descriptor dot products:
  --train_targets lg    RDD fully frozen; only LightGlue is fine-tuned.
  --train_targets both  LightGlue *and* RDD's descriptor network are
                         fine-tuned jointly (LightGlue via its confidence loss,
                         RDD via the same matched_descriptor_loss used by
                         train_by_strong_matches_triplet_loss.py, computed on
                         the same match indices).

LightGlueMasked always `.detach()`s its descriptor *inputs* internally (see
rdd_patch/lightglue_masked.py), so the LG-confidence loss can only ever reach
LightGlue's own transformer/assignment weights — it cannot backprop into RDD.
That's why --train_targets both needs a *separate* additive loss term for RDD,
rather than one loss magically training both.
"""


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune LightGlue's own matching weights (--train_targets lg), "
                     "or LightGlue + RDD jointly (--train_targets both), instead of "
                     "treating LightGlue as a fixed external match-evaluation oracle."
    )
    add_common_args(p)
    p.add_argument("--train_targets", choices=["lg", "both"], default="both")
    p.add_argument(
        "--margin", type=float, default=0.5,
        help="Margin for the RDD raw-descriptor triplet loss (only used with --train_targets both)",
    )
    p.add_argument("--lg_margin", type=float, default=0.5, help="Margin for the LightGlue match-confidence loss")
    p.add_argument(
        "--lg_loss_weight", type=float, default=1.0,
        help="Weight of the LG-confidence loss relative to the RDD descriptor loss (only used with --train_targets both)",
    )
    args = p.parse_args()

    # add_common_args also defines --ema_decay / --match_boost_weight (for the
    # RDD-only scripts' anti-collapse mechanisms), but this script only wires
    # up --neg_grad_scale for the RDD half of --train_targets both. Fail loudly
    # rather than silently ignoring them if someone reuses a config that sets these.
    if args.ema_decay > 0:
        p.error("--ema_decay is not implemented in train_lg_matching_loss.py (no EMA-teacher decoupling here — LightGlue itself is being trained directly)")
    if args.match_boost_weight > 0:
        p.error("--match_boost_weight is not implemented in train_lg_matching_loss.py")
    return args


# ── loss ──────────────────────────────────────────────────────────────────────
def lg_confidence_loss(
    pred_pos: dict, pred_neg: dict, margin: float, device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    Margin loss on LightGlue's OWN matching confidence (`scores`), mirroring
    matched_descriptor_loss's structure but with LG's per-match probability in
    place of raw descriptor cosine similarity:
      loss = relu(margin - mean(pos_conf) + mean(neg_conf))
    """
    losses = []
    n_skipped = 0
    pos_match_list, neg_match_list = [], []
    pos_conf_list, neg_conf_list = [], []

    for s_pos, s_neg in zip(pred_pos["scores"], pred_neg["scores"]):
        if s_pos.shape[0] == 0:
            n_skipped += 1
            continue

        pos_match_list.append(s_pos.shape[0])
        neg_match_list.append(s_neg.shape[0])
        pos_conf_list.append(s_pos.mean().item())

        if s_neg.shape[0] > 0:
            neg_conf_list.append(s_neg.mean().item())
            losses.append(F.relu(margin - s_pos.mean() + s_neg.mean()))
        else:
            losses.append(F.relu(margin - s_pos).mean())

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


# ── training epoch ────────────────────────────────────────────────────────────
def train_epoch_lg(
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
    match_monitor: MatchTrendMonitor | None = None,
) -> tuple[float, int]:
    """
    `match_monitor` (built when --train_targets both and --neg_grad_scale !=
    1.0) mirrors train_common.train_epoch's dynamic gating: --neg_grad_scale
    only takes effect once matches/mean_pos drops below its post-warm-up norm,
    fading back to 1.0 (unchanged) as it recovers — see MatchTrendMonitor.
    """
    train_rdd = args.train_targets == "both"

    def _set_modes():
        if train_rdd:
            rdd.train()
        else:
            _unwrap(rdd).eval()
        lg.train()

    _set_modes()
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

        # RDD is only gradient-tracked when it's a training target — frozen
        # runs use no_grad purely to skip building an unused graph.
        extract_ctx = torch.enable_grad() if train_rdd else torch.no_grad()
        with extract_ctx:
            feats_a = extract_train(rdd, anchors_r)
            feats_p = extract_train(rdd, positives_r)
            feats_n = extract_train(rdd, negatives_r)

        pred_pos, pred_neg = run_lg_matching_grad(lg, feats_a, feats_p, feats_n, H_r, W_r)

        lg_loss, lg_stats = lg_confidence_loss(pred_pos, pred_neg, args.lg_margin, device)

        boost_activation = 0.0
        if match_monitor is not None:
            mean_pos_now = sum(len(m) for m in pred_pos["matches"]) / max(len(pred_pos["matches"]), 1)
            boost_activation = match_monitor.update(mean_pos_now)
        effective_neg_grad_scale = 1.0 - (1.0 - args.neg_grad_scale) * boost_activation

        if train_rdd:
            rdd_loss, rdd_stats = matched_descriptor_loss(
                feats_a, feats_p, feats_n,
                pred_pos["matches"], pred_neg["matches"],
                args.margin,
                neg_grad_scale=effective_neg_grad_scale,
            )
            loss = rdd_loss + args.lg_loss_weight * lg_loss
        else:
            rdd_loss, rdd_stats = None, None
            loss = lg_loss

        epoch_skipped += lg_stats["n_skipped"]
        epoch_images  += len(feats_a)

        trainable_params = [p for p in _unwrap(lg).parameters() if p.requires_grad]
        if train_rdd:
            trainable_params += [p for p in _unwrap(rdd).parameters() if p.requires_grad]

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        loss_val    = loss.item()
        epoch_loss += loss_val
        global_step += 1

        progress = (epoch * steps_per_epoch + step + 1) / (total_epochs * steps_per_epoch)
        pbar.set_postfix(loss=f"{loss_val:.4f}", skip=lg_stats["n_skipped"])

        if accelerator.is_main_process:
            log = {
                "train/loss":                    loss_val,
                "train/lg_loss":                  lg_loss.item(),
                "matches/mean_pos":               lg_stats["mean_pos_matches"],
                "matches/mean_neg":               lg_stats["mean_neg_matches"],
                "lg_confidence/mean_pos_conf":    lg_stats["mean_pos_conf"],
                "lg_confidence/mean_neg_conf":    lg_stats["mean_neg_conf"],
                "progress":                       progress,
            }
            if train_rdd:
                log["train/rdd_loss"]       = rdd_loss.item()
                log["matches/mean_pos_sim"] = rdd_stats["mean_pos_sim"]
                log["matches/mean_neg_sim"] = rdd_stats["mean_neg_sim"]
                log["train/mean_na"]        = rdd_stats["mean_na"]
                log["mechanisms/neg_grad_scale"] = effective_neg_grad_scale
                if match_monitor is not None:
                    log["mechanisms/boost_activation"] = boost_activation
                    log["mechanisms/boost_activation_freq"] = match_monitor.activation_freq
                    if match_monitor.norm is not None:
                        log["mechanisms/boost_norm"] = match_monitor.norm
            accelerator.log(log, step=global_step)

        if (step + 1) % 100 == 0:
            t_mini_start = time.perf_counter()
            mini_train_m = eval_epoch(accelerator, rdd, lg, mini_train_loader, args, prefix="mini_train")
            mini_val_m   = eval_epoch(accelerator, rdd, lg, mini_val_loader,   args, prefix="mini_val")
            mini_eval_time += time.perf_counter() - t_mini_start
            _set_modes()
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

    return epoch_loss / max(steps_per_epoch, 1), global_step


# ── full training run ─────────────────────────────────────────────────────────
def run_training_lg(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    accelerator = Accelerator(log_with="wandb" if args.project else None)
    device = accelerator.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    transform = transforms.ToTensor()
    train_ds = IndexAssignedTripletDataset(args.train_index, root=args.data_root, transform=transform)
    val_ds   = IndexAssignedTripletDataset(args.val_index,   root=args.data_root, transform=transform)

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

    train_rdd = args.train_targets == "both"
    if not train_rdd:
        for p in rdd.parameters():
            p.requires_grad_(False)

    trainable_params = [p for p in lg.parameters() if p.requires_grad]
    if train_rdd:
        trainable_params += [p for p in rdd.parameters() if p.requires_grad]

    # See train_epoch_lg / train_common.MatchTrendMonitor: --neg_grad_scale only
    # takes effect once matches/mean_pos drops below its post-warm-up norm.
    # (--match_boost_weight isn't wired up here — parse_args() rejects it.)
    match_monitor = (
        MatchTrendMonitor(
            warmup_steps=int(args.match_boost_warmup_epochs * len(train_loader)),
            window=args.match_boost_window,
            drop_frac=args.match_boost_drop_frac,
            freq_decay=args.match_boost_freq_decay,
        )
        if (train_rdd and args.neg_grad_scale != 1.0) else None
    )

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if train_rdd:
        rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    else:
        # rdd is fully frozen (requires_grad=False everywhere) — DDP-wrapping a
        # module with no trainable parameters is unnecessary and can error out
        # under some Accelerate/PyTorch versions, so it's left unprepared.
        lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )

    if args.project:
        accelerator.init_trackers(
            args.project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name, "tags": build_wandb_tags(args)}},
        )

    # ── baseline eval (before any training) ──
    global_step = 0
    baseline_train = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
    baseline_val   = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
    if train_rdd:
        rdd.train()
    else:
        _unwrap(rdd).eval()
    lg.train()
    if accelerator.is_main_process:
        accelerator.log({**baseline_train, **baseline_val, "epoch": -1}, step=global_step)

    # ── loop ──
    for epoch in range(args.epochs):
        epoch_loss, global_step = train_epoch_lg(
            accelerator, rdd, lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
            match_monitor=match_monitor,
        )

        t_eval_start = time.perf_counter()
        train_eval_metrics = eval_pseudo_accuracy(accelerator, rdd, lg, eval_train_subset, args, prefix="train_eval")
        val_metrics        = eval_pseudo_accuracy(accelerator, rdd, lg, eval_val_subset,   args, prefix="val")
        epoch_eval_time = time.perf_counter() - t_eval_start
        if train_rdd:
            rdd.train()
        else:
            _unwrap(rdd).eval()
        lg.train()

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":             epoch,
            "train/epoch_loss":  epoch_loss,
            "train/lr":          lr,
            "time/epoch_eval_s": epoch_eval_time,
            **train_eval_metrics,
            **val_metrics,
        }

        if accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)
            accelerator.save_state(str(args.output_dir / f"epoch_{epoch:02d}"))

    if args.project:
        accelerator.end_training()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_training_lg(args)


if __name__ == "__main__":
    main()
