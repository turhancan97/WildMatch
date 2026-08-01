from __future__ import annotations

import argparse
import contextlib
import copy
import math
import random
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import gather_object
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms

from torch.utils.data import Subset

from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_rdd, build_masked_lg
from contrastive_finetuning.train_common import (
    _extract_chunked, _lg_scores, _unwrap, add_common_args, batch_features,
    build_pseudo_accuracy_loader, build_wandb_tags, eval_epoch, eval_pseudo_accuracy,
    extract_train, resize_long_side, resolve_trained_models, seed_all,
)

"""
Trains via a margin loss on LightGlue's OWN matching confidence (`scores`,
the per-match probability LG assigns to the pairs it selects) rather than on
raw descriptor dot products — i.e. treating LG's own matches as the training
signal instead of a fixed, external oracle for evaluating matches (as a
descriptor-training setup would).

`--trained_model` (lg / rdd / lg+rdd) decides which model(s) are unfrozen and
receive gradient; the other stays frozen. LightGlueForTraining detaches its
descriptor *inputs* by default (see `detach_descriptors` in
rdd_patch/lightglue_masked_training.py), which is what normally keeps this loss from
backpropagating into RDD; when `--trained_model` includes 'rdd', LG is built
with `detach_descriptors=False` so gradient can reach RDD too.

Independent, combinable anti-overfitting mechanisms, each off by default:
  --augment    photometric-only data augmentation on the training images
               (never on val — keypoint geometry is untouched, detection runs
               after augmentation, so correspondences stay valid). Note that
               with RDD frozen (--trained_model lg, the usual case) a
               photometric-augmentation-invariant RDD absorbs most of this
               before LightGlue ever sees it — the feature-space perturbations
               below (--keypoint_dropout, --multi_scale_*) are the ones that
               are guaranteed to reach the matcher.
  --margin_activation softplus
               replaces the hinge's relu with softplus(beta*z)/beta so
               triplets already past the margin keep a small, exponentially-
               decaying gradient instead of dropping out of the loss — the
               direct counter to the signal saturating once train accuracy
               approaches 1. train/active_frac (logged always) is the gauge:
               the fraction of triplets still inside the margin.
  --adaptive_margin
               raises the margin between epochs to track the model's own
               measured pos/neg separation gap (clamped to
               [--lg_margin, --adaptive_margin_max]), so the margin recedes
               ahead of the model instead of being cleared once and forever.
  --num_negatives K
               K independently-sampled negatives per triplet, combined with a
               smooth max — each step effectively trains against the hardest
               negative it sampled, which keeps the negative term alive long
               after the average negative is solved.
  --keypoint_dropout / --multi_scale_min/--multi_scale_max
               training-only perturbations of LightGlue's actual input space
               (random keypoint deletion; per-step random resize target) —
               regularization that cannot be absorbed by a frozen RDD, unlike
               image-level photometric jitter.
  --hard_positive_sampling / --hard_negative_sampling
               replace uniform draws from each entry's fixed positive/negative
               pools with draws weighted toward historically hard pairs (low
               positive confidence / high negative confidence), using
               confidences the training loss already computes — a cheap,
               forward-pass-free stand-in for re-mining the index as the model
               solves the easy candidates.
  --frame_jitter_query / --frame_jitter_db (+ --frame_jitter_prob)
               temporal augmentation: substitute an index frame with a
               neighbour from the same video. The index samples ~20 frames per
               video while every frame is on disk, so this reaches ~32x more
               material at zero extra compute, and a neighbouring frame is the
               same individual by construction — no gradient masking needed,
               unlike --weak_queries' uncurated cross-video pairing. Aimed at
               the positive term specifically: the negative side already draws
               from the whole frame pool via --random_negative_prob, while the
               positive stays locked to the index's handful of candidates,
               which is where the saturation concentrates. Costs a raised
               zero-match rate (a neighbour was never retrieval-verified —
               watch train/skip_rate_pos_index), which is what the
               correspondence / healing_on_positives distillation modes exist
               to catch.
  --negative_mining_frame_prob
               frame-level sharpening of --negative_mining's lynx-level
               matrix: remembers specific frames that produced high negative
               confidence and re-serves them, again at zero extra forward
               cost.
  --lora / --lora_rank
               freeze LightGlue's own pretrained weights and train only
               low-rank adapters injected into its attention/assignment
               Linear layers — drastically fewer trainable params than full
               fine-tuning. The rotary positional encoding (posenc.Wr) is
               deliberately left un-adapted.
  --ema_decay  exponential moving average of LightGlue's weights, used for all
               evaluation/checkpointing instead of the raw (noisier) live
               weights — smooths over late-training variance/overfitting.

--distill_model / --distill_model_lambda / --distill_signal_type / --distill_loss
    Adds a consistency loss pulling whichever model(s) --trained_model unfreezes
    back toward a reference, added to the margin loss as
    `total_loss = loss + distill_model_lambda * consistency_loss`.
      --distill_model pretrained  anchors to a frozen snapshot of the model(s)
                                   as they are at the start of this run.
      --distill_model ema         anchors to an exponential moving average of
                                   the student's own weights instead, which
                                   drifts slowly away from the pretrained start
                                   rather than staying fixed there; its decay
                                   (see compute_distill_ema_decay) is chosen
                                   from --epochs so it always has room to move
                                   by the end of training, however long that is.
      --distill_signal_type weights      L1/L2 distance between the trained
                                          model's own parameters and the
                                          reference's — a direct anchor,
                                          independent of any particular batch.
                                          Applied via direct gradient
                                          injection after backward(), not as
                                          part of the batch's autograd graph
                                          — see accumulate_distill_weights_grad,
                                          required for multi-GPU DDP safety
                                          against LightGlue's always-dead
                                          token_confidence/log_assignment
                                          parameters (find_unused_parameters,
                                          below).
      --distill_signal_type activations  L1/L2 distance between LightGlue's
                                          per-layer descriptor embeddings (the
                                          residual stream self_attn/cross_attn
                                          write into and log_assignment reads
                                          to score matches) computed by the
                                          student vs. the reference LightGlue
                                          on the exact same input keypoints/
                                          descriptors this step already
                                          extracted — a check on the
                                          *matching-relevant* representation,
                                          not on attention weights or Q/K/V,
                                          which have no stable shape/pairing to
                                          compare across a training run and
                                          don't directly feed the loss the way
                                          the embeddings LightGlue leaves
                                          behind at each layer do. Because both
                                          sides see the same input, this is
                                          only meaningful when 'lg' is in
                                          --trained_model (validated below).
                                          By default both the (anchor,
                                          positive) and (anchor, negative)
                                          pair are matched against the
                                          reference, which spends half the
                                          anchor's budget holding the
                                          *negative* pair's representation
                                          where the pretrained model had it —
                                          directly opposing the margin loss's
                                          negative term, the one part of the
                                          objective that is supposed to move.
                                          --activations_on_positives narrows
                                          the comparison to the positive pair
                                          alone, and skips the reference's
                                          negative forward entirely.
      --distill_signal_type correspondence
                                          Targeted rescue, not a general
                                          regularizer: only touches samples
                                          the student currently reports zero
                                          positive matches for (pos_empty —
                                          see lg_confidence_loss). For those
                                          samples only, the reference's own
                                          matches0/valid0 on the exact same
                                          (anchor, positive) pair become
                                          pseudo-labels, and an NLL loss pulls
                                          the student's raw, pre-filter
                                          assignment_scores (added to
                                          LightGlueForTraining's output dict
                                          specifically for this — see
                                          rdd_patch/lightglue_masked_training.py)
                                          toward those positions. This reaches
                                          the actual matching decision
                                          (log_assignment) the way
                                          'activations' cannot (that mode only
                                          touches the transformer backbone,
                                          upstream of log_assignment), while
                                          staying per-sample the way 'weights'
                                          cannot (that mode is pure parameter
                                          distance, blind to which pair is
                                          dead). A healthy pair's gradient is
                                          untouched by this, and so is a
                                          --weak_queries sample even with an
                                          empty match set — same distrust as
                                          lg_confidence_loss's weak_mask,
                                          since the "pretrained matched every
                                          index positive" premise doesn't
                                          extend to weak_queries' uncurated
                                          pairing. --distill_loss
                                          doesn't apply (NLL, not a distance).
                                          Also requires 'lg' in
                                          --trained_model. Works with either
                                          --distill_model, but 'pretrained' is
                                          the better fit for rescuing dead
                                          pairs specifically: 'ema' drifts
                                          toward the student's own live
                                          trajectory — the same one that
                                          produced the collapse — while
                                          'pretrained' stays fixed at the
                                          point where every index positive
                                          matched.
      --distill_signal_type healing_on_positives
                                          A wider-gated variant of
                                          'correspondence': same pseudo-labels
                                          (the reference's matches0/valid0 on
                                          the same (anchor, positive) pair),
                                          same NLL on the student's
                                          assignment_scores, same exemption
                                          for --weak_queries samples. What
                                          changes is *when* it fires. Instead
                                          of waiting for a positive pair to
                                          match nothing at all, it fires as
                                          soon as the student's live
                                          confidence for that pair (_lg_scores,
                                          the same number eval ranks on) drops
                                          below what the PRETRAINED model
                                          scored for that exact (query,
                                          positive) pair. Zero matches is just
                                          the far end of that condition, so
                                          this is a strict superset of
                                          'correspondence' — it starts pulling
                                          a pair back while it is merely
                                          degrading rather than after it has
                                          died, which is also when the pull is
                                          small enough not to be a shock.
                                          The per-pair reference scores are
                                          measured once, before training, by
                                          measure_pretrained_positive_scores;
                                          a run revisits each (query, positive)
                                          pair on the order of
                                          epochs/n_positives times, so one
                                          clean pass up front is both far
                                          cheaper than re-scoring it every step
                                          and a *fixed* target rather than a
                                          moving one. That target is the
                                          pretrained model's regardless of
                                          --distill_model (which still chooses
                                          where the pseudo-label matches come
                                          from): the gate asks "am I worse than
                                          where I started", which an EMA
                                          reference — drifting along with the
                                          student — cannot answer. Note with
                                          --augment the reference is measured
                                          on clean images while training sees
                                          jittered ones, which biases the gate
                                          slightly towards firing. Requires
                                          'lg' in --trained_model.
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
        "--margin_activation", type=str, default="relu", choices=["relu", "softplus"],
        help="Activation over the margin violation z = margin - pos_conf + neg_conf: "
             "'relu' (default) is the classic hinge — exactly zero gradient for every "
             "triplet already past the margin, so the training signal dies off as the "
             "training set gets solved; 'softplus' (softplus(beta*z)/beta, see "
             "--softplus_beta) keeps an exponentially-decaying but never-zero gradient "
             "on solved triplets, so they keep anchoring the model instead of dropping "
             "out of the loss entirely.",
    )
    p.add_argument(
        "--softplus_beta", type=float, default=10.0,
        help="Sharpness of --margin_activation softplus; higher hugs relu more closely "
             "(smaller gradient leak on solved triplets).",
    )
    p.add_argument(
        "--adaptive_margin", action="store_true",
        help="Raise the margin between epochs as the model separates positives from "
             "negatives: next epoch's margin = clamp(mean(pos_conf - neg_conf) + "
             "--adaptive_margin_offset, --lg_margin, --adaptive_margin_max), where the "
             "mean separation gap is measured over this epoch's training batches "
             "(kept samples only, all ranks). Keeps roughly the below-average half of "
             "triplets active instead of letting the whole set clear a fixed margin. "
             "Logged as train/margin and train/sep_gap.",
    )
    p.add_argument(
        "--adaptive_margin_max", type=float, default=0.9,
        help="Ceiling for --adaptive_margin (confidences live in [0, 1], so a margin "
             "close to 1 is unreachable by construction).",
    )
    p.add_argument(
        "--adaptive_margin_offset", type=float, default=0.0,
        help="Added to the measured separation gap before clamping (--adaptive_margin). "
             "Positive keeps more triplets active; negative fewer.",
    )
    p.add_argument(
        "--num_negatives", type=int, default=1,
        help="Negatives sampled per triplet (each drawn independently through the full "
             "sampling stack: index/random mix, mining, hard sampling). With K > 1 the "
             "loss trains against a smooth max (tau*logsumexp(conf/tau), see "
             "--multi_neg_tau) of the K negatives' confidences — effectively always "
             "the hardest negative sampled this step, which saturates far slower than "
             "one random negative. Multiplies the negative-side forward cost by K.",
    )
    p.add_argument(
        "--multi_neg_tau", type=float, default=0.1,
        help="Smooth-max temperature over the K negatives' confidences "
             "(--num_negatives > 1); lower approaches a hard max.",
    )
    p.add_argument(
        "--keypoint_dropout", type=float, default=0.0,
        help="Training-only probability of dropping each detected keypoint (with its "
             "descriptor) before matching. Perturbs LightGlue's actual input space — "
             "unlike photometric image augmentation, which a frozen, "
             "augmentation-invariant RDD largely absorbs before LG ever sees it — so "
             "the matcher can't lean on memorized keypoint constellations. Never "
             "applied in eval.",
    )
    p.add_argument(
        "--multi_scale_min", type=int, default=0,
        help="Training-only multi-scale: per step, the resize target is drawn "
             "uniformly from [--multi_scale_min, --multi_scale_max] snapped to /32, "
             "instead of the fixed --resize (which eval always keeps). 0 (default) "
             "disables. Varies keypoint count/density and detection scale per step.",
    )
    p.add_argument(
        "--multi_scale_max", type=int, default=0,
        help="Upper bound for --multi_scale_min's range; both must be set together.",
    )
    p.add_argument(
        "--hard_positive_sampling", action="store_true",
        help="Sample each entry's positive weighted toward historically LOW LG "
             "confidence (hard positives) instead of uniformly, from a per-"
             "(query_frame, candidate_frame) EMA fed once per epoch with the "
             "confidences the training loss already computed — no extra forward "
             "passes (see --hard_pair_temperature / --hard_pair_decay).",
    )
    p.add_argument(
        "--hard_negative_sampling", action="store_true",
        help="Same for the entry's index-mined negatives, weighted toward "
             "historically HIGH LG confidence — the cheap substitute for re-mining "
             "the index: the fixed top_m negatives stop being drawn uniformly once "
             "some are solved. Only affects the index branch; the random branch "
             "stays --negative_mining's territory.",
    )
    p.add_argument(
        "--hard_pair_temperature", type=float, default=0.1,
        help="Softmax temperature over EMA confidences for --hard_positive_sampling / "
             "--hard_negative_sampling / --negative_mining_frame_prob; lower "
             "concentrates on the hardest known candidate.",
    )
    p.add_argument(
        "--hard_pair_decay", type=float, default=0.9,
        help="EMA decay for the per-pair confidence memory behind "
             "--hard_positive_sampling / --hard_negative_sampling.",
    )
    p.add_argument(
        "--frame_jitter_query", type=int, default=0,
        help="Temporal augmentation on the QUERY side: with probability "
             "--frame_jitter_prob, replace the index's query frame with one up to this "
             "many positions away in its own video (uniform over [-k, k], clamped to "
             "the video's frame range). The index samples ~20 frames per video but "
             "every frame is on disk (~32x more), and a neighbouring frame is the same "
             "individual by construction — free, exactly-labelled data the index never "
             "reaches. 0 (default) disables. Never applied to eval, nor to "
             "--weak_queries samples (already uniform draws over the whole pool).",
    )
    p.add_argument(
        "--frame_jitter_db", type=int, default=0,
        help="Same, for the candidate/database side: the positive and any index-mined "
             "negative (a --random_negative_prob negative is already drawn from the "
             "whole pool, so it's left alone). Combine with --frame_jitter_query for "
             "two-sided jitter.",
    )
    p.add_argument(
        "--frame_jitter_prob", type=float, default=1.0,
        help="Probability that an eligible frame is jittered at all; otherwise the "
             "index frame is used unchanged. Below 1 the batch mixes exact index pairs "
             "with jittered ones, which preserves part of the index's curated "
             "difficulty (positives are top_k retrieved, negatives top_m hard-mined — "
             "a neighbour was never retrieved and is on average easier) and keeps "
             "feeding the pair-level EMAs real index-pair observations.",
    )
    p.add_argument(
        "--negative_mining_frame_prob", type=float, default=0.0,
        help="Frame-level upgrade to --negative_mining (required): probability that, "
             "after the mined lynx is chosen, the specific frame is drawn from a "
             "per-frame EMA confidence memory (softmax toward historically-confusing "
             "frames) instead of uniformly from that lynx's pool. Fed opportunistically "
             "from training batches, so it costs no extra forwards — addresses "
             "lynx-level mining being too coarse to surface strong candidates.",
    )
    p.add_argument(
        "--warmup_steps", type=int, default=0,
        help="Linearly ramp the LR from 0 to --lr over this many optimizer steps "
             "at the start of training (0, the default, disables warmup). Applied "
             "per-step inside train_epoch_lg (see apply_warmup_lr), overriding "
             "whatever the per-epoch CosineAnnealingLR scheduler set for that step "
             "— without this, epoch 0 runs at the full --lr from step 0. Once "
             "warmup_steps optimizer steps have elapsed, the cosine schedule "
             "(unchanged, still stepped once per epoch) takes over uninterrupted.",
    )
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
    p.add_argument(
        "--weak_queries", action="store_true",
        help="With probability --weak_queries_prob, replace the index-driven query for a "
             "sample with a fully random frame from the whole candidate pool (any lynx), "
             "paired with a random same-lynx positive and a random different-lynx negative "
             "— there's no index entry to mine a pool from for these, so both are plain "
             "uniform draws (independent of --negative_mining, which only biases the "
             "index-query random-negative branch; --random_negative_prob's index-vs-random "
             "coin flip doesn't apply here either — weak queries are always fully random). "
             "Because the 'positive' pairing isn't curated, its contribution to the margin "
             "loss gradient is masked (forward value unchanged, like .detach(), but "
             "per-sample since a batch mixes weak and index queries — see lg_confidence_loss "
             "weak_mask); the negative-confidence term still trains normally. Requires "
             "--weak_queries_prob > 0.",
    )
    p.add_argument(
        "--weak_queries_prob", type=float, default=0.0,
        help="Probability of drawing a --weak_queries sample instead of the index-driven "
             "query, per training item.",
    )
    p.add_argument(
        "--distill_model", type=str, default="none", choices=["none", "pretrained", "ema"],
        help="Adds a consistency loss pulling the trained model(s) (per --trained_model) "
             "back toward a reference: 'pretrained' anchors to a frozen snapshot taken at "
             "the start of this run; 'ema' anchors to an exponential moving average of the "
             "student's own weights (decay from --distill_ema_decay, auto-derived from "
             "--epochs if unset). 'none' (default) disables distillation. See module "
             "docstring for the full picture and --distill_signal_type for what's compared.",
    )
    p.add_argument(
        "--distill_model_lambda", type=float, default=0.0,
        help="Weight of the distillation consistency loss: "
             "total_loss = loss + distill_model_lambda * consistency_loss. Required > 0 "
             "when --distill_model is not 'none'. Tune separately per --distill_signal_type "
             "— 'correspondence' is an NLL over assignment log-probabilities, a different "
             "scale than the L1/L2 distances 'weights'/'activations' produce.",
    )
    p.add_argument(
        "--distill_signal_type", type=str, default="weights",
        choices=["weights", "activations", "correspondence", "healing_on_positives"],
        help="What the consistency loss is computed on: 'weights' is an L1/L2 distance "
             "between the trained model's parameters and the reference's; 'activations' "
             "matches LightGlue's per-layer descriptor embeddings between student and "
             "reference on this step's own keypoints/descriptors; 'correspondence' is a "
             "targeted rescue that only fires for samples whose positive pair the student "
             "currently matches nothing on, pulling it towards the reference's own matches "
             "for that pair; 'healing_on_positives' is the same rescue on a wider gate — "
             "it fires for any index positive the student now scores below the pretrained "
             "model's score for that same pair, of which a zero-match pair is the extreme "
             "case. The last two require 'lg' in --trained_model — see module docstring "
             "for all four.",
    )
    p.add_argument(
        "--activations_on_positives", action="store_true",
        help="With --distill_signal_type activations, match only the (anchor, positive) "
             "pair's embeddings against the reference instead of averaging positive and "
             "negative pairs. The default (both) spends half the anchor's budget pinning "
             "the negative pair's representation to the pretrained model, which works "
             "against the margin loss's negative term; this also skips the reference's "
             "negative forward pass.",
    )
    p.add_argument(
        "--distill_loss", type=str, default="l2", choices=["l1", "l2"],
        help="Distance function for the distillation consistency loss (weights or "
             "activations).",
    )
    p.add_argument(
        "--distill_ema_decay", type=float, default=None,
        help="EMA decay for --distill_model ema's reference. Defaults to a value derived "
             "from --epochs and the training set size so the reference's memory of the "
             "pretrained starting point decays to ~1%% of its original weight by the end "
             "of training (see compute_distill_ema_decay); set explicitly to override.",
    )
    args = p.parse_args()
    if args.moving_negative_prob is not None:
        if args.random_negative_prob <= 0:
            p.error("--moving_negative_prob requires --random_negative_prob > 0")
        if not (0.0 <= args.moving_negative_prob <= 1.0):
            p.error("--moving_negative_prob must be in [0, 1]")
    if args.negative_mining and args.random_negative_prob <= 0:
        p.error("--negative_mining requires --random_negative_prob > 0")
    if args.weak_queries and not (0.0 < args.weak_queries_prob <= 1.0):
        p.error("--weak_queries requires --weak_queries_prob in (0, 1]")
    if args.distill_model != "none":
        if args.distill_model_lambda <= 0:
            p.error("--distill_model requires --distill_model_lambda > 0")
        if (
            args.distill_signal_type in ("activations", "correspondence", "healing_on_positives")
            and "lg" not in args.trained_model.split("+")
        ):
            p.error(
                f"--distill_signal_type {args.distill_signal_type} requires --trained_model "
                "to include 'lg': with only 'rdd' trained, LightGlue's own weights never "
                "change, so its output on a fixed input would never drift from the reference"
            )
    elif args.distill_model_lambda > 0:
        p.error("--distill_model_lambda requires --distill_model to be 'pretrained' or 'ema'")
    if args.activations_on_positives and args.distill_signal_type != "activations":
        p.error("--activations_on_positives only applies to --distill_signal_type activations")
    if args.distill_ema_decay is not None and not (0.0 < args.distill_ema_decay < 1.0):
        p.error("--distill_ema_decay must be in (0, 1)")
    if args.warmup_steps < 0:
        p.error("--warmup_steps must be >= 0")
    if args.num_negatives < 1:
        p.error("--num_negatives must be >= 1")
    if args.multi_neg_tau <= 0:
        p.error("--multi_neg_tau must be > 0")
    if args.softplus_beta <= 0:
        p.error("--softplus_beta must be > 0")
    if not (0.0 <= args.keypoint_dropout < 1.0):
        p.error("--keypoint_dropout must be in [0, 1)")
    if (args.multi_scale_min > 0) != (args.multi_scale_max > 0):
        p.error("--multi_scale_min and --multi_scale_max must be set together")
    if args.multi_scale_max > 0 and not (32 <= args.multi_scale_min <= args.multi_scale_max):
        p.error("--multi_scale_min must be >= 32 and <= --multi_scale_max")
    if args.adaptive_margin and args.adaptive_margin_max < args.lg_margin:
        p.error("--adaptive_margin_max must be >= --lg_margin")
    if not (0.0 <= args.negative_mining_frame_prob <= 1.0):
        p.error("--negative_mining_frame_prob must be in [0, 1]")
    if args.negative_mining_frame_prob > 0 and not args.negative_mining:
        p.error("--negative_mining_frame_prob requires --negative_mining")
    if args.hard_pair_temperature <= 0:
        p.error("--hard_pair_temperature must be > 0")
    if not (0.0 < args.hard_pair_decay < 1.0):
        p.error("--hard_pair_decay must be in (0, 1)")
    if args.frame_jitter_query < 0 or args.frame_jitter_db < 0:
        p.error("--frame_jitter_query/--frame_jitter_db must be >= 0")
    if not (0.0 <= args.frame_jitter_prob <= 1.0):
        p.error("--frame_jitter_prob must be in [0, 1]")
    if (args.frame_jitter_query > 0 or args.frame_jitter_db > 0) and args.frame_jitter_prob == 0:
        p.error("--frame_jitter_prob 0 silently disables --frame_jitter_query/--frame_jitter_db")
    return args


# ── loss ──────────────────────────────────────────────────────────────────────
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean of `x` over `mask`, 0.0 when the mask selects nothing."""
    n = int(mask.sum())
    return float((x * mask).sum() / n) if n else 0.0


def lg_confidence_loss(
    pred_pos: dict, pred_neg: dict, margin: float,
    data_a: dict, data_p: dict, data_n: dict, weak_mask: torch.Tensor | None = None,
    data_a_neg: dict | None = None, num_negatives: int = 1, multi_neg_tau: float = 0.1,
    margin_activation: str = "relu", softplus_beta: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """
    Margin loss on LightGlue's OWN matching confidence:
      loss = act(margin - pos_conf + neg_conf)
    where act is relu (default) or a scaled softplus (--margin_activation:
    softplus(beta*z)/beta keeps a small, exponentially-decaying gradient on
    triplets already past the margin instead of relu's exact zero — the
    anti-saturation escape hatch for a mostly-solved training set).

    With num_negatives=K > 1, pred_neg/data_n cover a flat (B*K) batch of
    negative pairs (data_a_neg is the anchor features repeated per negative;
    defaults to data_a for K == 1) and neg_conf is a smooth max —
    tau*logsumexp(conf/tau) — over each sample's K negatives, so the margin is
    enforced against (approximately) the hardest negative sampled this step.

    pos_conf/neg_conf are sum(confidence) / min(valid keypoints on each
    side) — the same normalization eval_pseudo_accuracy uses (see
    _lg_scores in train_common.py) — which keeps a couple of lucky
    high-confidence matches from dominating the loss for an otherwise
    poorly-matched pair.

    Fully vectorized over the batch, reading LightGlueForTraining's dense
    `valid0` rather than looping over its ragged `scores` list. Two things that
    used to be control flow are now arithmetic:

      - A sample whose *positive* pair found no match contributes no loss term.
        That `continue` is now a 0/1 weight, so such a sample adds exactly 0 to
        both numerator and denominator. Same value, but it stays in the graph
        instead of dropping out of it — which is what makes the all-empty batch
        safe: `w.sum().clamp(min=1)` turns 0/0 into a plain 0.0 that still has a
        grad_fn. The old code returned a fresh `torch.zeros(requires_grad=True)`
        there, disconnected from every parameter, so backward fired no gradient
        hook at all and DDP's reducer never finished the iteration.
      - A *negative* pair with no match used to take a separate
        `relu(margin - pos_conf)` branch. An empty negative scores exactly 0, so
        that is the same expression as the general one; the branch is gone.

    Equivalence with the previous per-sample implementation (confidences, match
    counts, loss, and every logged stat) is pinned by
    tests/test_dense_matching.py.

    weak_mask: optional (B,) bool tensor from --weak_queries — True marks
    samples whose query is a random frame (not the curated index), paired
    with an uncurated random same-lynx "positive". We still want neg_conf to
    train normally for these (broad, unbiased negative signal), but don't
    trust that random pairing enough to reinforce it as a positive match, so
    pos_conf is masked to a constant for just those samples: same forward
    value (loss magnitude/logging unaffected), zero backward gradient — the
    per-sample torch.where is necessary because a batch mixes weak and
    index-query samples, so the whole pos_conf_all tensor can't just be
    .detach()'d.

    A weak sample is also exempt from the pos_empty drop below. Dropping is
    there to keep a *trusted* positive that found nothing from contributing a
    meaningless term; a weak sample's positive was never trusted (it is
    already detached), so the only thing its term carries is neg_conf — which
    is exactly the signal --weak_queries exists to provide, and which is
    unaffected by whatever the random same-lynx pairing did. Dropping those
    samples would silently discard the broad negative signal on precisely the
    hardest, least index-like queries. Note this raises the reported loss for
    a --weak_queries run: a pos-empty weak sample contributes
    relu(margin + neg_conf) >= margin instead of nothing.
    """
    pos_conf_all  = _lg_scores(pred_pos, data_a, data_p)  # (B,)
    neg_conf_flat = _lg_scores(pred_neg, data_a_neg if data_a_neg is not None else data_a, data_n)  # (B*K,)

    if weak_mask is not None:
        pos_conf_all = torch.where(weak_mask, pos_conf_all.detach(), pos_conf_all)

    if num_negatives > 1:
        neg_conf_all = multi_neg_tau * torch.logsumexp(
            neg_conf_flat.view(-1, num_negatives) / multi_neg_tau, dim=1
        )  # (B,) smooth max over each sample's K negatives
    else:
        neg_conf_all = neg_conf_flat

    pos_matches = pred_pos["valid0"].sum(dim=1)  # (B,)
    neg_matches = pred_neg["valid0"].sum(dim=1)  # (B*K,)
    pos_empty = pos_matches == 0
    neg_empty = neg_matches == 0

    violation = margin - pos_conf_all + neg_conf_all  # (B,)
    if margin_activation == "softplus":
        per_sample = F.softplus(violation * softplus_beta) / softplus_beta
    else:
        per_sample = F.relu(violation)
    keep_mask = ~pos_empty if weak_mask is None else (~pos_empty | weak_mask)
    kept = keep_mask.to(per_sample.dtype)
    loss = (per_sample * kept).sum() / kept.sum().clamp(min=1)

    # Under no_grad so the float() conversions below don't drag detach() calls
    # (or a warning) along; these are diagnostics only. A handful of syncs per
    # step instead of the ~2*batch_size the per-sample .item() calls used to
    # cost. The per-sample pos_conf/neg_conf lists exist so the caller can
    # feed the hard-pair / mining / gap statistics without re-running
    # _lg_scores on its own.
    with torch.no_grad():
        # Deliberately ~pos_empty, not keep_mask: these describe pairs that
        # actually produced matches, so a pos-empty weak sample has no
        # positive match count / confidence to average in even though its
        # term now counts towards the loss.
        keep = ~pos_empty
        keep_neg = keep.repeat_interleave(num_negatives) if num_negatives > 1 else keep
        stats = {
            "pos_skipped":      pos_empty.tolist(),  # list[bool], length B — caller buckets by source
            "neg_skipped":      neg_empty.tolist(),  # list[bool], length B*K, row-major (sample, negative)
            "pos_conf":         pos_conf_all.tolist(),   # list[float], length B
            "neg_conf":         neg_conf_flat.tolist(),  # list[float], length B*K (per negative, pre-smooth-max)
            "mean_pos_matches": _masked_mean(pos_matches.to(per_sample.dtype), keep),
            "mean_neg_matches": _masked_mean(neg_matches.to(per_sample.dtype), keep_neg),
            "mean_pos_conf":    _masked_mean(pos_conf_all, keep),
            "mean_neg_conf":    _masked_mean(neg_conf_flat, keep_neg & ~neg_empty),
            # Fraction of loss-contributing samples still inside the margin —
            # the direct gauge of how saturated the training signal is (relu:
            # exactly the fraction with nonzero gradient).
            "active_frac":      _masked_mean((violation > 0).to(per_sample.dtype), keep_mask),
        }
    return loss, stats


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


def drop_keypoints(feats: list[dict], p: float) -> list[dict]:
    """
    --keypoint_dropout: independently drops each detected keypoint (with its
    descriptor) with probability p. Training-only — eval paths never call this.

    Operates on extract_train's output rather than the image, i.e. in
    LightGlue's actual input space: a frozen RDD largely absorbs photometric
    image-level augmentation before LG ever sees it, but LG cannot be
    invariant to which subset of keypoints exists, so this perturbation always
    lands. Plain tensor indexing, so when RDD is being trained the surviving
    descriptors keep their autograd path. At least one keypoint is always kept
    (batch_features/LG assume non-empty keypoint sets).
    """
    out = []
    for f in feats:
        n = f["keypoints"].shape[0]
        if n <= 1:
            out.append(f)
            continue
        keep = torch.rand(n, device=f["keypoints"].device) >= p
        if not keep.any():
            keep[int(torch.randint(n, (1,)))] = True
        out.append({"keypoints": f["keypoints"][keep], "descriptors": f["descriptors"][keep]})
    return out


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
    establish --moving_negative_prob's pre-training baseline ratio. Also
    temporarily disables weak_queries (if on), since those samples have no
    index counterpart to compare against and would bias the baseline. Restores
    the dataset's original probability/return_meta/weak_queries before
    returning. Uses a throwaway, non-persistent-worker loader so it never
    interferes with the main train_loader's worker pool; that loader is sharded
    across processes with the standalone `prepare_data_loader` rather than
    `accelerator.prepare_data_loader`, because both variants of the latter
    append to `accelerator._dataloaders` and this one-shot loader has no
    business being in every checkpoint's sampler state. Its sums/counts are
    reduced afterwards, so every rank ends up with the *same*
    baseline instead of each computing its own from the same data. The shared
    value matters: run_training_lg turns it into
    `train_ds.random_negative_prob`, which would otherwise diverge per rank and
    give each process a different negative-sampling distribution.

    `n_batches` is per process, so the baseline is measured over
    n_batches * batch_size * num_processes samples — an N-GPU run gets an
    N-times larger sample at the same wall-clock cost, rather than the same
    sample N times over.

    Returns (mean_index_conf, mean_random_conf); either is 0.0 if that source
    didn't come up in the sampled batches.
    """
    prev_prob = dataset.random_negative_prob
    prev_meta = dataset.return_meta
    prev_weak = dataset.weak_queries
    prev_k    = dataset.num_negatives
    prev_hard = dataset.hard_negative_sampling
    prev_jq, prev_jd = dataset.frame_jitter_query, dataset.frame_jitter_db
    dataset.random_negative_prob = force_prob
    dataset.return_meta = True
    dataset.weak_queries = False
    # One negative per sample (this loop unpacks single-image negatives),
    # uniform index-negative choice, and exact index frames — the baseline
    # should describe the index's natural difficulty, not a hard-sampling- or
    # jitter-skewed slice of it (temporal jitter lowers index-negative
    # confidence specifically, which is exactly the ratio being measured).
    dataset.num_negatives = 1
    dataset.hard_negative_sampling = False
    dataset.frame_jitter_query = dataset.frame_jitter_db = 0
    try:
        loader = prepare_data_loader(
            get_loader(
                dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, persistent_workers=False,
            ),
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
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
            neg_conf = _lg_scores(pred_neg, data_a, data_n).tolist()

            for src, is_weak, conf in zip(neg_meta["neg_source"], neg_meta["is_weak_query"], neg_conf):
                if is_weak:
                    continue
                (index_confs if src == "index" else random_confs).append(conf)
    finally:
        dataset.random_negative_prob = prev_prob
        dataset.return_meta = prev_meta
        dataset.weak_queries = prev_weak
        dataset.num_negatives = prev_k
        dataset.hard_negative_sampling = prev_hard
        dataset.frame_jitter_query, dataset.frame_jitter_db = prev_jq, prev_jd

    # Reduce sums and counts (not the two means) so the combined average is
    # weighted by how many samples each rank actually contributed.
    totals = torch.tensor(
        [sum(index_confs), len(index_confs), sum(random_confs), len(random_confs)],
        device=accelerator.device, dtype=torch.float64,
    )
    index_sum, index_n, random_sum, random_n = accelerator.reduce(totals, reduction="sum").tolist()

    mean_index  = index_sum  / index_n  if index_n  else 0.0
    mean_random = random_sum / random_n if random_n else 0.0
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


# ── distillation ──────────────────────────────────────────────────────────────
def compute_distill_ema_decay(total_steps: int, target_weight: float = 0.01) -> float:
    """
    Decay for the --distill_model ema reference, picked relative to how long
    this run actually is rather than a fixed constant like --ema_decay's usual
    0.999: after `total_steps` EMA updates, the pretrained starting point's
    remaining weight in the reference (decay ** total_steps) is
    `target_weight`. A short run (few epochs / small dataset) gets a smaller
    decay so the reference still moves meaningfully before training ends; a
    long run gets a decay close to 1 so it drifts slowly and keeps anchoring
    the student for most of training instead of collapsing onto it early.
    """
    total_steps = max(total_steps, 1)
    return target_weight ** (1.0 / total_steps)


class ActivationCapture:
    """
    Hooks every TransformerLayer in an LG's `transformers` ModuleList and
    records its (desc0, desc1) output, in call order, while active.

    desc0/desc1 are the per-keypoint embeddings LightGlue refines at each
    layer — the residual stream self_attn/cross_attn write into and the value
    log_assignment reads to score matches — so comparing them layer-by-layer
    between student and reference is a direct check on the representation
    that actually drives matching, independent of which internal attention
    mechanics produced it. If the wrapped model's forward is called more than
    once while a single capture is active (e.g. once for the positive pair,
    once for the negative), activations from every call are appended to the
    same list back-to-back — callers that need to tell them apart slice by
    `len(model.transformers)`.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = _unwrap(model)
        self.activations: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._handles: list = []

    def __enter__(self) -> "ActivationCapture":
        self.activations = []
        for layer in self.model.transformers:
            self._handles.append(layer.register_forward_hook(self._hook))
        return self

    def _hook(self, module, inputs, output) -> None:
        desc0, desc1 = output
        self.activations.append((desc0, desc1))

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def accumulate_distill_weights_grad(
    student: nn.Module, reference: nn.Module, loss_type: str, lambda_: float,
) -> torch.Tensor:
    """
    Adds lambda_ * d(consistency_loss)/d(param) straight into every trainable
    parameter's `.grad`, instead of building an autograd edge through
    `student`'s own parameter tensors and routing it through backward().
    Returns the plain (un-lambda'd) consistency loss as a scalar tensor, for
    logging only.

    This has to bypass autograd/DDP, not just avoid it for style: with
    LightGlueForTraining's do_early_stop hardcoded False, every
    token_confidence layer and all but the last log_assignment layer never
    run in a normal forward (see the find_unused_parameters=True docstring on
    run_training_lg) — DDP's reducer, with find_unused_parameters=True,
    pre-marks exactly those parameters "ready" the moment `lg(...)` returns,
    based on a reachability scan from *that* output. A plain
    `student_param - reference_param` edge on those same parameters is
    invisible to that scan (it never goes through `lg`'s forward at all), so
    the moment such an edge produces a real backward contribution for one of
    them, the reducer sees it marked ready a second time and raises "Expected
    to mark a variable ready only once" — reproducibly, only under multi-GPU
    DDP, only for this ('weights') signal type; 'activations' distills
    through LightGlue's actual forward output, which DDP already tracks fine.

    The gradient here is closed-form (2*diff/n for l2, sign(diff)/n for l1),
    and both sides are already bitwise-identical across ranks — the student's
    via DDP's own all-reduce on the margin loss, the reference's via
    update_ema on that same synced student (or an unchanged pretrained
    snapshot) — so there is nothing to collectively communicate here; adding
    it locally on every rank is exact, not an approximation.

    Parameters whose `.grad` is still None after backward are skipped rather
    than having one created for them. That is not an optimization, it is what
    keeps this anchor from causing the very drift it exists to prevent:

      - The always-dead parameters (all 9 token_confidence layers plus
        log_assignment[0..n_layers-2], see the find_unused_parameters=True
        docstring on run_training_lg) never take part in a backward, so under
        both single-GPU and DDP their .grad stays None. Verified: DDP with
        find_unused_parameters=True leaves it None rather than writing zeros.
      - torch.optim.Adam skips `p.grad is None` outright, so such a parameter
        is also exempt from --weight_decay. Materializing a .grad for it
        enrols it: from then on Adam's own L2 term (weight_decay * p) is its
        ONLY gradient, and after Adam's adaptive rescaling that is a step of
        ~lr per iteration straight towards zero, every step, forever. Over a
        300-epoch run at lr=1e-5 that is enough to zero those weights out
        completely — a drift with no data behind it at all, which then shows
        up as a large (and entirely spurious) consistency_loss.
      - Skipping is exact, not an approximation: with nothing else writing to
        these parameters they never leave the reference, so their diff stays 0
        and their true anchor gradient is 0 too. The loss below still sums
        over them for the same reason — they contribute exactly 0.
    """
    student = _unwrap(student)
    pairs = [
        (s_p, r_p)
        for (_, s_p), (_, r_p) in zip(student.named_parameters(), reference.named_parameters())
        if s_p.requires_grad
    ]
    if not pairs:
        return next(student.parameters()).new_zeros(())
    n_elems = sum(s_p.numel() for s_p, _ in pairs)
    total = pairs[0][0].new_zeros(())
    for s_p, r_p in pairs:
        diff = s_p - r_p
        total = total + (diff.abs().sum() if loss_type == "l1" else diff.pow(2).sum())
        if s_p.grad is None:
            continue  # took no part in this backward — see the docstring
        grad = (diff.sign() if loss_type == "l1" else 2 * diff) / n_elems
        s_p.grad.add_(grad, alpha=lambda_)
    return total / n_elems


def distill_activation_loss(
    student_acts: list,
    reference_acts: list,
    mask0: torch.Tensor,
    mask1: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """
    Mean L1/L2 distance between student and reference per-layer descriptor
    embeddings (see ActivationCapture), over valid (non-padding) keypoints
    only — padded slots are masked out of attention and carry whatever the
    FFN does to arbitrary padding content, so matching them would just be
    matching noise. Averaged within each side/layer first (so layers and the
    two sides contribute equally regardless of keypoint count), then over
    layers.
    """
    def _masked_diff(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(a.dtype)  # (B, M, 1), broadcasts over the descriptor dim
        diff = a - b.detach()
        per_elem = diff.abs() if loss_type == "l1" else diff.pow(2)
        denom = (mask.sum() * a.shape[-1]).clamp(min=1)
        return (per_elem * mask).sum() / denom

    layer_losses = [
        _masked_diff(s0, r0, mask0) + _masked_diff(s1, r1, mask1)
        for (s0, s1), (r0, r1) in zip(student_acts, reference_acts)
    ]
    return torch.stack(layer_losses).mean()


def distill_correspondence_loss(
    live_scores: torch.Tensor,
    ref_matches0: torch.Tensor,
    ref_valid0: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """
    NLL of the reference's own matched correspondences under the student's
    raw (pre-filter) assignment scores, restricted to the samples
    `sample_mask` selects. Shared by --distill_signal_type correspondence and
    healing_on_positives, which differ only in that mask.

    live_scores: (B, M+1, N+1) dense log-assignment from the STUDENT —
        LightGlueForTraining's `assignment_scores` output (see
        rdd_patch/lightglue_masked_training.py). live_scores[b, i, j] moves
        together with the three things filter_matches actually checks: the
        row-softmax (is j point i's best candidate), the column-softmax (is
        i point j's best candidate — mutual agreement), and both points'
        matchability certainty (dustbin preference). So pushing it up at a
        specific (i, j) isn't just "differentiable" — it's the right
        direction to eventually clear filter_matches' mutual-nearest-
        neighbor-plus-threshold gate, not merely a proxy for it.
    ref_matches0/ref_valid0: (B, M) each, from a frozen REFERENCE LightGlue
        (--distill_model's distill_lg_ref) run on the exact same (anchor,
        positive) descriptors — pseudo-labels, not ground truth. RDD is
        frozen, so keypoint index i means the same point for student and
        reference; only ref_valid0[b, i]=True rows contribute. Deliberately
        one-sided: we assert "point i matches ref_matches0[i]" but never
        assert "point i matches nothing" for a reference-unmatched point —
        this is a rescue signal, so it should only ever push a pair towards
        more matches, never towards fewer.
    sample_mask: (B,) bool — which samples this loss applies to at all.
        Everything else is untouched, i.e. trains exactly as it would if this
        loss didn't exist. --distill_signal_type correspondence passes
        pos_empty (the student's own pred_pos["valid0"] is all-False this
        step); healing_on_positives passes the wider "the student now scores
        this pair below the pretrained model did" mask, of which pos_empty is
        the extreme case. Either way the mask is computed by the caller, so
        this function stays agnostic to which gate produced it.

    Reading live_scores at a specific (i, j) is a plain index into an
    already-differentiable tensor — unlike matching_scores0 * valid0 (used
    everywhere else, e.g. _lg_scores), it never passes through
    filter_matches' torch.where(mutual0, ..., zero), so it carries nonzero
    gradient even for a sample whose valid0 is currently all-False. That's
    the one thing this loss exists to provide.

    Returns a plain 0.0 (no grad_fn — safe under find_unused_parameters=True,
    see run_training_lg) when no
    sample is both selected by sample_mask and has at least one
    reference-matched point to learn from.
    """
    weight = ref_valid0.to(live_scores.dtype) * sample_mask[:, None].to(live_scores.dtype)
    if weight.sum() == 0:
        return live_scores.new_zeros(())

    M = ref_matches0.shape[1]
    j_idx = ref_matches0.clamp(min=0)  # dummy index for unmatched rows; zeroed out by `weight`
    nll = -live_scores[:, :M, :].gather(2, j_idx.unsqueeze(-1)).squeeze(-1)  # (B, M)

    per_sample = (nll * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)
    sample_weight = (weight.sum(dim=1) > 0).to(per_sample.dtype)
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1)


@torch.no_grad()
def measure_pretrained_positive_scores(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg_ref: torch.nn.Module,
    loader,
    args: argparse.Namespace,
) -> dict[tuple[str, str], float]:
    """
    Scores every (query_frame, positive) pair in the training index with the
    pretrained LightGlue, once, before training — the fixed reference
    --distill_signal_type healing_on_positives gates on. Returns
    {(query_rel, positive_rel): _lg_scores confidence}.

    Precomputed rather than recomputed per step because the same pair comes
    back on the order of epochs/n_positives times over a run (~60 for a
    300-epoch run on a top_k=5 index), and because a fixed target is the point:
    the gate asks whether the student has fallen below where training *started*
    on this specific pair.

    `loader` is the training split's eval_pseudo_accuracy loader (see
    build_pseudo_accuracy_loader): already constructed, already sharded across
    processes, and already carrying each query's whole candidate pool in one
    batch. Reusing it means not standing up a second dataset or duplicating the
    chunked-extraction logic; the cost is scoring that pool's negatives too and
    discarding them, a constant factor on a one-time pass.

    Uses the same _lg_scores the margin loss and eval use, on the same clean
    (never augmented) eval transform, so the gate compares like with like — see
    the --augment caveat in the module docstring.

    The observations are all-gathered, so every rank ends up with the same
    complete table. That matters: the gate has to agree across ranks, or the
    same pair would heal on one GPU and not on another and the two would
    disagree about what the loss even is. Accelerate pads the last shard by
    repeating samples; the duplicates re-score identically and collapse onto
    the same dict key.
    """
    device = accelerator.device
    _unwrap(rdd).eval()
    ds = loader.dataset

    observations: list[tuple[str, str, float]] = []
    for query_batch, cand_batch, idx_batch in tqdm(
        loader, desc="healing-ref", leave=False, disable=not accelerator.is_main_process
    ):
        query_batch = query_batch.to(device)
        cand_batch  = cand_batch.to(device)
        B, n_cand, C, H, W = cand_batch.shape

        query_r = resize_long_side(query_batch, args.resize)
        H_q, W_q = query_r.shape[-2:]
        feats_q = _extract_chunked(_unwrap(rdd), query_r, args.batch_size)

        cand_r = resize_long_side(cand_batch.view(B * n_cand, C, H, W), args.resize)
        H_c, W_c = cand_r.shape[-2:]
        feats_c = _extract_chunked(_unwrap(rdd), cand_r, args.batch_size)

        feats_q_rep = [f for f in feats_q for _ in range(n_cand)]
        data_q = batch_features(feats_q_rep, H_q, W_q)
        data_c = batch_features(feats_c,     H_c, W_c)
        pred = lg_ref({"image0": data_q, "image1": data_c})
        # Candidates are stacked positives-then-negatives (PseudoAccuracyDataset),
        # so the leading n_pos columns are what this table is about.
        pos_scores = _lg_scores(pred, data_q, data_c).view(B, n_cand)[:, :ds.n_pos]

        for row, idx in zip(pos_scores.tolist(), idx_batch.tolist()):
            entry = ds.entries[idx]
            for pos_rel, score in zip(entry["positives"], row):
                observations.append((entry["query_frame"], pos_rel, score))

    return {(q, p): s for q, p, s in gather_object(observations)}


# ── LR warmup ─────────────────────────────────────────────────────────────────
def apply_warmup_lr(
    optimizer: torch.optim.Optimizer, global_step: int, args: argparse.Namespace,
) -> float | None:
    """
    Linearly ramps every param group's LR from ~0 to args.lr over
    args.warmup_steps optimizer steps (global_step is the count of steps
    already completed *before* this one, so the first call uses global_step=0
    and the last uses global_step=warmup_steps-1, landing exactly on args.lr).

    Called once per step, right before optimizer.step() — see train_epoch_lg.
    Needed because CosineAnnealingLR (run_training_lg) is only stepped once
    per *epoch*, so without this, epoch 0 would run at the full --lr for its
    entire duration; this override simply wins for any step where it's
    active, regardless of what the epoch-level scheduler set beforehand.
    Returns None (no-op, param groups untouched) once warmup_steps have
    elapsed or when --warmup_steps is 0 (the default).
    """
    if args.warmup_steps <= 0 or global_step >= args.warmup_steps:
        return None
    warmup_lr = args.lr * (global_step + 1) / args.warmup_steps
    for group in optimizer.param_groups:
        group["lr"] = warmup_lr
    return warmup_lr


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
    distill_lg_ref: torch.nn.Module | None = None,
    distill_rdd_ref: torch.nn.Module | None = None,
    prev_dead_pos_index: set[str] | None = None,
    pretrained_pos_scores: dict[tuple[str, str], float] | None = None,
    margin: float | None = None,
) -> tuple[float, int, dict | None, set[str], dict]:
    """
    `eval_lg` is what mini-evals run against (the EMA shadow when --ema_decay >
    0, else `lg` itself); `lg` is always what receives gradient. `ema_lg` (same
    object as `eval_lg` when EMA is on, else None) is updated after every step.

    `distill_lg_ref`/`distill_rdd_ref` are the --distill_model reference
    models (None unless that model is needed — see run_training_lg), fixed
    for --distill_model pretrained or updated after every step for ema. Used
    to add a consistency loss to the margin loss; see
    accumulate_distill_weights_grad / distill_activation_loss /
    distill_correspondence_loss and the module docstring. Note
    accumulate_distill_weights_grad doesn't go through backward_loss like the
    other two — it injects gradient directly after accelerator.backward(),
    see its own docstring and the comment at that call site below.

    `loader`'s dataset is always built with return_meta=True, so every batch
    carries a 4th `neg_meta` element (neg_source, query_lynx, neg_lynx,
    is_weak_query). This is used for two independent things:
      - Always: train/skip_rate_{pos,neg}_{index,random} — the fraction of
        pairs with zero matches, split by pair type (pos/neg) and by whether
        that pair's query came from the curated index or not (see skip_counts
        below; "random" here means an --weak_queries query, since only the
        query side of a *positive* pair has an index/random distinction).
      - Only when --moving_negative_prob or --negative_mining is active
        (gap_tracking_active): opportunistically bucket that epoch's
        *index-query* negatives' LG confidence by `neg_source` ("index" vs
        "random") into the returned dict, for run_training_lg to act on after
        the epoch — see NEG_GAP_EPS / compute_moving_prob /
        IndexAssignedTripletDataset.update_mining_stats. --weak_queries
        samples are excluded here (see docstring on update_mining_stats) —
        they don't share the index-query random-negative distribution these
        two mechanisms compare against. The returned dict's gap-tracking
        fields are all-empty/zero when gap_tracking_active is False.

    Always (independent of gap_tracking_active): index-query positive skips
    are also tracked by identity via `neg_meta["query_frame"]`, giving
    `dead_this_epoch` — the set of query_frames whose pos_index pair was
    empty at least once this epoch, returned as the 4th tuple element (fed
    back in next epoch as `prev_dead_pos_index` — see run_training_lg). When
    a previous epoch's set is available, two extra metrics are logged
    alongside the skip rates below:
      train/skip_rate_pos_index_dead_count   len(dead_this_epoch) — how many
                                              distinct index queries had a
                                              zero-match positive this epoch.
      train/skip_rate_pos_index_recurrence   what fraction of dead_this_epoch
                                              was *also* in prev_dead_pos_index
                                              — i.e. of today's dead
                                              candidates, how many were
                                              already dead last epoch, as
                                              opposed to newly dead this
                                              epoch. Absent on the first
                                              epoch (no previous set yet).

    `pretrained_pos_scores` is --distill_signal_type healing_on_positives'
    lookup table, {(query_frame, pos_frame): pretrained confidence}, built once
    before training by measure_pretrained_positive_scores; None for every other
    signal type. When it is in use one more metric is logged:
      train/heal_rate    of the index-query samples this epoch (weak queries
                          have no reference score and are excluded), the
                          fraction whose positive pair the student now scores
                          below the pretrained model — i.e. how much of the
                          traffic the healing loss actually fired on. 0 means
                          the signal is inert; near 1 means the whole positive
                          side has regressed.

    `margin` overrides args.lg_margin for this epoch (--adaptive_margin's
    live value; None keeps the configured constant). The 5th return element
    is a dict with `pair_observations` (all-gathered (query, candidate, conf)
    triples for train_ds.update_pair_stats — empty unless
    --hard_positive_sampling/--hard_negative_sampling) and `sep_gap` (the
    epoch's mean pos/neg confidence separation, identical on every rank —
    --adaptive_margin's input).
    """
    train_rdd, train_lg = resolve_trained_models(args.trained_model)
    _unwrap(rdd).train(train_rdd)
    lg.train(train_lg)

    # --adaptive_margin passes the live (per-epoch) margin; everything else
    # runs at the configured constant.
    margin = args.lg_margin if margin is None else margin
    K = args.num_negatives

    distill_active         = args.distill_model != "none"
    distill_weights_active = distill_active and args.distill_signal_type == "weights"
    distill_acts_active    = distill_active and args.distill_signal_type == "activations"
    distill_corr_active    = distill_active and args.distill_signal_type == "correspondence"
    distill_heal_active    = distill_active and args.distill_signal_type == "healing_on_positives"

    gap_tracking_active = args.moving_negative_prob is not None or args.negative_mining
    mining_active        = args.negative_mining
    weak_active           = args.weak_queries
    hard_pair_active      = args.hard_positive_sampling or args.hard_negative_sampling
    epoch_index_confs:  list[float] = []
    epoch_random_confs: list[float] = []
    epoch_mining_obs:   list[tuple[str, str, str, float]] = []
    epoch_dead_pos_index_frames: list[str] = []
    # (query_frame, candidate_frame, conf) for --hard_positive_sampling /
    # --hard_negative_sampling — fed to train_ds.update_pair_stats once per
    # epoch by run_training_lg. Positives and index negatives share one list
    # (their keys can't collide — a candidate is never both for one query).
    epoch_pair_obs: list[tuple[str, str, float]] = []
    # Mean (pos_conf - neg_conf) separation gap over the epoch, from the
    # already-all-reduced per-step means — identical on every rank, which
    # --adaptive_margin relies on to keep the margin in sync across processes.
    epoch_gap_sum = 0.0

    # [n_skipped, n_total] per (pair, query-source) bucket, for
    # train/skip_rate_{pos,neg}_{index,random}.
    skip_counts = {
        "pos_index": [0, 0], "pos_random": [0, 0],
        "neg_index": [0, 0], "neg_random": [0, 0],
    }
    # [n_healed, n_eligible] for train/heal_rate — how much of the index
    # positive traffic --distill_signal_type healing_on_positives actually
    # fires on. Eligible means "has a pretrained reference score", i.e. every
    # index-query sample and no --weak_queries one. Accumulated on-device so
    # counting costs no per-step GPU sync; read once at the end of the epoch.
    heal_counts = torch.zeros(2, device=accelerator.device, dtype=torch.float64)

    def _bump_skip(key: str, skipped: bool) -> None:
        counts = skip_counts[key]
        counts[0] += int(skipped)
        counts[1] += 1

    epoch_loss     = 0.0
    mini_eval_time = 0.0
    steps_per_epoch = len(loader)
    t_epoch_start = time.perf_counter()

    pbar = tqdm(
        enumerate(loader),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:02d}",
        disable=not accelerator.is_main_process,
    )
    for step, (anchors, positives, negatives, neg_meta) in pbar:
        device = accelerator.device
        # --multi_scale_min/max: per-step random resize target (train only —
        # every eval path keeps the fixed args.resize).
        step_resize = args.resize
        if args.multi_scale_max > 0:
            step_resize = random.choice(range(args.multi_scale_min, args.multi_scale_max + 1, 32))
        if K > 1:
            # (B, K, C, H, W) from the dataset -> one flat (B*K) negative
            # batch, row-major (sample, negative) — the layout every flat
            # per-negative structure below (stats/meta/confs) shares.
            negatives = negatives.reshape(-1, *negatives.shape[2:])
        anchors_r   = resize_long_side(anchors,   step_resize).to(device)
        positives_r = resize_long_side(positives, step_resize).to(device)
        negatives_r = resize_long_side(negatives, step_resize).to(device)
        H_r, W_r = anchors_r.shape[-2:]

        # When RDD isn't being trained, no_grad purely skips building an unused graph.
        with contextlib.nullcontext() if train_rdd else torch.no_grad():
            feats_a = extract_train(rdd, anchors_r)
            feats_p = extract_train(rdd, positives_r)
            feats_n = extract_train(rdd, negatives_r)

        if args.keypoint_dropout > 0:
            feats_a = drop_keypoints(feats_a, args.keypoint_dropout)
            feats_p = drop_keypoints(feats_p, args.keypoint_dropout)
            feats_n = drop_keypoints(feats_n, args.keypoint_dropout)

        data_a = batch_features(feats_a, H_r, W_r)
        data_p = batch_features(feats_p, H_r, W_r)
        data_n = batch_features(feats_n, H_r, W_r)
        # Anchor features repeated per negative, so the negative pass stays a
        # single flat (B*K)-pair LG call — the same repeat-the-query pattern
        # eval_pseudo_accuracy uses. Same object as data_a when K == 1.
        data_a_neg = (
            data_a if K == 1
            else batch_features([f for f in feats_a for _ in range(K)], H_r, W_r)
        )

        # ActivationCapture accumulates hooks fired during *every* forward
        # call made while it's active — both lg() calls below run in one
        # context, so the flat list is sliced by n_layers below instead of
        # using two separate captures.
        cap_ctx = ActivationCapture(lg) if distill_acts_active else contextlib.nullcontext()
        with cap_ctx as stu_cap:
            pred_pos = lg({"image0": data_a,     "image1": data_p})
            pred_neg = lg({"image0": data_a_neg, "image1": data_n})
        if distill_acts_active:
            n_layers = len(_unwrap(lg).transformers)
            stu_acts_pos, stu_acts_neg = stu_cap.activations[:n_layers], stu_cap.activations[n_layers:]

        weak_mask = None
        if weak_active:
            weak_mask = torch.as_tensor(neg_meta["is_weak_query"], dtype=torch.bool, device=device)

        loss, stats = lg_confidence_loss(
            pred_pos, pred_neg, margin,
            data_a=data_a, data_p=data_p, data_n=data_n, weak_mask=weak_mask,
            data_a_neg=data_a_neg, num_negatives=K, multi_neg_tau=args.multi_neg_tau,
            margin_activation=args.margin_activation, softplus_beta=args.softplus_beta,
        )

        consistency_loss = loss.new_zeros(())
        backward_loss = loss
        if distill_acts_active:
            # Reference LG runs on the exact same keypoints/descriptors the
            # student just used (data_a/data_p/data_n), so the two are always
            # shape-compatible even when RDD is also being trained and its
            # descriptors/keypoint set have drifted from the reference RDD's —
            # this isolates how much LightGlue's own transform of a given
            # input has moved, which is what --trained_model requiring 'lg'
            # for this mode (see parse_args) is there to make meaningful.
            with torch.no_grad():
                with ActivationCapture(distill_lg_ref) as ref_cap:
                    distill_lg_ref({"image0": data_a, "image1": data_p})
                    if not args.activations_on_positives:
                        distill_lg_ref({"image0": data_a_neg, "image1": data_n})
            # ref_acts_neg is an empty list under --activations_on_positives —
            # the forward that would have filled it never ran, and nothing
            # below reads it. (The student's negative activations are captured
            # either way: the margin loss needs that forward regardless, so
            # there is nothing to save there.)
            ref_acts_pos, ref_acts_neg = ref_cap.activations[:n_layers], ref_cap.activations[n_layers:]
            mask_a = data_a["masks"].squeeze(1)
            mask_p = data_p["masks"].squeeze(1)
            consistency_loss = distill_activation_loss(
                stu_acts_pos, ref_acts_pos, mask_a, mask_p, args.distill_loss
            )
            if not args.activations_on_positives:
                mask_an = data_a_neg["masks"].squeeze(1)
                mask_n  = data_n["masks"].squeeze(1)
                consistency_loss = 0.5 * (
                    consistency_loss
                    + distill_activation_loss(stu_acts_neg, ref_acts_neg, mask_an, mask_n, args.distill_loss)
                )
        elif distill_corr_active:
            # Only the positive pair — there's nothing to "rescue" on the
            # negative side (an empty negative is the desired outcome, not a
            # problem; see lg_confidence_loss's neg_empty handling).
            with torch.no_grad():
                ref_pred_pos = distill_lg_ref({"image0": data_a, "image1": data_p})
            pos_empty = pred_pos["valid0"].sum(dim=1) == 0
            if weak_mask is not None:
                # Same distrust as lg_confidence_loss's weak_mask: a
                # --weak_queries "positive" is an uncurated random same-lynx
                # frame pair, not a curated index entry — the "pretrained
                # matched every index positive" premise this loss rests on
                # doesn't extend to it, so a weak query never counts as
                # rescuable here, however empty its match set is.
                pos_empty = pos_empty & ~weak_mask
            consistency_loss = distill_correspondence_loss(
                pred_pos["assignment_scores"], ref_pred_pos["matches0"], ref_pred_pos["valid0"], pos_empty,
            )
        elif distill_heal_active:
            # Same rescue as 'correspondence' above (positive pair only, same
            # pseudo-labels), on a wider gate: fire wherever the student now
            # scores this exact (query, positive) pair below what the
            # pretrained model scored for it before training started.
            with torch.no_grad():
                ref_pred_pos = distill_lg_ref({"image0": data_a, "image1": data_p})
                live_pos_conf = _lg_scores(pred_pos, data_a, data_p)
            # -inf for any pair the pre-training pass didn't score, so `live <
            # ref` is False for it and it simply never heals.
            ref_pos_conf = torch.tensor(
                [
                    pretrained_pos_scores.get((q_frame, p_frame), -math.inf)
                    for q_frame, p_frame in zip(neg_meta["query_frame"], neg_meta["pos_frame"])
                ],
                device=device, dtype=live_pos_conf.dtype,
            )
            eligible = torch.isfinite(ref_pos_conf)
            if weak_mask is not None:
                # Same distrust as lg_confidence_loss's weak_mask and as the
                # 'correspondence' branch above. Masked explicitly rather than
                # left to the -inf default: a weak triplet draws its query and
                # positive from the same pool the index was built over, so it
                # can land on a (query, positive) pair that *does* have a
                # reference score by coincidence, and that pair is still an
                # uncurated random pairing this loss has no business enforcing.
                eligible = eligible & ~weak_mask
            heal_mask = (live_pos_conf < ref_pos_conf) & eligible
            heal_counts[0] += heal_mask.sum()
            heal_counts[1] += eligible.sum()
            consistency_loss = distill_correspondence_loss(
                pred_pos["assignment_scores"], ref_pred_pos["matches0"], ref_pred_pos["valid0"], heal_mask,
            )
        # distill_weights_active is deliberately *not* handled here — its
        # consistency_loss is computed after accelerator.backward() below by
        # accumulate_distill_weights_grad, which injects gradient directly
        # into .grad instead of joining this autograd graph. See that
        # function's docstring: an autograd edge straight through the
        # trained model's own parameter tensors (bypassing its forward())
        # crashes multi-GPU DDP, because DDP's find_unused_parameters=True
        # already pre-marks LightGlue's always-dead token_confidence /
        # non-final log_assignment parameters "ready" from lg()'s own
        # forward output, and a second, invisible-to-DDP path to those same
        # parameters trips "Expected to mark a variable ready only once".
        if distill_acts_active or distill_corr_active or distill_heal_active:
            backward_loss = loss + args.distill_model_lambda * consistency_loss

        def _per_negative(key: str) -> list[list]:
            # Meta fields that are per-negative: scalars for K == 1 (collated
            # to a flat length-B sequence), K-length lists for K > 1 (collated
            # to K sequences of B — transposed back here). Either way the
            # result is indexed [sample][negative].
            v = neg_meta[key]
            if K == 1:
                return [[x] for x in v]
            return [list(col) for col in zip(*v)]

        neg_sources = _per_negative("neg_source")
        neg_lynxes  = _per_negative("neg_lynx")
        neg_frames  = _per_negative("neg_frame")

        # One pass over the batch for every per-sample/per-negative statistic:
        # skip buckets, dead-positive tracking, hard-pair observations
        # (--hard_positive_sampling / --hard_negative_sampling), and the
        # index/random confidence buckets + mining observations
        # (gap_tracking_active). Confidences come from the loss's own stats,
        # so nothing is recomputed.
        for b, (is_weak, pos_skip, q_frame, q_lynx, pos_frame) in enumerate(zip(
            neg_meta["is_weak_query"], stats["pos_skipped"],
            neg_meta["query_frame"], neg_meta["query_lynx"], neg_meta["pos_frame"],
        )):
            is_weak = bool(is_weak)
            _bump_skip("pos_random" if is_weak else "pos_index", pos_skip)
            if not is_weak and pos_skip:
                epoch_dead_pos_index_frames.append(q_frame)
            if args.hard_positive_sampling and not is_weak:
                epoch_pair_obs.append((q_frame, pos_frame, stats["pos_conf"][b]))
            for k in range(K):
                src  = neg_sources[b][k]
                conf = stats["neg_conf"][b * K + k]
                _bump_skip("neg_random" if src == "random" else "neg_index", stats["neg_skipped"][b * K + k])
                if is_weak:
                    continue  # no index counterpart to compare against — see docstring above
                if src == "index":
                    if args.hard_negative_sampling:
                        epoch_pair_obs.append((q_frame, neg_frames[b][k], conf))
                    if gap_tracking_active:
                        epoch_index_confs.append(conf)
                elif gap_tracking_active:
                    epoch_random_confs.append(conf)
                    if mining_active:
                        epoch_mining_obs.append((q_lynx, neg_lynxes[b][k], neg_frames[b][k], conf))

        trainable_params = [p for p in _unwrap(lg).parameters() if p.requires_grad]
        if train_rdd:
            trainable_params += [p for p in _unwrap(rdd).parameters() if p.requires_grad]

        optimizer.zero_grad()
        accelerator.backward(backward_loss)

        # --distill_signal_type weights: inject its gradient now, straight
        # into .grad, after the margin/activations/correspondence backward()
        # (and DDP's all-reduce on it) has already completed — see the
        # comment above and accumulate_distill_weights_grad's docstring.
        if distill_weights_active:
            if train_lg and distill_lg_ref is not None:
                consistency_loss = consistency_loss + accumulate_distill_weights_grad(
                    lg, distill_lg_ref, args.distill_loss, args.distill_model_lambda
                )
            if train_rdd and distill_rdd_ref is not None:
                consistency_loss = consistency_loss + accumulate_distill_weights_grad(
                    rdd, distill_rdd_ref, args.distill_loss, args.distill_model_lambda
                )

        accelerator.clip_grad_norm_(trainable_params, args.grad_clip)
        apply_warmup_lr(optimizer, global_step, args)
        optimizer.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if ema_lg is not None:
            update_ema(ema_lg, lg, args.ema_decay)
        if distill_active and args.distill_model == "ema":
            if train_lg and distill_lg_ref is not None:
                update_ema(distill_lg_ref, lg, args.distill_ema_decay)
            if train_rdd and distill_rdd_ref is not None:
                update_ema(distill_rdd_ref, rdd, args.distill_ema_decay)

        # Every rank only sees its own shard of the batch, so the logged
        # scalars would otherwise describe 1/num_processes of the data. All
        # seven go into a single tensor to keep this to one collective per step
        # (the loss already forced a sync via .item(), so the added cost is
        # just the all-reduce itself). The four `stats` entries are per-rank
        # means over differing sample counts, so their average is approximate —
        # fine for diagnostics, unlike the loss, which is an exact mean because
        # every rank contributes one loss value. total_loss is computed here
        # as plain Python arithmetic (not an autograd tensor) — for
        # 'weights', consistency_loss never had a graph in the first place
        # (see accumulate_distill_weights_grad); for the others, loss/
        # consistency_loss's graphs were already consumed by backward() above,
        # so reusing their values for a fresh sum is safe but adds nothing.
        loss_item = loss.item()
        consistency_loss_item = consistency_loss.item()
        total_loss_item = (
            loss_item + args.distill_model_lambda * consistency_loss_item if distill_active else loss_item
        )
        step_metrics = torch.tensor(
            [
                loss_item, consistency_loss_item, total_loss_item,
                stats["mean_pos_matches"], stats["mean_neg_matches"],
                stats["mean_pos_conf"],    stats["mean_neg_conf"],
                stats["active_frac"],
            ],
            device=device, dtype=torch.float32,
        )
        (
            loss_val, consistency_loss_val, total_loss_val,
            mean_pos_matches, mean_neg_matches, mean_pos_conf, mean_neg_conf,
            active_frac_val,
        ) = accelerator.reduce(step_metrics, reduction="mean").tolist()
        epoch_loss += loss_val
        epoch_gap_sum += mean_pos_conf - mean_neg_conf
        global_step += 1

        progress = (epoch * steps_per_epoch + step + 1) / (total_epochs * steps_per_epoch)
        pbar.set_postfix(
            loss=f"{loss_val:.4f}",
            pos_skip=sum(stats["pos_skipped"]),
            neg_skip=sum(stats["neg_skipped"]),
        )

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "train/loss":                 loss_val,
                    "train/consistency_loss":     consistency_loss_val,
                    "train/total_loss":           total_loss_val,
                    "train/lr_step":               current_lr,
                    # Fraction of loss-contributing triplets still violating
                    # the margin — the saturation gauge: near 0 means almost
                    # no sample produces gradient any more.
                    "train/active_frac":           active_frac_val,
                    "matches/mean_pos":           mean_pos_matches,
                    "matches/mean_neg":           mean_neg_matches,
                    "lg_confidence/mean_pos_conf": mean_pos_conf,
                    "lg_confidence/mean_neg_conf": mean_neg_conf,
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

    # Sum the raw [n_skipped, n_total] pairs across ranks before turning them
    # into rates — each rank only counted its own shard, and a ratio of sums is
    # not the mean of the per-rank ratios when the buckets are unevenly filled
    # (which they are: whether a pair is index- or random-sourced is sampled
    # per item, so bucket sizes differ from rank to rank).
    skip_keys = ("pos_index", "pos_random", "neg_index", "neg_random")
    skip_totals = accelerator.reduce(
        torch.tensor(
            [c for key in skip_keys for c in skip_counts[key]],
            device=accelerator.device, dtype=torch.float64,
        ),
        reduction="sum",
    ).tolist()
    skip_counts = {key: skip_totals[2 * i:2 * i + 2] for i, key in enumerate(skip_keys)}

    def _skip_rate(key: str) -> float:
        n_skip, n_total = skip_counts[key]
        return n_skip / n_total if n_total else 0.0

    # Collective (gather_object flattens a list input across ranks — same
    # pattern as epoch_mining_obs below), so this must run on every rank
    # unconditionally, same as the skip_totals reduce above.
    dead_this_epoch = set(gather_object(epoch_dead_pos_index_frames))
    recurrence = None
    if prev_dead_pos_index is not None and dead_this_epoch:
        recurrence = len(dead_this_epoch & prev_dead_pos_index) / len(dead_this_epoch)

    # Same reduce-then-divide as the skip rates above, and same reason it can be
    # guarded: distill_heal_active comes from args alone, so every rank agrees.
    heal_rate = None
    if distill_heal_active:
        n_healed, n_eligible = accelerator.reduce(heal_counts, reduction="sum").tolist()
        heal_rate = n_healed / n_eligible if n_eligible else 0.0

    if accelerator.is_main_process:
        skip_log = {
            "train/skip_rate_pos_index":  _skip_rate("pos_index"),
            "train/skip_rate_pos_random": _skip_rate("pos_random"),
            "train/skip_rate_neg_index":  _skip_rate("neg_index"),
            "train/skip_rate_neg_random": _skip_rate("neg_random"),
            "train/skip_rate_pos_index_dead_count": len(dead_this_epoch),
            "time/train_s":     epoch_train_time,
            "time/mini_eval_s": mini_eval_time,
        }
        if recurrence is not None:
            skip_log["train/skip_rate_pos_index_recurrence"] = recurrence
        if heal_rate is not None:
            skip_log["train/heal_rate"] = heal_rate
        accelerator.log(skip_log, step=global_step)

    neg_gap_stats = None
    if gap_tracking_active:
        # Both consumers of this dict mutate train_ds, which every rank holds
        # its own copy of — so the inputs have to be made global here or the
        # ranks drift into different sampling distributions and different
        # mining matrices. Sums/counts are reduced (means computed after, so
        # they're sample-weighted); the mining observations are all-gathered as
        # objects, since they're (str, str, str, float) tuples rather than tensors.
        # Both are collectives, so they must run on every rank —
        # gap_tracking_active is derived from args alone, so it always does.
        totals = accelerator.reduce(
            torch.tensor(
                [sum(epoch_index_confs), len(epoch_index_confs),
                 sum(epoch_random_confs), len(epoch_random_confs)],
                device=accelerator.device, dtype=torch.float64,
            ),
            reduction="sum",
        ).tolist()
        index_sum, index_n, random_sum, random_n = totals

        neg_gap_stats = {
            "mean_index_conf":  index_sum  / index_n  if index_n  else 0.0,
            "mean_random_conf": random_sum / random_n if random_n else 0.0,
            "n_index":  int(index_n),
            "n_random": int(random_n),
            "mining_observations": gather_object(epoch_mining_obs) if mining_active else [],
        }

    epoch_extras = {
        # All-gathered for the same reason as mining_observations above: every
        # rank's train_ds copy must apply the identical update_pair_stats or
        # the ranks drift into different sampling distributions. A collective,
        # so guarded by hard_pair_active, which is derived from args alone and
        # agrees across ranks.
        "pair_observations": gather_object(epoch_pair_obs) if hard_pair_active else [],
        # Built from per-step all-reduced means, so already identical on every
        # rank — --adaptive_margin turns this into next epoch's margin.
        "sep_gap": epoch_gap_sum / max(steps_per_epoch, 1),
    }
    return epoch_loss / max(steps_per_epoch, 1), global_step, neg_gap_stats, dead_this_epoch, epoch_extras


# ── full training run ─────────────────────────────────────────────────────────
def run_training_lg(args: argparse.Namespace) -> None:
    seed_all(args.seed)

    # find_unused_parameters=True is mandatory here, not a precaution: LightGlue
    # is instantiated with depth_confidence/width_confidence = -1 (see
    # build_masked_lg), and LightGlueForTraining._forward hardcodes the matching
    # do_early_stop/do_point_pruning to False. That leaves whole trainable
    # submodules off the forward graph every step — all 8 `token_confidence`
    # layers, plus `log_assignment[0..n_layers-2]`, of which only the last is
    # ever applied. DDP's default (False) makes the reducer wait for gradients
    # on those parameters that never arrive, and the step after aborts with
    # "Expected to have finished reduction in the prior iteration before
    # starting a new one".
    accelerator = Accelerator(
        log_with="wandb" if args.project else None,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    moving_active  = args.moving_negative_prob is not None
    mining_active  = args.negative_mining
    weak_active    = args.weak_queries
    hard_pair_active = args.hard_positive_sampling or args.hard_negative_sampling
    # see persistent_workers note below
    dataset_mutates = moving_active or mining_active or hard_pair_active

    # ── data ──
    train_transform, eval_transform = build_transforms(args.augment)
    train_ds = IndexAssignedTripletDataset(
        args.train_index, root=args.data_root, transform=train_transform,
        random_negative_prob=args.random_negative_prob,
        negative_mining=mining_active,
        negative_mining_temperature=args.negative_mining_temperature,
        negative_mining_decay=args.negative_mining_decay,
        negative_mining_frame_prob=args.negative_mining_frame_prob,
        weak_queries=weak_active,
        weak_queries_prob=args.weak_queries_prob,
        num_negatives=args.num_negatives,
        hard_positive_sampling=args.hard_positive_sampling,
        hard_negative_sampling=args.hard_negative_sampling,
        hard_pair_temperature=args.hard_pair_temperature,
        hard_pair_decay=args.hard_pair_decay,
        frame_jitter_query=args.frame_jitter_query,
        frame_jitter_db=args.frame_jitter_db,
        frame_jitter_prob=args.frame_jitter_prob,
        # Always on: train_epoch_lg uses neg_source/is_weak_query to split
        # train/skip_rate_* by pair type regardless of which (if any) of the
        # adaptive-sampling flags below are active.
        return_meta=True,
    )
    # Diagnostics/eval on the training split must stay on clean, index-only,
    # uniformly-sampled single-negative queries, even when the actual training
    # loader is augmented and/or mixes in random negatives, weak queries, hard
    # sampling, or K negatives — otherwise train_eval/mini_train metrics would
    # be noisier than val's and not comparable across epochs. Always a
    # separate dataset: train_ds carries return_meta=True (4-tuples), which
    # eval_epoch's 3-tuple unpack can't consume.
    train_ds_eval = IndexAssignedTripletDataset(args.train_index, root=args.data_root, transform=eval_transform)
    val_ds = IndexAssignedTripletDataset(args.val_index, root=args.data_root, transform=eval_transform)

    # persistent_workers=True (get_loader's default) would pickle train_ds into
    # long-lived worker processes once and never see it again — fatal for
    # --moving_negative_prob/--negative_mining/--hard_*_sampling, which mutate
    # train_ds (random_negative_prob, the mining EMA matrix, the pair-stats
    # EMA) from the main process between epochs. Disabling it means each
    # epoch's fresh worker spawn re-pickles the current state instead.
    # (--weak_queries and return_meta don't mutate anything post-construction,
    # so they don't need this.)
    train_loader = get_loader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, seed=args.seed,
        persistent_workers=(not dataset_mutates) and args.num_workers > 0,
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
    # These two are prepared (and their worker pools spun up) once here rather
    # than per evaluation — see build_pseudo_accuracy_loader.
    eval_train_loader = build_pseudo_accuracy_loader(accelerator, train_ds_eval, args)
    eval_val_loader   = build_pseudo_accuracy_loader(accelerator, val_ds, args)

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

    # ── --distill_model reference(s) ──
    # Snapshotting here (after LoRA/freeze_confidence_head, before
    # accelerator.prepare) means the reference always has the same structure
    # as the student it's compared against — the 'pretrained' case just never
    # updates it afterwards, while 'ema' updates it every step in
    # train_epoch_lg. Only built when actually needed: for 'weights', student
    # and a *frozen* reference are trivially identical, so lg_ref/rdd_ref are
    # only built for the model(s) --trained_model actually unfreezes. For
    # 'activations'/'correspondence'/'healing_on_positives', train_epoch_lg
    # always feeds the reference LG the student's own already-extracted
    # descriptors (not a separately-extracted reference RDD pass), so the
    # comparison is only ever non-trivial when LG itself is being trained —
    # parse_args enforces 'lg' in --trained_model for all three, which is
    # exactly why `train_lg` is guaranteed True here.
    distill_active   = args.distill_model != "none"
    distill_lg_ref  = None
    distill_rdd_ref = None
    if distill_active:
        if args.distill_signal_type in ("activations", "correspondence", "healing_on_positives") or train_lg:
            distill_lg_ref = build_ema(lg, device)
        if args.distill_signal_type == "weights" and train_rdd:
            distill_rdd_ref = build_ema(rdd, device)

    trainable_params = [p for p in lg.parameters() if p.requires_grad]
    if train_rdd:
        trainable_params += [p for p in rdd.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # DDP-wrapping a module with no trainable parameters is not just
    # unnecessary, it's a hard error ("DistributedDataParallel is not needed
    # when a module doesn't have any parameter that requires a gradient"), so
    # only the model(s) --trained_model actually unfreezes get prepared. The
    # frozen one stays a plain per-process replica already sitting on
    # accelerator.device, which is all it needs to be — it's only ever run
    # forward, identically on every rank.
    if train_rdd and train_lg:
        rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            rdd, lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    elif train_rdd:
        rdd, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            rdd, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    else:
        lg, optimizer, train_loader, mini_train_loader, mini_val_loader = accelerator.prepare(
            lg, optimizer, train_loader, mini_train_loader, mini_val_loader
        )
    eval_lg = ema_lg if ema_lg is not None else lg

    # Computed only now (post-prepare) so len(train_loader) reflects each
    # process's actual per-rank step count, not the pre-shard full dataset —
    # using the latter would understate how many EMA updates each rank really
    # performs per epoch on a multi-GPU run and leave the reference decaying
    # slower than --epochs was meant to produce.
    if distill_active and args.distill_model == "ema" and args.distill_ema_decay is None:
        args.distill_ema_decay = compute_distill_ema_decay(args.epochs * len(train_loader))
        accelerator.print(
            f"[distill] auto distill_ema_decay={args.distill_ema_decay:.6f} "
            f"(epochs={args.epochs}, steps/epoch={len(train_loader)})"
        )

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

    # ── healing_on_positives reference scores (before any training) ──
    # Has to run here, before the first optimizer step, for distill_lg_ref to
    # still be the pretrained model — under --distill_model ema it starts
    # drifting towards the student as soon as training begins, and the gate
    # this table feeds is specifically "worse than where I started".
    pretrained_pos_scores: dict[tuple[str, str], float] | None = None
    if distill_active and args.distill_signal_type == "healing_on_positives":
        accelerator.print(
            "[healing] scoring every (query, positive) pair in the train index "
            "with the pretrained LightGlue..."
        )
        pretrained_pos_scores = measure_pretrained_positive_scores(
            accelerator, rdd, distill_lg_ref, eval_train_loader, args
        )
        accelerator.print(
            f"[healing] reference confidence for {len(pretrained_pos_scores)} "
            f"(query, positive) pairs"
        )

    # ── baseline eval (before any training) ──
    global_step = 0
    baseline_train = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_train_loader, args, prefix="train_eval")
    baseline_val   = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_val_loader,   args, prefix="val")
    _unwrap(rdd).train(train_rdd)
    lg.train(train_lg)
    if accelerator.is_main_process:
        accelerator.log(
            {**baseline_train, **baseline_val, "epoch": -1, "train/random_negative_prob": train_ds.random_negative_prob},
            step=global_step,
        )

    # ── loop ──
    prev_dead_pos_index: set[str] | None = None
    live_margin = args.lg_margin
    for epoch in range(args.epochs):
        epoch_loss, global_step, neg_gap_stats, prev_dead_pos_index, epoch_extras = train_epoch_lg(
            accelerator, rdd, lg, eval_lg, optimizer, train_loader,
            mini_train_loader, mini_val_loader,
            epoch, args.epochs, args, global_step,
            ema_lg=ema_lg,
            distill_lg_ref=distill_lg_ref,
            distill_rdd_ref=distill_rdd_ref,
            prev_dead_pos_index=prev_dead_pos_index,
            pretrained_pos_scores=pretrained_pos_scores,
            margin=live_margin,
        )

        # --hard_positive_sampling / --hard_negative_sampling: fold this
        # epoch's observed pair confidences into the sampling EMA. Same
        # every-rank-identical-update contract as update_mining_stats below.
        if epoch_extras["pair_observations"]:
            train_ds.update_pair_stats(epoch_extras["pair_observations"])

        # --adaptive_margin: track the model's own separation gap so the
        # margin keeps a meaningful fraction of triplets active as the easy
        # ones get solved. epoch_extras["sep_gap"] is identical on every rank
        # (built from all-reduced step means), so live_margin stays in sync.
        if args.adaptive_margin:
            live_margin = min(
                max(epoch_extras["sep_gap"] + args.adaptive_margin_offset, args.lg_margin),
                args.adaptive_margin_max,
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
            train_eval_metrics = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_train_loader, args, prefix="train_eval")
            val_metrics        = eval_pseudo_accuracy(accelerator, rdd, eval_lg, eval_val_loader,   args, prefix="val")
        epoch_eval_time = time.perf_counter() - t_eval_start
        _unwrap(rdd).train(train_rdd)
        lg.train(train_lg)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        metrics = {
            "epoch":                      epoch,
            "train/epoch_loss":           epoch_loss,
            "train/lr":                   lr,
            "time/epoch_eval_s":          epoch_eval_time,
            # Logged unconditionally (not just when moving_neg/* fires) so the
            # wandb curve stays continuous even on epochs with zero sampled
            # random negatives, and so --negative_mining alone (static prob,
            # no --moving_negative_prob) still shows what was actually in use.
            "train/random_negative_prob": train_ds.random_negative_prob,
            # Same logic: constant without --adaptive_margin, but always
            # logged so runs stay comparable. train/margin is the value the
            # NEXT epoch will train at; sep_gap is what this epoch measured.
            "train/margin":               live_margin,
            "train/sep_gap":              epoch_extras["sep_gap"],
        }
        if do_eval:
            metrics.update(train_eval_metrics)
            metrics.update(val_metrics)

        if accelerator.is_main_process:
            accelerator.log(metrics, step=global_step)

        # save_state has to run on *every* process, not just the main one: it
        # writes a per-process random_states_{rank}.pkl (RNG state + step
        # counter) alongside the shared weights. Under is_main_process only
        # rank 0's file is written, and load_state swallows the missing ones in
        # a try/except — so a resume doesn't fail, it silently restarts ranks
        # 1..N-1 from whatever RNG state they happen to be in. That's the worse
        # failure mode: quietly non-reproducible, divergent resumes. The EMA
        # copy is identical on every rank (DDP keeps `lg` in sync), so that one
        # file is still written once.
        ckpt_dir = args.output_dir / f"epoch_{epoch:02d}"
        accelerator.save_state(str(ckpt_dir))
        if accelerator.is_main_process and ema_lg is not None:
            torch.save(ema_lg.state_dict(), ckpt_dir / "ema_lg.pt")

    if args.project:
        accelerator.end_training()


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_training_lg(args)


if __name__ == "__main__":
    main()
