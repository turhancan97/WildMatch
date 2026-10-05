"""
Regression tests pinning the dense match scoring now used in training/eval
against the ragged implementation it replaced.

The production path is LightGlueForTraining (rdd_patch/lightglue_masked_training.py)
+ train_common._lg_scores + train_by_lg_matches.lg_confidence_loss, all of which
read the dense `matching_scores0` / `valid0`. The reference path — frozen
verbatim copies of the pre-change code, kept in this file — is LightGlueMasked
(rdd_patch/lightglue_masked.py, untouched) + the ragged `scores` list. Both are
run over the same batches from the same frozen checkpoint and must agree.

Why the rewrite happened: LightGlue reports its matches as a ragged,
per-batch-item list holding only the entries that cleared `filter_threshold`. A
pair with no surviving match yields an empty tensor, and `_lg_scores` used to
substitute a detached `torch.zeros(())` for it. A batch where every positive was
empty then produced a loss with no grad_fn at all — backward fired no gradient
hook, so under DDP the reducer never finished the iteration and the next step
aborted with "Expected to have finished reduction in the prior iteration before
starting a new one". `matching_scores0 * valid0` selects exactly the same
entries but keeps a live graph, because masking with an all-False mask still
propagates zeros to every parameter.

Since 2026-09-29 two things are intentionally different, so the comparison is
narrowed accordingly:
  - LightGlueForTraining masks padded keypoints in its final assignment, while
    the frozen LightGlueMasked reference does not. The two paths are therefore
    compared only on *pad-free* pairs (both images at the batch's full keypoint
    count); padded pairs are counted and reported, not asserted.
  - The training loss is now the relaxed objective (lg_confidence_loss no
    longer drops empty positives), so loss equality with the ragged reference
    is no longer checked. Its behaviour is pinned by
    tests/test_relaxed_objective.py, which also replaces the two synthetic
    checks that used to live at the end of this file.
The filtered `_lg_scores` (still the evaluation score) must keep matching the
ragged reference exactly on pad-free pairs.

Two tests fed by a single pass over the dataloader:

  1. test_positive_pairs — on the index's own positive pairs the two paths must
     produce identical confidences, match counts, loss and logged stats. Also
     asserts every positive pair has at least one match: the index is built by
     retrieval against this same frozen checkpoint, so a positive with zero
     matches means the run is not configured the way the index was built.

  2. test_empty_matches — the pairs that produce no matches at all (in practice:
     random negatives) must have an all-False `valid0`, a dense confidence of
     exactly 0.0, and must be exactly the set the reference path skipped. Then
     two synthetic checks: that a sample with an empty *positive* contributes
     nothing to the loss (the weight-0 path that replaced `continue`), and that
     an all-empty batch scores 0.0 while keeping a grad_fn backward can follow
     to the weights.

Usage (single GPU is plenty — the data pass is forward-only):

    python -m tests.test_dense_matching \\
        --index index-rgb-8382/top_k=5_top_m=10_train_combined.json \\
        --data_root /shared/.../lynx-ds-Jul-20 \\
        --batches 100

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from contrastive_finetuning.loading import IndexAssignedTripletDataset, get_loader
from contrastive_finetuning.models import build_masked_lg
from contrastive_finetuning.train_by_lg_matches import lg_confidence_loss
from contrastive_finetuning.train_common import (
    _lg_scores, batch_features, extract_train, resize_long_side, seed_all,
)
from rdd.RDD.RDD import build as build_rdd_from_conf
from rdd.RDD.utils import read_config
from rdd_patch.lightglue_masked import LightGlueMasked

# A data-driven CLI script (see Usage above), not a pytest module: its test_*
# functions take the collected pass as arguments, so pytest must not collect them.
__test__ = False


# ── frozen reference: the ragged implementations the dense path replaced ──────
# Verbatim copies of train_common._lg_scores and
# train_by_lg_matches.lg_confidence_loss as they were before the rewrite. They
# live here, not in the production modules, so these tests keep comparing
# against fixed historical behaviour rather than against themselves.
def _lg_scores_ragged(pred: dict, q_data: dict, g_data: dict, device: torch.device) -> torch.Tensor:
    B = q_data["keypoints"].shape[0]
    sums = torch.stack([
        pred["scores"][i].sum() if pred["scores"][i].numel() > 0 else torch.zeros((), device=device)
        for i in range(B)
    ])
    n_q = q_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    n_g = g_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    return sums / torch.minimum(n_q, n_g)


def lg_confidence_loss_ragged(
    pred_pos: dict, pred_neg: dict, margin: float, device: torch.device,
    data_a: dict, data_p: dict, data_n: dict, weak_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    pos_conf_all = _lg_scores_ragged(pred_pos, data_a, data_p, device)
    neg_conf_all = _lg_scores_ragged(pred_neg, data_a, data_n, device)

    if weak_mask is not None:
        pos_conf_all = torch.where(weak_mask, pos_conf_all.detach(), pos_conf_all)

    losses = []
    pos_skipped: list[bool] = []
    neg_skipped: list[bool] = []
    pos_match_list, neg_match_list = [], []
    pos_conf_list, neg_conf_list = [], []

    for i, (s_pos, s_neg) in enumerate(zip(pred_pos["scores"], pred_neg["scores"])):
        pos_empty = s_pos.shape[0] == 0
        neg_empty = s_neg.shape[0] == 0
        pos_skipped.append(pos_empty)
        neg_skipped.append(neg_empty)

        if pos_empty:
            continue

        pos_match_list.append(s_pos.shape[0])
        neg_match_list.append(s_neg.shape[0])

        pos_conf = pos_conf_all[i]
        pos_conf_list.append(pos_conf.item())

        if not neg_empty:
            neg_conf = neg_conf_all[i]
            neg_conf_list.append(neg_conf.item())
            losses.append(F.relu(margin - pos_conf + neg_conf))
        else:
            losses.append(F.relu(margin - pos_conf))

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    stats = {
        "pos_skipped":      pos_skipped,
        "neg_skipped":      neg_skipped,
        "mean_pos_matches": _mean(pos_match_list),
        "mean_neg_matches": _mean(neg_match_list),
        "mean_pos_conf":    _mean(pos_conf_list),
        "mean_neg_conf":    _mean(neg_conf_list),
    }
    if not losses:
        return torch.zeros(1, device=device, requires_grad=True).squeeze(), stats
    return torch.stack(losses).mean(), stats


# ── harness ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("index-rgb-8382/top_k=5_top_m=10_train_combined.json"))
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--rdd_weights", type=str, default="rdd/weights/RDD-v2.pth")
    p.add_argument("--lg_weights", type=str, default="rdd/weights/RDD_lg-v2.pth")
    p.add_argument("--batches", type=int, default=100, help="How many dataloader batches to compare")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--resize", type=int, default=512, help="Must match how the index was built")
    p.add_argument("--top_k", type=int, default=512, help="Must match how the index was built")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lg_margin", type=float, default=0.5)
    p.add_argument(
        "--random_negative_prob", type=float, default=0.5,
        help="Random (rather than index-mined) negatives are where zero-match pairs "
             "actually show up, so test_empty_matches needs this above 0",
    )
    p.add_argument(
        "--filter_threshold", type=float, default=0.01,
        help="LightGlue match threshold. The default is the production value "
             "(models.build_masked_lg); raise it to force zero-match pairs so "
             "test_empty_matches can be exercised on a small sample instead of "
             "waiting for one to occur naturally",
    )
    p.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for confidences and loss")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_rdd_on(weights: str, device: torch.device, top_k: int):
    """models.build_rdd, but honouring `device`.

    rdd/configs/default.yaml hardcodes `device: cuda` and RDD.build moves the
    model there before build_rdd gets a chance to override it, so the stock
    helper cannot be run on a CPU-only box. Same model either way.
    """
    conf = {**read_config("rdd/configs/default.yaml"), "device": str(device)}
    model = build_rdd_from_conf(conf, weights=str(weights))
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    model.to(device)
    model.eval()
    return model


def build_reference_lg(device: torch.device, weights: str, filter_threshold: float) -> LightGlueMasked:
    """The pre-change matcher, under the same config models.build_masked_lg uses."""
    conf = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": filter_threshold,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": weights,
        "detach_descriptors": True,
    }
    return LightGlueMasked("rdd", **conf).to(device).eval()


@torch.no_grad()
def collect(args: argparse.Namespace) -> dict:
    """One pass over `--batches` batches, running both paths on identical inputs."""
    device = torch.device(args.device)
    seed_all(args.seed)

    ds = IndexAssignedTripletDataset(
        args.index, root=args.data_root, transform=transforms.ToTensor(),
        random_negative_prob=args.random_negative_prob, return_meta=True,
    )
    loader = get_loader(
        ds, batch_size=args.batch_size, shuffle=True, seed=args.seed,
        num_workers=args.num_workers, persistent_workers=False,
    )

    rdd = build_rdd_on(args.rdd_weights, device, args.top_k)
    lg_ref = build_reference_lg(device, args.lg_weights, args.filter_threshold)
    lg_new = build_masked_lg(device, weights=args.lg_weights, init_threshold=args.filter_threshold)
    # Rules out "the two classes loaded different weights" as an explanation for
    # any disagreement below.
    lg_new.load_state_dict(lg_ref.state_dict())

    out = {
        "pos_conf_ref": [], "pos_conf_new": [],
        "neg_conf_ref": [], "neg_conf_new": [],
        "pos_matches_ref": [], "pos_matches_new": [],
        "neg_matches_ref": [], "neg_matches_new": [],
        "pos_empty_ref": [], "pos_empty_new": [],
        "neg_empty_ref": [], "neg_empty_new": [],
        "loss_ref": [], "loss_new": [],
        "stats_ref": [], "stats_new": [],
        "neg_source": [],
        "batch_of_sample": [],
        "pos_pad_free": [], "neg_pad_free": [],
        "n_batches": 0, "n_samples": 0,
    }

    pbar = tqdm(enumerate(loader), total=args.batches, desc="comparing")
    for step, (anchors, positives, negatives, neg_meta) in pbar:
        if step >= args.batches:
            break

        anchors_r   = resize_long_side(anchors,   args.resize).to(device)
        positives_r = resize_long_side(positives, args.resize).to(device)
        negatives_r = resize_long_side(negatives, args.resize).to(device)
        H_r, W_r = anchors_r.shape[-2:]

        feats_a = extract_train(rdd, anchors_r)
        feats_p = extract_train(rdd, positives_r)
        feats_n = extract_train(rdd, negatives_r)

        data_a = batch_features(feats_a, H_r, W_r)
        data_p = batch_features(feats_p, H_r, W_r)
        data_n = batch_features(feats_n, H_r, W_r)

        # Same features into both matchers, so any difference is the scoring
        # path and nothing else.
        pred_pos_ref = lg_ref({"image0": data_a, "image1": data_p})
        pred_neg_ref = lg_ref({"image0": data_a, "image1": data_n})
        pred_pos_new = lg_new({"image0": data_a, "image1": data_p})
        pred_neg_new = lg_new({"image0": data_a, "image1": data_n})

        loss_ref, stats_ref = lg_confidence_loss_ragged(
            pred_pos_ref, pred_neg_ref, args.lg_margin, device,
            data_a=data_a, data_p=data_p, data_n=data_n,
        )
        loss_new, stats_new = lg_confidence_loss(
            pred_pos_new, pred_neg_new, args.lg_margin,
            data_a=data_a, data_p=data_p, data_n=data_n,
        )

        B = anchors_r.shape[0]
        full_a = data_a["masks"].reshape(B, -1).all(dim=1)
        out["pos_pad_free"].append((full_a & data_p["masks"].reshape(B, -1).all(dim=1)).cpu())
        out["neg_pad_free"].append((full_a & data_n["masks"].reshape(B, -1).all(dim=1)).cpu())
        out["pos_conf_ref"].append(_lg_scores_ragged(pred_pos_ref, data_a, data_p, device).cpu())
        out["neg_conf_ref"].append(_lg_scores_ragged(pred_neg_ref, data_a, data_n, device).cpu())
        out["pos_conf_new"].append(_lg_scores(pred_pos_new, data_a, data_p).cpu())
        out["neg_conf_new"].append(_lg_scores(pred_neg_new, data_a, data_n).cpu())

        out["pos_matches_ref"].append(torch.tensor([s.numel() for s in pred_pos_ref["scores"]]))
        out["neg_matches_ref"].append(torch.tensor([s.numel() for s in pred_neg_ref["scores"]]))
        out["pos_matches_new"].append(pred_pos_new["valid0"].sum(dim=1).cpu())
        out["neg_matches_new"].append(pred_neg_new["valid0"].sum(dim=1).cpu())

        out["pos_empty_ref"].append(torch.tensor(stats_ref["pos_skipped"]))
        out["neg_empty_ref"].append(torch.tensor(stats_ref["neg_skipped"]))
        out["pos_empty_new"].append(torch.tensor(stats_new["pos_skipped"]))
        out["neg_empty_new"].append(torch.tensor(stats_new["neg_skipped"]))

        out["loss_ref"].append(float(loss_ref))
        out["loss_new"].append(float(loss_new))
        out["stats_ref"].append(stats_ref)
        out["stats_new"].append(stats_new)
        out["neg_source"].extend(neg_meta["neg_source"])
        out["batch_of_sample"].append(torch.full((B,), step))
        out["n_batches"] += 1
        out["n_samples"] += B

    for k, v in out.items():
        if isinstance(v, list) and v and torch.is_tensor(v[0]):
            out[k] = torch.cat(v)
    return out


# ── tests ─────────────────────────────────────────────────────────────────────
def test_positive_pairs(r: dict, args: argparse.Namespace) -> None:
    """Dense and ragged paths agree on the index's positive pairs."""
    print("\n=== test_positive_pairs ===")

    # The index is built by retrieval against this same frozen checkpoint, so a
    # positive with no matches means the run is not configured the way the
    # index was built.
    bad = torch.where(r["pos_empty_ref"])[0]
    assert bad.numel() == 0, (
        f"{bad.numel()}/{r['n_samples']} index positive pairs produced ZERO matches "
        f"(first offenders at flat sample idx {bad[:10].tolist()}, "
        f"batches {r['batch_of_sample'][bad[:10]].tolist()}). The index is built by "
        f"retrieval on this checkpoint, so this should not happen — check that "
        f"--resize ({args.resize}) / --top_k ({args.top_k}) / --lg_weights / "
        f"--filter_threshold ({args.filter_threshold}) match how index {args.index} was built."
    )
    print(f"  all {r['n_samples']} positive pairs have >=1 match "
          f"(min {int(r['pos_matches_ref'].min())}, mean {float(r['pos_matches_ref'].float().mean()):.1f})")

    pf = r["pos_pad_free"]
    nf = r["neg_pad_free"]
    assert bool(pf.any()), "no pad-free positive pair occurred; lower --batch_size (1 makes every pair pad-free)"
    print(f"  comparing {int(pf.sum())}/{r['n_samples']} pad-free positive and "
          f"{int(nf.sum())} pad-free negative pairs (padded pairs are masked only in the training path)")

    assert torch.equal(r["pos_matches_ref"][pf], r["pos_matches_new"][pf]), (
        "positive-pair match counts differ on pad-free pairs: "
        f"{int((r['pos_matches_ref'][pf] != r['pos_matches_new'][pf]).sum())} samples disagree"
    )
    print("  match counts identical on every pad-free pair")

    d_pos = (r["pos_conf_ref"][pf] - r["pos_conf_new"][pf]).abs()
    assert float(d_pos.max()) <= args.atol, (
        f"positive-pair confidence differs by up to {float(d_pos.max()):.3e} > atol {args.atol}"
    )
    print(f"  filtered confidence agrees, max |ref-new| = {float(d_pos.max()):.3e} (atol {args.atol})")

    if bool(nf.any()):
        d_neg = (r["neg_conf_ref"][nf] - r["neg_conf_new"][nf]).abs()
        assert float(d_neg.max()) <= args.atol, (
            f"negative-pair confidence differs by up to {float(d_neg.max()):.3e} > atol {args.atol}"
        )
        print(f"  negative-pair confidence agrees too, max |ref-new| = {float(d_neg.max()):.3e}")

    padded = ~pf
    if bool(padded.any()):
        d_pad = (r["pos_conf_ref"][padded] - r["pos_conf_new"][padded]).abs()
        print(f"  padded positive pairs (not asserted): max |ref-new| = {float(d_pad.max()):.3e} "
              "- the reference still lets padding into its assignment")


def test_empty_matches(r: dict, args: argparse.Namespace) -> None:
    """Zero-match pairs: same set as before, empty valid mask, zero confidence."""
    print("\n=== test_empty_matches ===")

    n_empty = int(r["neg_empty_ref"].sum())
    assert n_empty > 0, (
        f"no zero-match pair occurred in {r['n_batches']} batches, so this test proved "
        f"nothing. Raise --batches (now {args.batches}) or --random_negative_prob "
        f"(now {args.random_negative_prob})."
    )
    src = [s for s, e in zip(r["neg_source"], r["neg_empty_ref"].tolist()) if e]
    print(f"  {n_empty}/{r['n_samples']} negative pairs have zero matches "
          f"({src.count('random')} random-drawn, {src.count('index')} index-mined)")

    nf, pf = r["neg_pad_free"], r["pos_pad_free"]
    assert torch.equal(r["neg_empty_ref"][nf], r["neg_empty_new"][nf]), (
        f"the zero-match negative pairs are not the same set: "
        f"{int((r['neg_empty_ref'][nf] != r['neg_empty_new'][nf]).sum())} pad-free samples disagree"
    )
    assert torch.equal(r["pos_empty_ref"][pf], r["pos_empty_new"][pf]), (
        f"the zero-match positive pairs are not the same set: "
        f"{int((r['pos_empty_ref'][pf] != r['pos_empty_new'][pf]).sum())} pad-free samples disagree"
    )
    print("  on pad-free pairs, empty(valid0) picks out exactly the reference's empty pairs, pos and neg")

    empty = r["neg_empty_new"]
    assert int(r["neg_matches_new"][empty].sum()) == 0, "an 'empty' pair has a non-empty valid0"
    assert float(r["neg_conf_new"][empty].abs().max()) == 0.0, (
        f"a zero-match negative scored non-zero confidence "
        f"(max {float(r['neg_conf_new'][empty].abs().max()):.3e})"
    )
    print("  their valid0 is all-False and their filtered confidence is exactly 0.0")


def main() -> None:
    args = parse_args()
    print(f"device={args.device} batches={args.batches} batch_size={args.batch_size} "
          f"resize={args.resize} top_k={args.top_k} random_negative_prob={args.random_negative_prob} "
          f"filter_threshold={args.filter_threshold}")
    r = collect(args)
    test_positive_pairs(r, args)
    test_empty_matches(r, args)
    print(f"\nOK — dense filtered scoring matches the frozen ragged reference on pad-free pairs over "
          f"{r['n_batches']} batches / {r['n_samples']} samples.")


if __name__ == "__main__":
    main()
