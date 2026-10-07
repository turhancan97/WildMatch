"""Regression tests for the shared RDD/LoMa training objective (2026-09-29).

Covers the pieces that make RDD-LightGlue train exactly like LoMa:
  - `_lg_relaxed_scores` equals LoMa's `train_pair_score` formula;
  - LightGlueForTraining excludes padded keypoints from its assignment, so a
    padded batch scores a pair like the pair on its own;
  - the margin loss keeps every triplet (no empty-positive drop) and still
    trains pairs for which no match survives the inference filters;
  - --grad_accum_steps / --resume fail closed with unsupported features, and
    resume refuses checkpoints from a different configuration;
  - accelerate checkpoints now carry the LR scheduler, so resume continues it.

Tests needing the pretrained LightGlue weights are skipped when they are absent.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from wildmatch.matcher_finetune.loma_backend import train_pair_score
from wildmatch.matcher_finetune.rdd_patch.lightglue_masked_training import (
    MASKED_LOG_PROB,
    double_softmax,
    sigmoid_log_double_softmax,
    valid_pair_mask,
)
from wildmatch.matcher_finetune.train_by_lg_matches import (
    OPTIMIZER_ID,
    TRAIN_STATE_FILE,
    accumulation_and_resume_errors,
    lg_confidence_loss,
    read_resume_state,
    run_training_lg,
    train_state_identity,
    write_train_state,
)
from wildmatch.matcher_finetune.train_common import (
    TRAINING_SCORE_ID,
    _lg_relaxed_scores,
    _lg_scores,
    batch_features,
)


def _lg_weights() -> Path:
    from wildmatch.paths import path

    folder = path("external.rdd_weights_dir")
    return Path("/nonexistent") if folder is None else folder / "RDD_lg-v2.pth"


LG_WEIGHTS = _lg_weights()
needs_weights = pytest.mark.skipif(not LG_WEIGHTS.is_file(), reason="pretrained LightGlue weights not available")


def _masks(counts: list[int], length: int) -> torch.Tensor:
    mask = torch.zeros(len(counts), 1, length, 1, dtype=torch.bool)
    for b, n in enumerate(counts):
        mask[b, :, :n] = True
    return mask


def _random_log_assignment(b: int, m: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    sim = torch.randn(b, m, n, generator=g) * 3
    z0 = torch.randn(b, m, 1, generator=g)
    z1 = torch.randn(b, n, 1, generator=g)
    return sigmoid_log_double_softmax(sim, z0, z1)


# ── relaxed score ──────────────────────────────────────────────────────────────
def test_relaxed_score_matches_loma_formula_on_unpadded_input():
    scores = _random_log_assignment(3, 7, 5)

    class Fixed(nn.Module):
        def forward(self, *_):
            return {"scores": scores}

    expected = train_pair_score(Fixed(), None, None, None, None)
    got = _lg_relaxed_scores(
        {"assignment_scores": scores}, {"masks": _masks([7] * 3, 7)}, {"masks": _masks([5] * 3, 5)}
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-7)


def test_relaxed_score_ignores_padded_rows_and_columns():
    scores = _random_log_assignment(1, 6, 4, seed=1)
    ref = _lg_relaxed_scores({"assignment_scores": scores}, {"masks": _masks([6], 6)}, {"masks": _masks([4], 4)})

    # Append 3 padded rows and 2 padded columns carrying large (bogus) mass.
    padded = torch.zeros(1, 10, 7)
    padded[:, :6, :4] = scores[:, :6, :4]
    padded[:, 6:9, :] = 0.0  # log p = 0 -> p = 1 in padded rows
    padded[:, :, 4:6] = 0.0  # and in padded columns
    padded[:, :6, -1] = scores[:, :6, -1]
    padded[:, -1, :4] = scores[:, -1, :4]
    got = _lg_relaxed_scores({"assignment_scores": padded}, {"masks": _masks([6], 9)}, {"masks": _masks([4], 6)})
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-7)


def test_relaxed_score_is_zero_with_graph_when_a_side_has_no_keypoints():
    param = torch.zeros((), requires_grad=True)
    scores = torch.zeros(2, 1, 5) + param  # (B, M+1, N+1) with M == 0, anchored to a parameter
    got = _lg_relaxed_scores({"assignment_scores": scores}, {"masks": _masks([0, 0], 0)}, {"masks": _masks([4, 4], 4)})
    assert torch.equal(got, torch.zeros(2))
    assert got.requires_grad
    got.sum().backward()
    assert param.grad is not None


# ── padding mask inside the assignment ────────────────────────────────────────
def test_assignment_mask_is_bit_identical_without_padding():
    g = torch.Generator().manual_seed(2)
    sim, z0, z1 = (
        torch.randn(2, 5, 6, generator=g),
        torch.randn(2, 5, 1, generator=g),
        torch.randn(2, 6, 1, generator=g),
    )
    all_valid0, all_valid1 = torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 6, dtype=torch.bool)
    assert torch.equal(
        sigmoid_log_double_softmax(sim, z0, z1), sigmoid_log_double_softmax(sim, z0, z1, all_valid0, all_valid1)
    )
    assert torch.equal(double_softmax(sim), double_softmax(sim, valid_pair_mask(sim, all_valid0, all_valid1)))


def test_assignment_mask_equals_the_unpadded_computation():
    g = torch.Generator().manual_seed(3)
    sim, z0, z1 = (
        torch.randn(1, 5, 4, generator=g),
        torch.randn(1, 5, 1, generator=g),
        torch.randn(1, 4, 1, generator=g),
    )
    ref = sigmoid_log_double_softmax(sim, z0, z1)

    sim_p = torch.cat(
        [torch.cat([sim, torch.randn(1, 5, 2, generator=g) * 5], 2), torch.randn(1, 3, 6, generator=g) * 5], 1
    )
    z0_p = torch.cat([z0, torch.randn(1, 3, 1, generator=g)], 1)
    z1_p = torch.cat([z1, torch.randn(1, 2, 1, generator=g)], 1)
    valid0 = torch.tensor([[True] * 5 + [False] * 3])
    valid1 = torch.tensor([[True] * 4 + [False] * 2])
    got = sigmoid_log_double_softmax(sim_p, z0_p, z1_p, valid0, valid1)

    torch.testing.assert_close(got[:, :5, :4], ref[:, :5, :4], rtol=0, atol=1e-6)
    torch.testing.assert_close(got[:, :5, -1], ref[:, :5, -1], rtol=0, atol=1e-6)
    torch.testing.assert_close(got[:, -1, :4], ref[:, -1, :4], rtol=0, atol=1e-6)
    assert torch.isfinite(got).all()
    # Every entry touching padding carries exactly zero probability.
    assert float(got[:, 5:8, :].exp().max()) == 0.0
    assert float(got[:, :, 4:6].exp().max()) == 0.0
    assert float(torch.tensor(MASKED_LOG_PROB).exp()) == 0.0


@needs_weights
def test_lightglue_never_matches_padding_and_scores_padded_pairs_like_unpadded():
    from wildmatch.matcher_finetune.models import build_masked_lg

    lg = build_masked_lg(torch.device("cpu"), weights=str(LG_WEIGHTS))
    g = torch.Generator().manual_seed(4)
    k = torch.rand(40, 2, generator=g) * 500
    d = F.normalize(torch.randn(40, 256, generator=g), dim=-1)
    near = {"keypoints": k + 1, "descriptors": F.normalize(d + 0.05 * torch.randn(40, 256, generator=g), dim=-1)}
    short = {"keypoints": k[:30], "descriptors": d[:30]}  # forces padding of the first batch item

    with torch.no_grad():
        single = lg({"image0": batch_features([short], 512, 512), "image1": batch_features([near], 512, 512)})
        data0 = batch_features([short, {"keypoints": k, "descriptors": d}], 512, 512)
        data1 = batch_features([near, near], 512, 512)
        batched = lg({"image0": data0, "image1": data1})

    assert not bool(batched["valid0"][0, 30:].any()), "a padded keypoint was matched"
    assert float(batched["assignment_scores"][0, 30:40, :].exp().max()) == 0.0
    s_single = _lg_relaxed_scores(single, batch_features([short], 512, 512), batch_features([near], 512, 512))
    s_batched = _lg_relaxed_scores(batched, data0, data1)[:1]
    # Only the transformer's masked attention differs numerically (pre-existing, ~1e-4).
    torch.testing.assert_close(s_batched, s_single, rtol=0, atol=2e-3)


# ── margin loss ────────────────────────────────────────────────────────────────
def _pred_from_probs(probs: torch.Tensor, valid0: torch.Tensor | None = None) -> dict:
    b, m, n = probs.shape
    scores = torch.full((b, m + 1, n + 1), MASKED_LOG_PROB)
    scores[:, :m, :n] = probs.clamp_min(1e-12).log()
    if valid0 is None:
        valid0 = torch.zeros(b, m, dtype=torch.bool)
    return {"assignment_scores": scores, "valid0": valid0, "matching_scores0": torch.zeros(b, m)}


def test_margin_loss_keeps_every_triplet_including_filtered_empty_positives():
    margin = 0.5
    data = {"masks": _masks([2, 2, 2], 2)}
    pos = torch.tensor([[[0.9, 0.1], [0.2, 0.7]], [[0.3, 0.3], [0.3, 0.3]], [[0.6, 0.1], [0.1, 0.6]]])
    neg = torch.tensor([[[0.2, 0.1], [0.1, 0.2]], [[0.5, 0.1], [0.1, 0.5]], [[0.1, 0.1], [0.1, 0.1]]])
    valid_pos = torch.tensor([[True, True], [False, False], [True, True]])  # sample 1: nothing survives filters
    loss, stats = lg_confidence_loss(
        _pred_from_probs(pos, valid_pos), _pred_from_probs(neg), margin, data_a=data, data_p=data, data_n=data
    )

    def relaxed(p):
        return 0.5 * (p.max(dim=2).values.mean(dim=1) + p.max(dim=1).values.mean(dim=1))

    expected = F.relu(margin - relaxed(pos) + relaxed(neg)).mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)
    assert stats["pos_skipped"] == [False, True, False]  # diagnostics only

    # The filtered-empty positive now moves the loss.
    pos2 = pos.clone()
    pos2[1] = 0.05
    loss2, _ = lg_confidence_loss(
        _pred_from_probs(pos2, valid_pos), _pred_from_probs(neg), margin, data_a=data, data_p=data, data_n=data
    )
    assert float(loss2) > float(loss)


@needs_weights
def test_pairs_without_surviving_matches_still_train():
    from wildmatch.matcher_finetune.models import build_masked_lg

    lg = build_masked_lg(torch.device("cpu"), weights=str(LG_WEIGHTS), init_threshold=1.0)  # nothing survives
    lg.train()
    torch.manual_seed(5)

    def feats(n=24):
        return [
            {"keypoints": torch.rand(n, 2) * 400, "descriptors": F.normalize(torch.randn(n, 256), dim=-1)}
            for _ in range(2)
        ]

    data_a, data_p, data_n = (batch_features(feats(), 512, 512) for _ in range(3))
    pred_pos = lg({"image0": data_a, "image1": data_p})
    pred_neg = lg({"image0": data_a, "image1": data_n})
    assert not bool(pred_pos["valid0"].any()) and not bool(pred_neg["valid0"].any())
    assert float(_lg_scores(pred_pos, data_a, data_p).abs().max()) == 0.0  # filtered score is dead

    loss, _ = lg_confidence_loss(pred_pos, pred_neg, 0.5, data_a=data_a, data_p=data_p, data_n=data_n)
    lg.zero_grad()
    loss.backward()
    grads = [p.grad for p in lg.log_assignment[-1].parameters()]
    assert all(grad is not None for grad in grads)
    assert any(float(grad.abs().max()) > 0 for grad in grads), "relaxed loss produced no gradient"


def test_loss_is_a_plain_batch_mean_so_accumulation_is_exact():
    """grad(mean over 4) == sum over 4 single-sample chunks of grad(loss_chunk / 4)."""
    torch.manual_seed(6)
    logits_pos = torch.randn(4, 3, 3, requires_grad=True)
    logits_neg = torch.randn(4, 3, 3, requires_grad=True)
    data = {"masks": _masks([3] * 4, 3)}

    def loss_for(idx):
        sub = {"masks": data["masks"][idx]}
        pos = _pred_from_probs(logits_pos[idx].softmax(-1))
        neg = _pred_from_probs(logits_neg[idx].softmax(-1))
        return lg_confidence_loss(pos, neg, 0.9, data_a=sub, data_p=sub, data_n=sub)[0]

    full = torch.autograd.grad(loss_for(slice(0, 4)), (logits_pos, logits_neg))
    acc = [torch.zeros_like(logits_pos), torch.zeros_like(logits_neg)]
    for i in range(4):
        grads = torch.autograd.grad(loss_for(slice(i, i + 1)) / 4, (logits_pos, logits_neg))
        acc = [a + g for a, g in zip(acc, grads)]
    for f, a in zip(full, acc):
        torch.testing.assert_close(f, a, rtol=0, atol=1e-7)


# ── optimizer, accumulation guards, resume ─────────────────────────────────────
def _args(**overrides) -> argparse.Namespace:
    base = dict(
        grad_accum_steps=1,
        ema_decay=0.0,
        distill_model="none",
        warmup_steps=0,
        resume=None,
        moving_negative_prob=None,
        negative_mining=False,
        hard_positive_sampling=False,
        hard_negative_sampling=False,
        adaptive_margin=False,
        eval_only=False,
        trained_model="lg",
        rdd_train_component="all",
        batch_size=8,
        epochs=300,
        lr=1e-5,
        weight_decay=1e-4,
        lg_margin=0.5,
        train_index=Path("train.json"),
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_trainer_uses_adamw():
    source = inspect.getsource(run_training_lg)
    assert "torch.optim.AdamW(" in source and "torch.optim.Adam(" not in source
    assert OPTIMIZER_ID == "adamw" and TRAINING_SCORE_ID == "relaxed_v1"


def test_scheduler_is_registered_for_checkpointing():
    assert "register_for_checkpointing(scheduler)" in inspect.getsource(run_training_lg)


@pytest.mark.parametrize(
    "override",
    [
        {"ema_decay": 0.999},
        {"distill_model": "pretrained"},
        {"warmup_steps": 10},
    ],
)
def test_accumulation_rejects_per_step_features(override):
    assert accumulation_and_resume_errors(_args(grad_accum_steps=8, **override))


def test_accumulation_and_resume_allowed_for_plain_margin_loss():
    assert accumulation_and_resume_errors(_args(grad_accum_steps=8, resume=Path("x"))) == []
    assert accumulation_and_resume_errors(_args(grad_accum_steps=0))


@pytest.mark.parametrize(
    "override",
    [
        {"moving_negative_prob": 0.3},
        {"negative_mining": True},
        {"hard_positive_sampling": True},
        {"hard_negative_sampling": True},
        {"adaptive_margin": True},
        {"ema_decay": 0.99},
        {"distill_model": "ema"},
        {"eval_only": True},
    ],
)
def test_resume_rejects_uncheckpointed_state(override):
    assert accumulation_and_resume_errors(_args(resume=Path("x"), **override))


def test_resume_state_roundtrip_and_fail_closed(tmp_path):
    identity = train_state_identity(_args(), num_processes=4)
    ckpt = tmp_path / "epoch_41"
    ckpt.mkdir()

    with pytest.raises(FileNotFoundError):
        read_resume_state(ckpt, identity)  # pre-resume checkpoints have no state file

    write_train_state(ckpt, identity, epoch=41, global_step=1234, wandb_run_id="abc")
    state = read_resume_state(ckpt, identity)
    assert (state["epoch"], state["global_step"], state["wandb_run_id"]) == (41, 1234, "abc")
    assert json.loads((ckpt / TRAIN_STATE_FILE).read_text())["training_score"] == "relaxed_v1"

    with pytest.raises(ValueError, match="grad_accum_steps"):
        read_resume_state(ckpt, train_state_identity(_args(grad_accum_steps=8), num_processes=4))
    with pytest.raises(ValueError, match="num_processes"):
        read_resume_state(ckpt, train_state_identity(_args(), num_processes=2))

    final = tmp_path / "epoch_299"
    final.mkdir()
    write_train_state(final, identity, epoch=299, global_step=9, wandb_run_id=None)
    with pytest.raises(ValueError, match="no epochs are left"):
        read_resume_state(final, identity)


def test_accelerate_checkpoint_restores_optimizer_and_scheduler(tmp_path):
    from accelerate import Accelerator

    def build():
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        acc = Accelerator(cpu=True)
        model, opt = acc.prepare(model, opt)
        acc.register_for_checkpointing(sched)
        return acc, model, opt, sched

    acc, model, opt, sched = build()
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(5, 4)).sum().backward()
        opt.step()
        sched.step()
    acc.save_state(str(tmp_path / "epoch_02"))
    expected_lr = sched.get_last_lr()[0]
    expected_weight = next(model.parameters()).detach().clone()

    acc2, model2, opt2, sched2 = build()
    acc2.load_state(str(tmp_path / "epoch_02"))
    assert sched2.last_epoch == 3
    assert sched2.get_last_lr()[0] == pytest.approx(expected_lr)
    assert torch.equal(next(model2.parameters()).detach(), expected_weight)
    assert opt2.state_dict()["state"], "optimizer moments were not restored"


# ── pseudo-accuracy batch routing (2026-09-27 regression) ─────────────────────
def test_pseudo_eval_routes_cached_batches_to_the_vectorized_path():
    from torch.utils.data import default_collate

    from wildmatch.matcher_finetune.loading import collate_pseudo_accuracy_images
    from wildmatch.matcher_finetune.train_common import _is_live_pseudo_batch

    cached_sample = ({"keypoints": torch.zeros(4, 2)}, {"keypoints": torch.zeros(6, 4, 2)}, 0)
    live_sample = (torch.zeros(3, 8, 8), [torch.zeros(3, 8, 10), torch.zeros(3, 12, 8)], 0)

    cached_batch = default_collate([cached_sample, cached_sample])
    assert isinstance(cached_batch, list)  # the reason isinstance(batch, list) was not enough
    assert not _is_live_pseudo_batch(cached_batch)
    assert _is_live_pseudo_batch(collate_pseudo_accuracy_images([live_sample, live_sample]))
    assert not _is_live_pseudo_batch([])
    # pin_memory=True (training) hands each live sample over as a list, not a tuple.
    pinned_live = [list(live_sample)]
    assert _is_live_pseudo_batch(collate_pseudo_accuracy_images(pinned_live))
    assert not _is_live_pseudo_batch([list(cached_sample)])  # starts with a feature dict


# ── checkpoint retention (--keep_every) ───────────────────────────────────────
def test_keep_every_retains_milestones_final_and_newest(tmp_path):
    from wildmatch.matcher_finetune.train_by_lg_matches import is_retained_epoch, prune_previous_checkpoint

    identity = train_state_identity(_args(epochs=12), num_processes=4)
    for epoch in range(12):
        ckpt = tmp_path / f"epoch_{epoch:02d}"
        ckpt.mkdir()
        write_train_state(ckpt, identity, epoch=epoch, global_step=epoch, wandb_run_id=None)
        prune_previous_checkpoint(tmp_path, epoch, keep_every=5, epochs=12, identity=identity)
        newest = sorted(p.name for p in tmp_path.iterdir())[-1]
        assert newest == f"epoch_{epoch:02d}"  # the resumable newest epoch always survives
    assert sorted(p.name for p in tmp_path.iterdir()) == ["epoch_00", "epoch_05", "epoch_10", "epoch_11"]
    assert all(is_retained_epoch(e, 0, 12) for e in range(12))  # 0 keeps everything


def test_keep_every_never_deletes_foreign_or_mismatched_directories(tmp_path):
    from wildmatch.matcher_finetune.train_by_lg_matches import prune_previous_checkpoint

    identity = train_state_identity(_args(epochs=300), num_processes=4)
    (tmp_path / "epoch_06").mkdir()  # no train_state.json, e.g. an archived pre-2026-09-29 checkpoint
    assert prune_previous_checkpoint(tmp_path, 7, 50, 300, identity) is None
    other = tmp_path / "epoch_07"
    other.mkdir()
    write_train_state(other, train_state_identity(_args(epochs=300, lr=1e-4), num_processes=4), 7, 7, None)
    assert prune_previous_checkpoint(tmp_path, 8, 50, 300, identity) is None
    assert (tmp_path / "epoch_06").is_dir() and other.is_dir()
