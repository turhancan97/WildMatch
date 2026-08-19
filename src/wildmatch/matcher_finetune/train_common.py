from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from torch.utils.data import Subset

from rdd.RDD.utils import to_pixel_coords
from contrastive_finetuning.keypoint_cache import is_cached_batch, unpad_cached_features
from contrastive_finetuning.loading import PseudoAccuracyDataset, get_loader
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
    p.add_argument(
        "--split_protocol", choices=["legacy", "strict"], default=None,
        help="Dataset split protocol recorded in run metadata; used by CzechLynx SLURM entry points.",
    )
    p.add_argument("--epochs",        type=int,  default=10)
    p.add_argument("--batch_size",    type=int,  default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--resize",        type=int,  default=512)
    p.add_argument("--top_k",         type=int,  default=512)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--seed",          type=int,  default=0)
    p.add_argument("--num_workers",   type=int,  default=4)
    p.add_argument("--eval_every_epochs", type=int, default=10)
    p.add_argument(
        "--eval_batch_size", type=int, default=4,
        help="Queries per DataLoader batch in eval_pseudo_accuracy (controls CPU "
             "decode/prefetch parallelism). Candidate images (eval_batch_size * "
             "(n_pos + n_neg) per DataLoader batch) are chunked to --batch_size "
             "before RDD's deformable attention, since that scales steeply with "
             "images-per-call and OOMs on larger top_k/top_m indices otherwise",
    )
    p.add_argument(
        "--keypoint_cache", type=Path, default=None,
        help="Directory holding a prebuilt RDD keypoint cache (see "
             "`python -m contrastive_finetuning.build_keypoint_cache`). When set, "
             "every RDD detection — training steps, both eval paths, and the "
             "pre-training measurement passes — is replaced by a lookup of "
             "precomputed keypoints/descriptors, and no image is decoded at all. "
             "Only valid with a FROZEN RDD (--trained_model lg) and a fixed input: "
             "--augment and --multi_scale_* are rejected, since both change the "
             "image RDD would have seen. The cache records the RDD weights hash, "
             "--resize, --top_k and the detection threshold it was built with, and "
             "refuses to open against a run that disagrees. Measured ~2.6x faster "
             "training steps and ~5x faster pseudo-accuracy eval.",
    )
    p.add_argument(
        "--wandb_tags", type=str, default="",
        help="Comma-separated wandb tags for this run",
    )
    p.add_argument(
        "--trained_model", type=str, default="lg", choices=["lg", "rdd", "lg+rdd"],
        help="Which model(s) are unfrozen and receive gradient: 'lg' (default) "
             "freezes RDD and trains only LightGlue; 'rdd' freezes LightGlue and "
             "trains only RDD; 'lg+rdd' trains both.",
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


def resolve_trained_models(trained_model: str) -> tuple[bool, bool]:
    """Splits `--trained_model` ('lg' | 'rdd' | 'lg+rdd') into (train_rdd, train_lg)."""
    models = trained_model.split("+")
    return "rdd" in models, "lg" in models


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


def _image_size_rows(image_h, image_w, n: int) -> list[tuple[int, int]]:
    """Normalize scalar or per-frame H/W values to ``(H, W)`` rows."""
    if isinstance(image_h, (list, tuple)):
        heights = [int(v) for v in image_h]
        widths = [int(v) for v in image_w]
        if len(heights) != n or len(widths) != n:
            raise ValueError(
                f"per-frame image sizes have lengths {len(heights)} and {len(widths)} "
                f"for a feature batch of {n}"
            )
        return list(zip(heights, widths))
    return [(int(image_h), int(image_w))] * n


def _repeat_image_sizes(image_h, image_w, repeats: int):
    """Repeat per-frame sizes in the same row-major order as repeated features."""
    if isinstance(image_h, list):
        return (
            [h for h in image_h for _ in range(repeats)],
            [w for w in image_w for _ in range(repeats)],
        )
    return image_h, image_w


def batch_features(feats: list[dict], image_h, image_w) -> dict:
    """Pack variable-length features with scalar or per-frame image sizes."""
    ks = [f["keypoints"]   for f in feats]
    ds = [f["descriptors"] for f in feats]
    device = ks[0].device

    ks_pad, masks = align_tensors_to_max_length(ks)
    ds_pad, _     = align_tensors_to_max_length(ds)
    sizes = torch.tensor(
        [[w, h] for h, w in _image_size_rows(image_h, image_w, len(feats))],
        device=device,
    ).contiguous()

    return {
        "keypoints":   ks_pad,
        "descriptors": ds_pad,
        "image_size":  sizes,
        "masks":       masks.unsqueeze(1),
    }


def _select_batch_rows(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict:
    """Select feature-batch rows while keeping the tensors on their device."""
    return {key: value.index_select(0, indices) for key, value in data.items()}


def run_lg_partitioned(lg: torch.nn.Module, data0: dict, data1: dict) -> dict:
    """Run LightGlue separately for each pair of image dimensions.

    Cached RDD coordinates are expressed in each frame's resized image space.
    This partitions mixed cached pair batches by
    ``(image0_h, image0_w, image1_h, image1_w)`` and merges dense outputs back
    into the original order. Uniform image batches take one unchanged forward.
    """
    size0 = data0["image_size"]
    size1 = data1["image_size"]
    if size0.shape[0] != size1.shape[0]:
        raise ValueError(
            f"LightGlue pair batches have different lengths: {size0.shape[0]} and {size1.shape[0]}"
        )

    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for row, (s0, s1) in enumerate(zip(size0.tolist(), size1.tolist())):
        groups[(*map(int, s0), *map(int, s1))].append(row)
    if len(groups) == 1:
        return lg({"image0": data0, "image1": data1})

    grouped_outputs = []
    for rows in groups.values():
        indices = torch.tensor(rows, dtype=torch.long, device=size0.device)
        grouped_outputs.append((indices, lg({
            "image0": _select_batch_rows(data0, indices),
            "image1": _select_batch_rows(data1, indices),
        })))

    batch_size = size0.shape[0]
    merged: dict = {}
    for key in grouped_outputs[0][1]:
        values = [output[key] for _, output in grouped_outputs]
        if torch.is_tensor(values[0]):
            target_shape = [batch_size] + [
                max(value.shape[dim] for value in values)
                for dim in range(1, values[0].dim())
            ]
            fill = -1 if key.startswith("matches") else 0
            result = values[0].new_full(target_shape, fill)
            for indices, output in grouped_outputs:
                value = output[key]
                padding = []
                for dim in reversed(range(1, value.dim())):
                    padding.extend((0, target_shape[dim] - value.shape[dim]))
                if padding:
                    value = F.pad(value, padding, value=fill)
                result = result.index_copy(0, indices, value)
            merged[key] = result
        elif isinstance(values[0], list):
            result = [None] * batch_size
            for indices, output in grouped_outputs:
                for row, value in zip(indices.tolist(), output[key]):
                    result[row] = value
            merged[key] = result
        else:
            merged[key] = max(values)
    return merged


# ── training-time feature extraction ─────────────────────────────────────────
def extract_train(rdd: torch.nn.Module, images: torch.Tensor) -> list[dict]:
    """
    One RDD forward pass. Whether RDD is frozen is decided by `--trained_model`
    in train_by_lg_matches.py; this stays a plain forward either way — the
    caller wraps it in torch.no_grad() when it wants to skip building the graph.

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
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Run LightGlue WITHOUT a no_grad wrapper, so LG's own parameters receive
    gradient from any loss computed on the returned `scores` /
    `matching_scores0`. Returns the full prediction dicts (not just match
    indices) — used by train_by_lg_matches.py.

    Also returns the batch_features dicts (data_a/data_p/data_n): callers
    that want keypoint-coverage-normalized scores (see _lg_scores) need
    their `masks`.

    Note LightGlueForTraining detaches its descriptor *inputs* by default (see
    `detach_descriptors` in rdd_patch/lightglue_masked_training.py), so gradient
    normally only reaches LG's own weights. train_by_lg_matches.py builds LG
    with `detach_descriptors=False` when `--trained_model` includes 'rdd', so
    gradient can also flow back into the network that produced feats_*.
    """
    data_a = batch_features(feats_a, image_h, image_w)
    data_p = batch_features(feats_p, image_h, image_w)
    data_n = batch_features(feats_n, image_h, image_w)
    pred_pos = run_lg_partitioned(lg, data_a, data_p)
    pred_neg = run_lg_partitioned(lg, data_a, data_n)
    return pred_pos, pred_neg, data_a, data_p, data_n


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
        feats_a, H_a, W_a = features_from_batch(anchors,   _unwrap(rdd), args.resize, device)
        feats_p, H_p, W_p = features_from_batch(positives, _unwrap(rdd), args.resize, device)
        feats_n, H_n, W_n = features_from_batch(negatives, _unwrap(rdd), args.resize, device)

        data_a = batch_features(feats_a, H_a, W_a)
        data_p = batch_features(feats_p, H_p, W_p)
        data_n = batch_features(feats_n, H_n, W_n)

        pred_pos = run_lg_partitioned(lg, data_a, data_p)
        pred_neg = run_lg_partitioned(lg, data_a, data_n)

        # valid0 is the dense form of the ragged `matches` list — same count,
        # no per-batch-item Python loop.
        total_pos += pred_pos["valid0"].sum()
        total_neg += pred_neg["valid0"].sum()
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
def _video_id(rel_path: str) -> str:
    """Index paths look like `{split}/{lynx_id}/{location}/{video_id}/{frame}.jpg`."""
    return str(Path(rel_path).parent)


def _lynx_id(rel_path: str) -> str:
    return Path(rel_path).parts[1]


def features_from_batch(
    batch,
    rdd: torch.nn.Module,
    resize: int,
    device: torch.device,
    chunk_size: int | None = None,
) -> tuple[list[dict], int | list[int], int | list[int]]:
    """Features for one DataLoader element, from the cache or from RDD.

    Returns `(feats, H, W)` where cached inputs provide per-frame H/W lists
    and live image inputs provide scalar dimensions.

    `batch` is either a stacked image tensor (the normal path, resized here and
    pushed through RDD) or the collated output of `KeypointCache.load_padded`
    (`--keypoint_cache`, where RDD never runs and no image was ever decoded).
    Every extraction site goes through this, and cached leading batch dims are
    flattened in the same row-major order as the live image path.

    Pass `chunk_size` to bound images-per-RDD-call (see `_extract_chunked`); it
    is irrelevant to the cached branch, which has no such forward.
    """
    if is_cached_batch(batch):
        return unpad_cached_features(batch, device)
    images = resize_long_side(batch, resize).to(device)
    h, w = images.shape[-2:]
    if chunk_size is None:
        return extract_train(rdd, images), h, w
    return _extract_chunked(rdd, images, chunk_size), h, w


def _extract_chunked(rdd: torch.nn.Module, images: torch.Tensor, chunk_size: int) -> list[dict]:
    """extract_train, chunked along the batch dim.

    RDD's deformable attention scales steeply with images-per-forward-call, so
    peak GPU memory needs to be bounded independent of how many candidate
    images a DataLoader batch happens to bring along (which varies with the
    index's top_k/top_m).
    """
    feats: list[dict] = []
    for i in range(0, images.shape[0], chunk_size):
        feats.extend(extract_train(rdd, images[i:i + chunk_size]))
    return feats


def _lg_scores(pred: dict, q_data: dict, g_data: dict) -> torch.Tensor:
    """Per-pair score = sum(match confidence) / min(valid keypoints in query, in candidate).

    Normalizing by keypoint coverage instead of averaging confidence over
    however many matches were found keeps a couple of lucky high-confidence
    matches from outscoring a pair that's genuinely well-matched throughout.

    Reads the dense `matching_scores0` / `valid0` that LightGlueForTraining
    exposes (see rdd_patch/lightglue_masked_training.py), so `pred` must come
    from that class — everything is built through models.build_masked_lg, which
    does. `matching_scores0 * valid0` selects exactly the entries LightGlue puts
    in its ragged `scores` list, so the value is unchanged, but it stays
    attached to the graph when a pair matched nothing: masking with all-False
    still propagates zeros back to every parameter, whereas the ragged list's
    empty tensor used to be replaced by a detached `torch.zeros(())` and cut the
    loss off from the model entirely.

    Stays lazy/on-device throughout (no `.item()`) — this is also called from
    the training loss every step, where a GPU sync per batch element would
    actually cost something, unlike in eval.
    """
    mscores = pred["matching_scores0"]
    sums = (mscores * pred["valid0"].to(mscores.dtype)).sum(dim=1)
    n_q = q_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    n_g = g_data["masks"].squeeze(1).squeeze(-1).sum(dim=1).clamp(min=1)
    return sums / torch.minimum(n_q, n_g)


def _pseudo_batch_dims(cand_batch) -> tuple[int, int]:
    """(queries, candidates per query) for a PseudoAccuracyDataset batch."""
    if is_cached_batch(cand_batch):
        return tuple(cand_batch["n_keypoints"].shape[:2])
    return int(cand_batch.shape[0]), int(cand_batch.shape[1])


def _flatten_candidates(cand_batch):
    """Fold the per-query candidate dim into the batch dim, for either payload.

    Images need an explicit view; cached features are flattened downstream by
    `unpad_cached_features`, which handles any number of leading dims, so they
    pass through untouched.
    """
    if is_cached_batch(cand_batch):
        return cand_batch
    B, n_cand, C, H, W = cand_batch.shape
    return cand_batch.view(B * n_cand, C, H, W)


def group_pseudo_accuracy_entries(entries: list[dict]) -> list[list[dict]]:
    """Group entries by positive/negative candidate counts deterministically."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for entry in entries:
        shape = (len(entry["positives"]), len(entry["negatives"]))
        buckets[shape].append(entry)
    return [buckets[shape] for shape in sorted(buckets)]


def build_pseudo_accuracy_loader(
    accelerator: Accelerator,
    dataset_subset,
    args: argparse.Namespace,
):
    """Build prepared pseudo-evaluation loaders grouped by candidate shape.

    The original Lynx indices have a fixed number of positives and negatives,
    but CzechLynx collections can have different positive counts. Entries are
    bucketed by ``(n_pos, n_neg)`` so each DataLoader remains stackable without
    padding or duplicating candidates. The return value is a list of
    ``(prepared_loader, dataset)`` pairs.
    """
    if isinstance(dataset_subset, Subset):
        base_ds = dataset_subset.dataset
        entries = [base_ds._entries[i] for i in dataset_subset.indices]
    else:
        base_ds = dataset_subset
        entries = base_ds._entries

    prepared = []
    for bucket in group_pseudo_accuracy_entries(entries):
        ds = PseudoAccuracyDataset(
            bucket,
            root=base_ds.root,
            transform=base_ds.transform,
            query_transform=base_ds.query_transform,
            loader=base_ds._loader,
            feature_cache=base_ds.feature_cache,
        )
        loader = get_loader(
            ds, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        )
        prepared.append((accelerator.prepare(loader), ds))

    if accelerator.is_main_process and len(prepared) > 1:
        accelerator.print(
            "pseudo-eval candidate buckets: "
            + ", ".join(
                f"{ds.n_pos}+{ds.n_neg}={len(ds)}" for _, ds in prepared
            )
        )
    return prepared


@torch.no_grad()
def eval_pseudo_accuracy(
    accelerator: Accelerator,
    rdd: torch.nn.Module,
    lg: torch.nn.Module,
    loader,
    args: argparse.Namespace,
    prefix: str,
    verbose: bool = False,
) -> dict:
    """Evaluate every query against its complete variable-size candidate pool.

    ``loader`` is the list returned by build_pseudo_accuracy_loader. Each
    bucket has a fixed candidate shape internally, while metrics are combined
    over all queries exactly once.
    """
    device = accelerator.device
    _unwrap(rdd).eval()

    entries = [entry for _, ds in loader for entry in ds.entries]
    accuracies: list[float] = []
    best_pos_scores: list[float] = []
    best_neg_scores: list[float] = []
    videos: dict[str, dict] = {}

    for group_loader, ds in loader:
        for query_batch, cand_batch, idx_batch in tqdm(
            group_loader, desc=f"{prefix}[{ds.n_pos}+{ds.n_neg}]", leave=False,
            disable=not accelerator.is_main_process
        ):
            if not is_cached_batch(query_batch):
                query_batch = query_batch.to(device)
                cand_batch = cand_batch.to(device)
            B, n_cand = _pseudo_batch_dims(cand_batch)
            feats_q, H_q, W_q = features_from_batch(
                query_batch, _unwrap(rdd), args.resize, device, chunk_size=args.batch_size)
            feats_c, H_c, W_c = features_from_batch(
                _flatten_candidates(cand_batch), _unwrap(rdd), args.resize, device,
                chunk_size=args.batch_size)

            feats_q_rep = [f for f in feats_q for _ in range(n_cand)]
            H_q_rep, W_q_rep = _repeat_image_sizes(H_q, W_q, n_cand)
            data_q = batch_features(feats_q_rep, H_q_rep, W_q_rep)
            data_c = batch_features(feats_c, H_c, W_c)
            pred = run_lg_partitioned(lg, data_q, data_c)
            scores = _lg_scores(pred, data_q, data_c).view(B, n_cand)

            score_pos, _ = scores[:, :ds.n_pos].max(dim=1)
            score_neg, _ = scores[:, ds.n_pos:].max(dim=1)
            score_best, idx_best = scores.max(dim=1)
            score_pos, score_neg, score_best, idx_best, idx_batch = accelerator.gather_for_metrics(
                (score_pos, score_neg, score_best, idx_best, idx_batch.to(device))
            )

            for sp, sn, sb, ib, idx in zip(
                score_pos.tolist(), score_neg.tolist(), score_best.tolist(),
                idx_best.tolist(), idx_batch.tolist(),
            ):
                accuracies.append(1.0 if sp > sn else 0.5 if sp == sn else 0.0)
                best_pos_scores.append(sp)
                best_neg_scores.append(sn)

                entry = ds.entries[idx]
                video_id = _video_id(entry["query_frame"])
                true_lynx = _lynx_id(entry["query_frame"])
                cand_paths = entry["positives"] + entry["negatives"]
                best_lynx = _lynx_id(cand_paths[ib])
                rec = videos.setdefault(video_id, {
                    "true_lynx": true_lynx, "best_score": -float("inf"),
                    "best_lynx": None, "best_query_frame": None,
                    "best_cand_path": None,
                })
                if sb > rec["best_score"]:
                    rec["best_score"] = sb
                    rec["best_lynx"] = best_lynx
                    rec["best_query_frame"] = entry["query_frame"]
                    rec["best_cand_path"] = cand_paths[ib]

    n = len(entries)
    video_correct = [
        1.0 if rec["best_lynx"] == rec["true_lynx"] else 0.0
        for rec in videos.values()
    ]

    if verbose:
        n_wrong = 0
        for video_id, rec in sorted(videos.items()):
            if rec["best_lynx"] == rec["true_lynx"]:
                continue
            n_wrong += 1
            accelerator.print(
                f"[{prefix}] MISMATCH video={video_id} true_lynx={rec['true_lynx']} "
                f"predicted_lynx={rec['best_lynx']} score={rec['best_score']:.4f} "
                f"query_frame={rec['best_query_frame']} matched_candidate={rec['best_cand_path']}"
            )
        accelerator.print(f"[{prefix}] {n_wrong}/{len(videos)} videos misclassified")

    return {
        f"{prefix}/frame_accuracy": sum(accuracies) / max(n, 1),
        f"{prefix}/mean_score_pos": sum(best_pos_scores) / max(n, 1),
        f"{prefix}/mean_score_neg": sum(best_neg_scores) / max(n, 1),
        f"{prefix}/video_accuracy": sum(video_correct) / max(len(video_correct), 1),
    }
