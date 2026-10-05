"""
Precompute RDD keypoints/descriptors for every frame, once, into a
`KeypointCache` a frozen-RDD run can read instead of re-running detection.

See contrastive_finetuning/keypoint_cache.py for what is stored and why it is
sound only while RDD is frozen.

Typical use — cache the whole dataset (train + test), which covers any index
and any `--random_negative_prob`::

    python -m contrastive_finetuning.build_keypoint_cache \
        --data_root /shared/.../lynx-ds-Jul-20 \
        --cache_root /shared/.../lynx-ds-Jul-20-extracted-rdd \
        --rdd_weights rdd/weights/RDD-v2.pth --resize 512 --top_k 512

Sizing (measured at --resize 512 / --top_k 512, mean 404 keypoints/frame):
~204 KiB per frame, ~90 frames/s on an idle H100 — so the 148k-frame dataset
is ~31 GB and ~28 min of GPU time. `--num_shards`/`--shard` splits that across
several GPUs (one process each); `--resume` skips frames already written, so an
interrupted build just continues.

`--verify` re-checks that every frame the enumeration covers actually has a
file, without running RDD — worth doing after a sharded build, since a shard
that died leaves a cache that looks complete until a training step happens to
draw the missing frame.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from contrastive_finetuning.keypoint_cache import (
    CacheSpec, KeypointCache, MANIFEST_NAME, weights_fingerprint,
)
from contrastive_finetuning.models import build_rdd
from contrastive_finetuning.train_common import _unwrap, extract_train, resize_long_side


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, required=True, help="Dataset root (holds train/, test/)")
    p.add_argument("--cache_root", type=Path, required=True, help="Where to write the cache")
    p.add_argument("--rdd_weights", type=Path, default=Path("rdd/weights/RDD-v2.pth"))
    p.add_argument(
        "--splits", nargs="*", default=["train", "test"],
        help="Split directories under --data_root to cache in full (default: both). "
             "The training split alone is what --random_negative_prob draws from; "
             "the test split adds the val indices' query frames, and caching it "
             "whole costs ~6.6 GB more but makes the cache index-agnostic.",
    )
    p.add_argument(
        "--index", type=Path, nargs="*", default=[],
        help="Optional index JSONs; every frame they reference is added to the "
             "enumeration even if it lies outside --splits.",
    )
    p.add_argument("--resize", type=int, default=512, help="Must match the training run's --resize")
    p.add_argument("--top_k", type=int, default=512, help="Must match the training run's --top_k")
    p.add_argument(
        "--batch_size", type=int, default=32,
        help="Frames per RDD forward. Deformable attention scales steeply with "
             "this; 32 saturates an H100 at --resize 512.",
    )
    p.add_argument("--num_workers", type=int, default=16, help="JPEG decode workers")
    p.add_argument("--num_shards", type=int, default=1, help="Split the frame list across N processes")
    p.add_argument("--shard", type=int, default=0, help="Which shard this process builds (0-based)")
    p.add_argument("--resume", action="store_true", help="Skip frames already present in the cache")
    p.add_argument("--verify", action="store_true", help="Only check coverage; run no RDD")
    p.add_argument("--limit", type=int, default=0, help="Cache at most this many frames (smoke tests)")
    args = p.parse_args()
    if not (0 <= args.shard < args.num_shards):
        p.error("--shard must be in [0, --num_shards)")
    return args


def enumerate_frames(args: argparse.Namespace) -> list[str]:
    """Dataset-relative paths of every frame the cache should cover, sorted."""
    frames: set[str] = set()
    for split in args.splits:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split {split!r} not found under {args.data_root}")
        frames.update(str(p.relative_to(args.data_root)) for p in split_dir.rglob("*.jpg"))
    for index_path in args.index:
        with open(index_path) as f:
            for entry in json.load(f):
                frames.add(entry["query_frame"])
                frames.update(entry["positives"])
                frames.update(entry["negatives"])
    return sorted(frames)


class FrameDataset(Dataset):
    """Loads and resizes one frame per item; RDD then runs on the batch.

    Resizing happens here (in the worker) rather than on the GPU so the decode
    and the resize both parallelize over --num_workers.
    """

    def __init__(self, frames: list[str], root: Path, resize: int) -> None:
        self.frames, self.root, self.resize = frames, root, resize
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        rel = self.frames[i]
        img = self.to_tensor(Image.open(self.root / rel).convert("RGB"))
        return resize_long_side(img.unsqueeze(0), self.resize)[0], rel


def collate_by_shape(items):
    """Group a batch by resized shape, so mixed-resolution datasets still stack.

    Every frame in this dataset is 960x384 (so there is one group), but a
    dataset with mixed source resolutions would otherwise fail in default_collate
    — and silently padding them together would corrupt the keypoint coordinates.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for img, rel in items:
        groups[tuple(img.shape)].append((img, rel))
    return [(torch.stack([i for i, _ in g]), [r for _, r in g]) for g in groups.values()]


def main() -> None:
    args = parse_args()
    frames = enumerate_frames(args)
    print(f"enumerated {len(frames):,} frames from {args.data_root} "
          f"(splits={args.splits}, {len(args.index)} extra index files)")

    if args.verify:
        cache = KeypointCache(args.cache_root)
        missing = [f for f in tqdm(frames, desc="verify") if not cache.has(f)]
        print(f"cache {args.cache_root}: {len(frames) - len(missing):,}/{len(frames):,} present, "
              f"{len(missing):,} missing")
        for f in missing[:20]:
            print(f"  missing: {f}")
        raise SystemExit(1 if missing else 0)

    total_planned = len(frames)
    shard = frames[args.shard::args.num_shards]
    if args.limit:
        shard = shard[:args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(shard):,} frames")

    if args.resume:
        before = len(shard)
        shard = [f for f in tqdm(shard, desc="resume scan")
                 if not (args.cache_root / Path(f).with_suffix(".npz")).exists()]
        print(f"resume: {before - len(shard):,} already cached, {len(shard):,} to go")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdd = build_rdd(args.rdd_weights, device, args.top_k)
    raw = _unwrap(rdd)

    # One frame up front, purely to read the descriptor width off a real
    # forward rather than hardcoding it — the spec has to be final before the
    # compatibility check below, not patched up after the build. Falls back to
    # the full list so that a --resume run with nothing left to do still reaches
    # the manifest write below: a build interrupted before that point leaves a
    # complete frame tree with no manifest, and re-running --resume is the
    # obvious way to finish it.
    probe_img = FrameDataset((shard or frames)[:1], args.data_root, args.resize)[0][0]
    with torch.no_grad():
        probe = extract_train(rdd, probe_img.unsqueeze(0).to(device))
    spec = CacheSpec(
        rdd_weights_sha256=weights_fingerprint(args.rdd_weights),
        resize=args.resize,
        top_k=args.top_k,
        detection_threshold=float(raw.detection_threshold),
        descriptor_dim=int(probe[0]["descriptors"].shape[1]),
    )

    # An existing manifest must agree with what this process is about to write,
    # or the cache would end up holding two incompatible feature sets.
    manifest_path = args.cache_root / MANIFEST_NAME
    if manifest_path.exists():
        KeypointCache(args.cache_root, spec=spec)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    cache = KeypointCache.for_writing(args.cache_root, spec)

    loader = DataLoader(
        FrameDataset(shard, args.data_root, args.resize),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_by_shape, pin_memory=True,
    )

    n_done, n_kpts = 0, 0
    t0 = time.perf_counter()
    pbar = tqdm(loader, desc=f"extract[{args.shard}]")
    for groups in pbar:
        for images, rels in groups:
            images = images.to(device, non_blocking=True)
            H_r, W_r = images.shape[-2:]
            with torch.no_grad():
                feats = extract_train(rdd, images)
            for rel, f in zip(rels, feats):
                cache.save(rel, f["keypoints"], f["descriptors"], (H_r, W_r))
                n_kpts += int(f["keypoints"].shape[0])
            n_done += len(rels)
        if n_done:
            pbar.set_postfix(fps=f"{n_done / (time.perf_counter() - t0):.1f}",
                             kpts=f"{n_kpts / max(n_done, 1):.0f}")

    elapsed = time.perf_counter() - t0
    manifest_extra = {
        "data_root": str(args.data_root),
        "splits": args.splits,
        "n_frames_enumerated": total_planned,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if n_done:
        manifest_extra["mean_keypoints_last_build"] = n_kpts / n_done
    KeypointCache.write_manifest(args.cache_root, spec, manifest_extra)
    if n_done:
        print(f"wrote {n_done:,} frames in {elapsed/60:.1f} min "
              f"({n_done/elapsed:.1f} frames/s, mean {n_kpts/n_done:.0f} keypoints)")
    else:
        print("nothing to extract — every frame in this shard was already cached")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
