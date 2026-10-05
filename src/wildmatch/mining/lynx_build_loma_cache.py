"""Build a resumable, benchmark-compatible LoMa feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from wildmatch.mining.loma_backend import (
    LOMA_PATCH_SIZE,
    build_loma,
    extract_batch,
)
from wildmatch.mining.lynx_dataset import list_sequences, sample_frames
from wildmatch.mining.wildlife_dataset import list_collections

MANIFEST_NAME = "manifest.json"


def weights_fingerprint(path: Path) -> str:
    if path.is_dir():
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            base = metadata.get("base_weights")
            if base:
                base_path = Path(base).expanduser()
                if not base_path.is_absolute():
                    base_path = path / base_path
                if base_path.is_file():
                    path = base_path
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=False)
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--frames_per_seq", type=int, default=20,
                        help="Frames per sequence in the legacy sampled mode.")
    parser.add_argument("--all_frames", action="store_true",
                        help="Cache every frame under the selected splits.")
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--index", type=Path, nargs="*", default=[],
                        help="Index files whose referenced frames must be included.")
    parser.add_argument("--resize_max", type=int, default=512)
    parser.add_argument("--num_keypoints", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="Check cache coverage without loading LoMa.")
    return parser.parse_args()


def frame_groups(dataset_root: Path, split: str) -> list[list[Path]]:
    """Return frame groups for both legacy Lynx and generic wildlife layouts.

    The original Lynx layout has ``split/identity/site/sequence/frame.jpg``.
    WildlifeReID uses ``split/identity/collection/frame.jpg``.  The latter is
    intentionally handled here instead of changing the original benchmark
    sequence loader used by existing commands.
    """
    sequences = list_sequences(dataset_root, split)
    if sequences:
        return [sequence.frame_paths for sequence in sequences]
    return [collection.frame_paths for collection in list_collections(dataset_root, split)]


def enumerate_frames(args) -> list[str]:
    frames: set[str] = set()
    for split in args.splits:
        for frame_paths in frame_groups(args.dataset_root, split):
            selected = frame_paths if args.all_frames else sample_frames(
                frame_paths, args.frames_per_seq
            )
            frames.update(str(path.relative_to(args.dataset_root)) for path in selected)
    for index_path in args.index:
        with index_path.open() as handle:
            for entry in json.load(handle):
                frames.add(entry["query_frame"])
                frames.update(entry["positives"])
                frames.update(entry["negatives"])
    return sorted(frames)


class FrameDataset(Dataset):
    def __init__(self, frames: list[str], root: Path, resize_max: int) -> None:
        self.frames = frames
        self.root = root
        self.resize_max = resize_max
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        from wildmatch.mining.loma_backend import resize_image

        rel = self.frames[index]
        image = self.to_tensor(Image.open(self.root / rel).convert("RGB"))
        image = resize_image(image.unsqueeze(0), self.resize_max)[0]
        return image, rel


def collate_by_shape(items):
    groups: dict[tuple[int, ...], list[tuple[torch.Tensor, str]]] = defaultdict(list)
    for image, rel in items:
        groups[tuple(image.shape)].append((image, rel))
    return [
        (torch.stack([image for image, _ in group]), [rel for _, rel in group])
        for group in groups.values()
    ]


def verify_cache(args, frames: list[str]) -> None:
    manifest_path = args.cache_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; cache has no metadata manifest")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "format": "lynx-loma-cache-v1",
        "backend": "loma",
        "variant": args.variant,
        "resize": args.resize_max,
        "num_keypoints": args.num_keypoints,
        "patch_size": LOMA_PATCH_SIZE,
    }
    mismatches = {
        key: (manifest.get(key), wanted)
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    if args.weights is not None and manifest.get("weights_sha256"):
        wanted_hash = weights_fingerprint(args.weights)
        if manifest["weights_sha256"] != wanted_hash:
            mismatches["weights_sha256"] = (manifest["weights_sha256"], wanted_hash)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (wanted {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        print(f"incompatible LoMa cache metadata: {details}")
        raise SystemExit(2)
    missing = [
        rel for rel in tqdm(frames, desc="verify")
        if not (args.cache_dir / Path(rel).with_suffix(".npz")).exists()
    ]
    print(f"cache {args.cache_dir}: {len(frames) - len(missing):,}/{len(frames):,} present")
    if missing:
        print("missing examples:")
        for rel in missing[:20]:
            print(f"  {rel}")
        raise SystemExit(1)


def write_manifest(args, total_frames: int, weights_hash: str | None) -> None:
    manifest = {
        "format": "lynx-loma-cache-v1",
        "backend": "loma",
        "variant": args.variant,
        "weights_sha256": weights_hash,
        "resize": args.resize_max,
        "num_keypoints": args.num_keypoints,
        "patch_size": LOMA_PATCH_SIZE,
        "splits": args.splits,
        "all_frames": args.all_frames,
        "frames_per_seq": args.frames_per_seq,
        "n_frames": total_frames,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def main():
    args = parse_args()
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise ValueError("--shard must be in [0, --num_shards)")
    frames = enumerate_frames(args)
    print(
        f"enumerated {len(frames):,} frames from {args.dataset_root} "
        f"(splits={args.splits}, all_frames={args.all_frames}, indexes={len(args.index)})"
    )

    if args.verify:
        verify_cache(args, frames)
        return
    if args.weights is None:
        raise ValueError("--weights is required when building a cache")

    shard_frames = frames[args.shard::args.num_shards]
    if args.resume and not args.overwrite:
        shard_frames = [
            rel for rel in shard_frames
            if not (args.cache_dir / Path(rel).with_suffix(".npz")).exists()
        ]
    print(f"shard {args.shard}/{args.num_shards}: {len(shard_frames):,} frames to extract")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    model = build_loma(device, args.weights, args.variant)
    loader = DataLoader(
        FrameDataset(shard_frames, args.dataset_root, args.resize_max),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_by_shape,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    for groups in tqdm(loader, desc=f"extract[{args.shard}]", leave=True):
        for images, rels in groups:
            features = extract_batch(model, images.to(device, non_blocking=True), args.num_keypoints)
            for rel, feature in zip(rels, features):
                output = args.cache_dir / Path(rel).with_suffix(".npz")
                if output.exists() and not args.overwrite:
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output, **feature)
                written += 1
    print(f"wrote {written:,} frames in {(time.perf_counter() - started) / 60:.1f} minutes")

    # All shards write the same compatibility metadata. The SLURM wrapper
    # verifies coverage after all shard/build processes have completed.
    write_manifest(args, len(frames), weights_fingerprint(args.weights))


if __name__ == "__main__":
    main()
