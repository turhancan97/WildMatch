"""Build a resumable fixed-DaD-keypoint cache for LoMa descriptor training."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from contrastive_finetuning.loma_backend import build_loma
from contrastive_finetuning.loma_keypoint_cache import (
    CACHE_FORMAT,
    MANIFEST_NAME,
    LomaKeypointCache,
    module_fingerprint,
)
from contrastive_finetuning.train_loma_matches import load_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--num_keypoints", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def list_frames(data_root: Path, splits: list[str]) -> list[Path]:
    frames: list[Path] = []
    for split in splits:
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"split directory does not exist: {split_root}")
        frames.extend(
            path for path in split_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(frames)


def target_hw(path: Path, resize: int) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if resize <= 0:
        return height, width
    scale = resize / max(height, width)
    patch = 14
    new_height = max(patch, int(height * scale) // patch * patch)
    new_width = max(patch, int(width * scale) // patch * patch)
    return new_height, new_width


def write_frame(path: Path, keypoints: np.ndarray, image_size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            keypoints=keypoints.astype(np.float32, copy=False),
            image_size=np.asarray(image_size, dtype=np.int32),
        )
    os.replace(temporary, path)


def _existing_record_is_valid(path: Path, num_keypoints: int) -> bool:
    try:
        with np.load(path) as data:
            return (
                data["keypoints"].shape == (num_keypoints, 2)
                and data["image_size"].shape == (2,)
            )
    except (OSError, KeyError, ValueError):
        return False


def build_cache(args: argparse.Namespace) -> None:
    frames = list_frames(args.data_root, args.splits)
    if not frames:
        raise ValueError(f"no images found below {args.data_root} for {args.splits}")

    device = torch.device(args.device)
    model = build_loma(args.variant, args.weights, device)
    detector_hash = module_fingerprint(model._detector)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / MANIFEST_NAME
    expected = {
        "format": CACHE_FORMAT,
        "backend": "loma",
        "variant": args.variant,
        "resize": args.resize,
        "num_keypoints": args.num_keypoints,
        "patch_size": 14,
        "detector_sha256": detector_hash,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "existing LoMa keypoint cache is incompatible; use a separate cache path: "
                + ", ".join(
                    f"{key}={actual!r} (wanted {wanted!r})"
                    for key, (actual, wanted) in mismatches.items()
                )
            )
    elif any(args.cache_dir.rglob("*.npz")):
        raise ValueError(
            f"LoMa keypoint files exist in {args.cache_dir} without a manifest; "
            "refusing to reuse features whose detector/preprocessing settings are unknown. "
            "Choose a new cache directory."
        )

    # Persist compatibility metadata before writing frames. If the job is
    # interrupted, the next run can safely validate the settings and resume
    # only missing/corrupt records instead of trusting unlabelled files.
    manifest = {
        **expected,
        "splits": args.splits,
        "n_frames": len(frames),
        "complete": False,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary_manifest, manifest_path)

    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for frame in frames:
        relative = frame.relative_to(args.data_root).as_posix()
        cache_path = args.cache_dir / Path(relative).with_suffix(".npz")
        if cache_path.is_file() and _existing_record_is_valid(cache_path, args.num_keypoints):
            continue
        grouped[target_hw(frame, args.resize)].append(frame)

    model.eval()
    with torch.inference_mode():
        for (height, width), group in sorted(grouped.items()):
            for start in tqdm(range(0, len(group), args.batch_size), desc=f"keypoints {height}x{width}"):
                paths = group[start:start + args.batch_size]
                images = torch.stack([load_image(path, args.resize) for path in paths]).to(device)
                if images.shape[-2:] != (height, width):
                    raise RuntimeError("keypoint-cache preprocessing shape mismatch")
                detected = model._detector.detect(
                    {"image": images}, num_keypoints=args.num_keypoints
                )["keypoints"]
                if detected.shape[1] != args.num_keypoints:
                    raise ValueError(
                        f"DaD returned {detected.shape[1]} keypoints, expected {args.num_keypoints}"
                    )
                for index, frame in enumerate(paths):
                    relative = frame.relative_to(args.data_root).as_posix()
                    write_frame(
                        args.cache_dir / Path(relative).with_suffix(".npz"),
                        detected[index].cpu().numpy(),
                        (height, width),
                    )

    missing = [
        str(frame.relative_to(args.data_root))
        for frame in frames
        if not (args.cache_dir / Path(frame.relative_to(args.data_root)).with_suffix(".npz")).is_file()
    ]
    if missing:
        raise RuntimeError(f"keypoint cache incomplete ({len(missing)} missing), e.g. {missing[:3]}")
    manifest = {
        **expected,
        "splits": args.splits,
        "n_frames": len(frames),
        "complete": True,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary_manifest, manifest_path)
    print(f"LoMa keypoint cache ready: {len(frames):,} frames at {args.cache_dir}")


def main() -> None:
    build_cache(parse_args())


if __name__ == "__main__":
    main()
