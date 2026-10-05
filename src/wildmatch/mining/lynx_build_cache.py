"""
Lightweight sequence-level retrieval benchmark for the lynx dataset.

Assumptions about dataset layout (confirmed):
    root/
      train/
        lynx_<id>/<site>/<sequence_id>/frame_XXXX.jpg
      test/
        lynx_<id>/<site>/<sequence_id>/frame_XXXX.jpg

The script:
 1) Indexes sequences.
 2) Samples up to N frames per sequence (uniform).
 3) Extracts RDD keypoints+descriptors (cached to .npz).
 4) Matches query (test) sequences to gallery (train) sequences with LightGlue.
 5) Aggregates frame scores to sequence scores and reports retrieval metrics.

Dependencies: torch, numpy, PIL, tqdm. Uses RDD+LightGlue from this repo.
"""

import argparse
import warnings
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from wildmatch.mining.lynx_benchmark import build_models, ensure_cache, extract_frame, list_sequences, sample_frames
from wildmatch.vendor.rdd import CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lynx sequence-level retrieval benchmark with RDD+LightGlue.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/shared/sets/datasets/confidential/lynx/processed_frames/segmented/dfk-hr-cleaned"),
        help="Root containing train/ and test/ splits.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("./outputs/lynx_cache"),
        help="Where to store per-frame feature npz files.",
    )
    parser.add_argument("--config_path", type=Path, default=CONFIG_PATH, help="RDD config path (unused).")
    parser.add_argument("--weights", type=Path, default=None, help="RDD weights path.")
    parser.add_argument("--frames_per_seq", type=int, default=10, help="Max frames sampled per sequence.")
    parser.add_argument(
        "--resize_max",
        type=int,
        default=1024,
        help="Resize longest side before RDD extract (matches demo). Use 0 to disable.",
    )
    parser.add_argument("--top_k", type=int, default=2048, help="Top-K keypoints for RDD soft detection.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model/matcher.")
    parser.add_argument("--limit_seqs", type=int, default=0, help="Debug: limit number of query sequences.")
    return parser.parse_args()


def get_new_image_size(h, w, resize=1600):
    aspect_ratio = w / h
    if h > w:
        size = (resize, int(resize * aspect_ratio))
    else:
        size = (int(resize / aspect_ratio), resize)

    size = list(map(lambda x: int(x // 32 * 32), size))  # make sure size is divisible by 32
    return size


def parse_input(x_path, resize, device):
    x = cv2.imread(x_path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    if len(x.shape) == 3:
        x = x[None, ...]

    if isinstance(x, np.ndarray):
        x = torch.tensor(x).permute(0, 3, 1, 2) / 255

    h, w = x.shape[-2:]
    size = h, w

    if resize is not None:
        size = get_new_image_size(h, w, resize)
        x = kornia.geometry.transform.resize(
            x,
            size,
            side="long",
            antialias=True,
            align_corners=None,
            interpolation="bilinear",
        )
    scale = torch.Tensor([x.shape[-1] / w, x.shape[-2] / h]).to(device)
    x = torch.tensor(x).to(device)

    return x, scale


def _load_image_exreid(
    image_path,
    resize_max,
    device,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")

    if resize_max and resize_max > 0:
        w, h = img.size
        scale = float(resize_max) / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_train_approach(
    image_path,
    resize_max,
    device,
) -> torch.Tensor:
    """Matches contrastive_finetuning.train_common exactly: PIL decode +
    ToTensor (float [0,1]) + resize_long_side — plain F.interpolate
    bilinear, NO antialiasing, target rounded down to a multiple of 32.
    Swapped in here (see main()) so this repo's cache is built with the same
    preprocessing lynx-finetuning-lg's own eval uses, instead of
    parse_input's cv2-decode + antialiased-kornia-resize, which measurably
    changes RDD's keypoints/descriptors relative to fresh extraction on the
    training side (see debug_cross_pipeline.py / debug_preprocessing.py)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, h, w = x.shape
    scale = resize_max / max(h, w)
    new_h, new_w = int(h * scale) // 32 * 32, int(w * scale) // 32 * 32
    return F.interpolate(x, (new_h, new_w), mode="bilinear", align_corners=False)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing sequences...")
    train_seqs = list_sequences(args.dataset_root, "train")
    test_seqs = list_sequences(args.dataset_root, "test")
    if args.limit_seqs > 0:
        test_seqs = test_seqs[: args.limit_seqs]
    print(f"Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

    print("Building models...")
    rdd_model, _ = build_models(args.config_path, args.weights, device, args.top_k)

    def feat_cache_path(frame_path: Path) -> Path:
        rel = frame_path.relative_to(args.dataset_root)
        return cache_dir / rel.parent / f"{frame_path.stem}.npz"

    all_sequences = train_seqs + test_seqs
    for seq in tqdm(all_sequences, desc="Extracting features"):
        sampled = sample_frames(seq.frame_paths, args.frames_per_seq)
        for fp in sampled:
            cp = feat_cache_path(fp)
            if cp.exists():
                warnings.warn(f"{cp} is already in cache", UserWarning)
                continue
            # img_torch, _ = parse_input(fp, resize=args.resize_max, device=device)
            # img_torch = _load_image_exreid(fp, resize_max=args.resize_max, device=device)
            img_torch = _load_train_approach(fp, resize_max=args.resize_max, device=device)
            feat = extract_frame(rdd_model, img_torch, device, args.top_k)
            ensure_cache(cp, feat)


if __name__ == "__main__":
    main()
