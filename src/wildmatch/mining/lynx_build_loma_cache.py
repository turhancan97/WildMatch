"""Build a fixed-keypoint LoMa feature cache for the lynx benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.lynx_dataset import list_sequences, sample_frames
from scripts.loma_backend import build_loma, extract_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Build a LoMa feature cache")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--frames_per_seq", type=int, default=20)
    parser.add_argument("--resize_max", type=int, default=512)
    parser.add_argument("--num_keypoints", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_loma(device, args.weights, args.variant)
    frames = []
    for split in ("train", "test"):
        for sequence in list_sequences(args.dataset_root, split):
            frames.extend(sample_frames(sequence.frame_paths, args.frames_per_seq))
    for index, frame_path in enumerate(frames, start=1):
        relative = frame_path.relative_to(args.dataset_root)
        output = args.cache_dir / relative.parent / f"{frame_path.stem}.npz"
        if output.exists() and not args.overwrite:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **extract_frame(model, frame_path, device, args.num_keypoints, args.resize_max))
        if index % 100 == 0:
            print(f"Cached {index}/{len(frames)} frames")


if __name__ == "__main__":
    main()

