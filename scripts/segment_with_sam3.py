#!/usr/bin/env python3
"""Remove image backgrounds with text-prompted SAM3 and build pre-masked metadata.

Two independent steps, selected by flags:

``--segment`` (GPU)
    Runs SAM3 on every row of ``--csv`` (paths relative to ``--root``) with a text
    ``--prompt``. Every image is assumed to show exactly one animal, so all detected
    instances are merged into one mask (``--merge union``): an occluding hand or
    finger makes SAM3 return the visible pieces of one animal as separate instances,
    and keeping only the best piece drops legs, tails or half the body. Rows with no
    detection are retried at ``--fallback-threshold`` and then with each
    ``--fallback-prompts`` entry; ``--threshold-override PATH=T`` pins a lower first
    threshold for individually reviewed images. Nothing is ever dropped: an image
    that still has no detection gets an empty mask and ``n_detections=0``.
    Writes ``<out-dir>/masked_images/<relative path>`` (background set to 0, JPEG
    q95) and ``<out-dir>/masks.csv`` (full-size COCO-RLE mask, best score,
    foreground fraction, instance count, threshold and prompt used).

``--write-metadata`` (CPU)
    Joins the source CSV with ``masks.csv`` into a pre-masked metadata file for the
    probe/fine-tuning pipelines: ``path`` points at the masked image, the original
    path moves to ``original_path``, and ``mask`` plus ``sam3_*`` quality columns are
    added. ``--split-map database=train,query=test --split-map-column
    split_train_test`` adds a second split column for tools that expect train/test
    values, leaving the original split column untouched. Fails closed on missing
    rows, duplicate paths, or masked files that do not exist.

The source CSV and images are only read. SAM3 runs in the ``lynx-app`` conda
environment, whose CUDA 13 PyTorch build has no kernels for V100 GPUs: use an A100
or H100 node. Used for SalamanderID2025 on 2026-09-29 with ``--prompt Salamander``
and ``--threshold-override query/images/9d1fc96e28c0058e_1277.jpg=0.1``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd



def _profile_value(key: str) -> Optional[str]:
    """A location from the active path profile (wildmatch.paths). This script runs in the
    separate `lynx-app` environment, where the package is usually not installed, so it falls
    back to importing wildmatch.paths from the repository's src/ (it only needs OmegaConf)."""
    try:
        from wildmatch.paths import path as profile_path
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from wildmatch.paths import path as profile_path
    value = profile_path(key)
    return None if value is None else str(value)


DEFAULT_CHECKPOINT = _profile_value("external.sam3_checkpoint")
DEFAULT_SAM3_DIR = _profile_value("external.sam3_repo")
MASKS_FILE = "masks.csv"
MASKED_DIR = "masked_images"
SAM3_COLUMNS = {
    "best_score": "sam3_score",
    "fg_fraction": "sam3_fg_fraction",
    "n_detections": "sam3_n_instances",
    "threshold_used": "sam3_threshold",
    "prompt_used": "sam3_prompt",
}


class MetadataError(ValueError):
    """Raised when pre-masked metadata cannot be built safely."""


# ── mask helpers (no SAM3 / torch needed) ─────────────────────────────────────
def merge_instances(masks: np.ndarray, scores: np.ndarray, merge: str) -> Tuple[Optional[np.ndarray], int, float]:
    """Combine SAM3 instance masks (N, H, W) into one boolean mask.

    Returns (mask or None when N == 0, instance count, best score).
    """
    if masks.ndim == 4:
        masks = masks[:, 0]
    if len(scores) == 0 or masks.shape[0] == 0:
        return None, 0, 0.0
    masks = masks.astype(bool)
    best = int(np.argmax(scores))
    if merge == "union":
        mask = masks.any(axis=0)
    elif merge == "best":
        mask = masks[best]
    else:
        raise ValueError(f"unknown merge policy: {merge!r}")
    return mask, int(len(scores)), float(scores[best])


def encode_mask(mask: np.ndarray) -> str:
    """Full-size boolean mask -> COCO-RLE JSON string (pycocotools layout)."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return json.dumps(rle)


def decode_mask(payload: str) -> np.ndarray:
    from pycocotools import mask as mask_utils

    rle = json.loads(payload)
    rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def parse_mapping(text: str) -> Dict[str, str]:
    """``'a=b,c=d'`` -> ``{'a': 'b', 'c': 'd'}``; rejects malformed or repeated keys."""
    mapping: Dict[str, str] = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        if item.count("=") != 1:
            raise ValueError(f"malformed mapping entry {item!r}; expected KEY=VALUE")
        key, value = (side.strip() for side in item.split("="))
        if not key or not value or key in mapping:
            raise ValueError(f"invalid or repeated mapping key in {item!r}")
        mapping[key] = value
    return mapping


def parse_threshold_overrides(items: Iterable[str]) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for item in items:
        path, sep, value = item.rpartition("=")
        if not sep or not path:
            raise ValueError(f"malformed --threshold-override {item!r}; expected PATH=THRESHOLD")
        overrides[path] = float(value)
    return overrides


# ── metadata ──────────────────────────────────────────────────────────────────
def build_masked_metadata(
    source: pd.DataFrame,
    masks: pd.DataFrame,
    root: Path,
    split_col: Optional[str] = None,
    split_map: Optional[Dict[str, str]] = None,
    split_map_column: Optional[str] = None,
) -> pd.DataFrame:
    """Source metadata + masks.csv -> pre-masked metadata (see module docstring)."""
    for frame, name in ((source, "source"), (masks, MASKS_FILE)):
        if "path" not in frame.columns:
            raise MetadataError(f"{name} has no 'path' column")
        if frame["path"].duplicated().any():
            raise MetadataError(f"{name} has duplicate paths")
    missing = sorted(set(source["path"]) - set(masks["path"]))
    if missing:
        raise MetadataError(f"{len(missing)} source rows have no mask, e.g. {missing[:3]}")
    for col in ("masked_path", "mask", *SAM3_COLUMNS):
        if col not in masks.columns:
            raise MetadataError(f"{MASKS_FILE} is missing column {col!r}")
    reserved = {"original_path", "mask", *SAM3_COLUMNS.values()}
    clash = sorted(reserved & set(source.columns))
    if clash:
        raise MetadataError(f"source already has columns {clash}")

    joined = source.merge(masks[["path", "masked_path", "mask", *SAM3_COLUMNS]], on="path",
                          how="left", validate="one_to_one")
    absent = [p for p in joined["masked_path"] if not (Path(root) / p).is_file()]
    if absent:
        raise MetadataError(f"{len(absent)} masked images do not exist, e.g. {absent[:3]}")

    out = joined.rename(columns={"path": "original_path", "masked_path": "path", **SAM3_COLUMNS})
    columns = list(source.columns) + ["original_path", "mask", *SAM3_COLUMNS.values()]
    if split_map:
        if not split_col or not split_map_column:
            raise MetadataError("--split-map needs --split-col and --split-map-column")
        if split_col not in out.columns:
            raise MetadataError(f"source has no split column {split_col!r}")
        if split_map_column in out.columns:
            raise MetadataError(f"column {split_map_column!r} already exists")
        unmapped = sorted(set(out[split_col].astype(str)) - set(split_map))
        if unmapped:
            raise MetadataError(f"split values without a mapping: {unmapped}")
        out[split_map_column] = out[split_col].astype(str).map(split_map)
        columns.append(split_map_column)
    return out[columns]


# ── segmentation (GPU) ────────────────────────────────────────────────────────
def run_segmentation(args: argparse.Namespace) -> None:
    from PIL import Image

    if not args.sam3_dir or not args.checkpoint:
        raise SystemExit("SAM 3 location unknown: pass --sam3-dir and --checkpoint, or set paths "
                         "external.sam3_repo and external.sam3_checkpoint (e.g. paths profile gmum)")
    sys.path.append(args.sam3_dir)
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    overrides = parse_threshold_overrides(args.threshold_override)
    meta = pd.read_csv(args.root / args.csv)
    unknown = sorted(set(overrides) - set(meta["path"]))
    if unknown:
        raise SystemExit(f"--threshold-override paths not in {args.csv}: {unknown}")
    if args.limit:
        meta = meta.sample(args.limit, random_state=args.sample_seed) if args.sample_seed is not None else meta.head(args.limit)

    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False, device="cuda", eval_mode=True)
    processor = Sam3Processor(model, device="cuda", confidence_threshold=args.threshold)

    def detect(state: dict):
        scores, masks = state.get("scores"), state.get("masks")
        if scores is None or masks is None:
            return None, 0, 0.0
        return merge_instances(masks.detach().cpu().numpy(), scores.detach().float().cpu().numpy(), args.merge)

    rows, t0 = [], time.time()
    for n, rel in enumerate(meta["path"].astype(str), 1):
        image = Image.open(args.root / rel).convert("RGB")
        first = overrides.get(rel, args.threshold)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            processor.set_confidence_threshold(first)
            base = processor.set_image(image, state={})
            state = processor.set_text_prompt(prompt=args.prompt, state=base)
            mask, n_det, score = detect(state)
            used_threshold, used_prompt = first, args.prompt
            if mask is None and args.fallback_threshold < used_threshold:
                state = processor.set_confidence_threshold(args.fallback_threshold, state)
                mask, n_det, score = detect(state)
                used_threshold = args.fallback_threshold
            for prompt in args.fallback_prompts:
                if mask is not None:
                    break
                processor.reset_all_prompts(base)
                mask, n_det, score = detect(processor.set_text_prompt(prompt=prompt, state=base))
                used_prompt = prompt
        width, height = image.size
        if mask is None:
            mask = np.zeros((height, width), dtype=bool)
        pixels = np.asarray(image).copy()
        pixels[~mask] = 0
        masked_rel = Path(MASKED_DIR) / rel
        (args.out_dir / masked_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.out_dir / masked_rel, quality=95)
        ys, xs = np.nonzero(mask)
        rows.append({
            "path": rel, "masked_path": masked_rel.as_posix(), "n_detections": n_det,
            "best_score": round(score, 4), "threshold_used": used_threshold, "prompt_used": used_prompt,
            "merge": args.merge, "fg_fraction": round(float(mask.mean()), 4),
            "bbox": json.dumps([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else []),
            "mask": encode_mask(mask),
        })
        if n % 50 == 0:
            print(f"[sam3] {n}/{len(meta)} images, {(time.time() - t0) / n:.2f} s/img", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / MASKS_FILE, index=False)
    print(f"[sam3] done: {len(out)} images, prompt={args.prompt!r}, merge={args.merge}, "
          f"empty={int((out.n_detections == 0).sum())}, "
          f"fallback_threshold_used={int((out.threshold_used < args.threshold).sum())}, "
          f"fallback_prompt_used={int((out.prompt_used != args.prompt).sum())}, "
          f"median fg={out.fg_fraction.median():.3f}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="Dataset root; CSV paths are relative to it")
    p.add_argument("--csv", type=Path, required=True, help="Source metadata CSV (relative to --root or absolute)")
    p.add_argument("--out-dir", type=Path, default=None, help="Where masked_images/ and masks.csv go (default: --root)")
    p.add_argument("--segment", action="store_true", help="Run SAM3 (GPU)")
    p.add_argument("--write-metadata", type=Path, default=None,
                   help="Write pre-masked metadata to this CSV (relative to --root or absolute)")
    p.add_argument("--prompt", default="Salamander")
    p.add_argument("--fallback-prompts", nargs="*", default=["Animal"])
    p.add_argument("--merge", choices=["union", "best"], default="union")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fallback-threshold", type=float, default=0.25)
    p.add_argument("--threshold-override", action="append", default=[], metavar="PATH=THRESHOLD")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--sam3-dir", default=DEFAULT_SAM3_DIR)
    p.add_argument("--limit", type=int, default=0, help="Segment only N rows (pilot)")
    p.add_argument("--sample-seed", type=int, default=None, help="With --limit, sample rows randomly")
    p.add_argument("--split-col", default="split")
    p.add_argument("--split-map", default="", help="e.g. database=train,query=test")
    p.add_argument("--split-map-column", default=None, help="e.g. split_train_test")
    args = p.parse_args(argv)
    if not args.segment and args.write_metadata is None:
        p.error("choose --segment and/or --write-metadata")
    args.out_dir = args.out_dir or args.root
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.segment:
        run_segmentation(args)
    if args.write_metadata is not None:
        source = pd.read_csv(args.root / args.csv)
        masks = pd.read_csv(args.out_dir / MASKS_FILE)
        metadata = build_masked_metadata(
            source, masks, args.out_dir, split_col=args.split_col,
            split_map=parse_mapping(args.split_map) or None, split_map_column=args.split_map_column,
        )
        target = args.root / args.write_metadata
        metadata.to_csv(target, index=False)
        extra = f", {args.split_map_column}={metadata[args.split_map_column].value_counts().to_dict()}" \
            if args.split_map_column in metadata.columns else ""
        print(f"[sam3] wrote {target} ({len(metadata)} rows{extra})")


if __name__ == "__main__":
    main()
