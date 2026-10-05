"""Mine CzechLynx positive/negative frame pairs with RDD/LightGlue or LoMa."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import time

import torch

from wildmatch.mining.batched_processing import sequence_score_per_video_and_per_frame
from wildmatch.mining.czechlynx_dataset import CzechLynxCollection
from wildmatch.mining.lightglue_masked import LightGlueMasked
from wildmatch.mining.lynx_benchmark import load_cached_feat, sample_frames


def list_collections(root: Path, split: str) -> list[CzechLynxCollection]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split {split!r} not found under {root}")
    output = []
    for identity_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for source_dir in sorted(p for p in identity_dir.iterdir() if p.is_dir()):
            frames = sorted(source_dir.rglob("frame_*.jpg"))
            if frames:
                output.append(CzechLynxCollection(split, identity_dir.name, source_dir.name, frames))
    return output


def validate_lightglue_weights(weights: Path) -> None:
    """Reject detector checkpoints before LightGlue's permissive load."""
    if not weights.is_file():
        raise FileNotFoundError(f"LightGlue checkpoint not found: {weights}")
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    keys = set(state) if isinstance(state, dict) else set()
    if not any(key.startswith(("transformers.", "log_assignment.")) for key in keys):
        raise ValueError(
            f"{weights} does not look like an RDD LightGlue checkpoint; "
            "pass RDD_lg-v2.pth via --lg_weights, not RDD-v2.pth"
        )


def build_masked_lg(device: torch.device, weights: Path):
    validate_lightglue_weights(weights)
    config = {
        "name": "lightglue",
        "input_dim": 256,
        "descriptor_dim": 256,
        "add_scale_ori": False,
        "n_layers": 9,
        "num_heads": 4,
        "flash": True,
        "mp": False,
        "filter_threshold": 0.01,
        "depth_confidence": -1,
        "width_confidence": -1,
        "weights": str(weights),
    }
    return LightGlueMasked("rdd", **config).to(device).eval()


def is_exact_query_frame(candidate_path: Path, query_path: Path, split: str) -> bool:
    """Whether a gallery frame must be excluded for this query."""
    return split == "train" and candidate_path == query_path


def select_diverse_topk(candidates: list[dict], k: int) -> list[dict]:
    by_collection: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_collection[candidate["collection"]].append(candidate)
    for values in by_collection.values():
        values.sort(key=lambda item: item["score"], reverse=True)
    order = sorted(by_collection, key=lambda name: by_collection[name][0]["score"], reverse=True)
    selected = []
    rank = 0
    while len(selected) < k:
        added = False
        for name in order:
            if rank < len(by_collection[name]):
                selected.append(by_collection[name][rank])
                added = True
                if len(selected) == k:
                    break
        if not added:
            break
        rank += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--rdd_weights", type=Path, default=None)
    parser.add_argument("--lg_weights", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--backend", choices=["rdd", "loma"], default="rdd")
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--query_id", type=int, required=True)
    parser.add_argument("--frames_per_collection", type=int, default=20)
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--top_m", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump_report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gallery = list_collections(args.dataset_root, "train")
    queries = list_collections(args.dataset_root, args.split)
    if not gallery:
        raise RuntimeError("no training gallery collections found")
    if not 0 <= args.query_id < len(queries):
        raise ValueError(f"query_id={args.query_id} out of range 0..{len(queries) - 1}")
    if args.backend == "rdd":
        lg_weights = args.lg_weights or args.weights
        if lg_weights is None:
            raise ValueError("RDD mining requires --lg_weights or --weights")
        model = build_masked_lg(device, lg_weights)
    else:
        loma_weights = args.weights or args.lg_weights
        if loma_weights is None:
            raise ValueError("LoMa mining requires --weights or --lg_weights")
        from wildmatch.mining.loma_backend import build_loma, score_all_loma

        model = build_loma(device, loma_weights, args.variant)

    def cache_path(path: Path) -> Path:
        return args.cache_dir / path.relative_to(args.dataset_root).with_suffix(".npz")

    def load_collection(collection):
        paths = sample_frames(collection.frame_paths, args.frames_per_collection)
        return paths, [load_cached_feat(cache_path(path)) for path in paths]

    query = queries[args.query_id]
    query_paths, query_features = load_collection(query)
    gallery_paths, gallery_features, gallery_names, gallery_ids = [], [], [], []
    started = time()
    for collection in gallery:
        paths, features = load_collection(collection)
        for path, feature in zip(paths, features):
            gallery_paths.append(path)
            gallery_features.append(feature)
            gallery_names.append(collection.name)
            gallery_ids.append(collection.identity)
    if not gallery_features:
        raise RuntimeError("gallery is empty after exact-frame exclusion")

    if args.backend == "loma":
        scores = score_all_loma(model, query_features, gallery_features, device, batch_size=32)
    else:
        chunks = []
        for start in range(0, len(gallery_features), 32):
            chunks.append(
                sequence_score_per_video_and_per_frame(
                    model, query_features, gallery_features[start : start + 32], device
                )
            )
        scores = torch.cat(chunks, dim=1)
    frames = []
    for query_index, query_path in enumerate(query_paths):
        positives, negatives = [], []
        for column, score in enumerate(scores[query_index].tolist()):
            candidate = {
                "score": float(score),
                "frame": str(gallery_paths[column]),
                "identity": gallery_ids[column],
                "collection": gallery_names[column],
            }
            # Keep the query collection in the train gallery so its other
            # frames remain valid positives; exclude only this exact query
            # frame from its own candidate pool.
            if is_exact_query_frame(gallery_paths[column], query_path, args.split):
                continue
            (positives if candidate["identity"] == query.identity else negatives).append(candidate)
        selected_pos = select_diverse_topk(positives, args.top_k_frames)
        selected_neg = select_diverse_topk(negatives, args.top_k_frames)
        values = [item["score"] for item in selected_pos + selected_neg]
        frames.append(
            {
                "query_frame": str(query_path),
                "query_frame_index": query_index,
                "selection_score": max(values) if values else 0.0,
                "positives": selected_pos,
                "negatives": selected_neg,
            }
        )
    selected = sorted(frames, key=lambda item: item["selection_score"], reverse=True)[: args.top_m]
    selected.sort(key=lambda item: item["query_frame_index"])
    output = {
        "dataset": "CzechLynx",
        "query": query.name,
        "query_identity": query.identity,
        "query_source": query.source,
        "query_split": args.split,
        "query_id": args.query_id,
        "backend": args.backend,
        "variant": args.variant if args.backend == "loma" else None,
        "weights": str((args.weights or args.lg_weights or args.rdd_weights).resolve()),
        "frames_per_collection": args.frames_per_collection,
        "top_k_frames": args.top_k_frames,
        "top_m": args.top_m,
        "gallery_collections": len(gallery),
        "selected_frames": selected,
        "all_frames": frames,
        "elapsed_s": time() - started,
    }
    args.dump_report.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(f"{args.dump_report}_{args.split}_{args.query_id}.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {output_path} ({len(selected)} selected query frames)")


if __name__ == "__main__":
    main()
