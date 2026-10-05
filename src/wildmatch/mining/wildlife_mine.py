"""Mine positive/negative frame pairs for one WildlifeReID collection query."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import time

import torch

from wildmatch.mining.batched_processing import sequence_score_per_video_and_per_frame
from wildmatch.mining.lightglue_masked import LightGlueMasked
from wildmatch.mining.lynx_benchmark import load_cached_feat, sample_frames
from wildmatch.mining.wildlife_dataset import list_collections


def weights_fingerprint(path: Path) -> str:
    """Return a stable fingerprint for a checkpoint file or LoMa bundle."""
    if path.is_dir():
        metadata_path = path / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        base = metadata.get("base_weights")
        if base:
            base_path = Path(base).expanduser()
            if not base_path.is_absolute():
                base_path = path / base_path
            if base_path.is_file():
                path = base_path
        else:
            path = next((path / name for name in ("model.safetensors", "matcher.safetensors", "weights.pth") if (path / name).is_file()), path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_loma_cache(cache_dir: Path, variant: str, weights: Path) -> dict:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"LoMa cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "backend": "loma",
        "variant": variant,
        "resize": 512,
        "num_keypoints": 512,
        "patch_size": 14,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    cached_hash = manifest.get("weights_sha256")
    if cached_hash:
        actual_hash = weights_fingerprint(weights)
        if cached_hash != actual_hash:
            mismatches["weights_sha256"] = (cached_hash, actual_hash)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (wanted {wanted!r})"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"incompatible LoMa cache at {cache_dir}: {details}")
    return manifest


def build_backend_model(backend: str, device: torch.device, weights: Path, variant: str):
    if backend == "rdd":
        return build_model(device, weights)
    if backend == "loma":
        from wildmatch.mining.loma_backend import build_loma

        return build_loma(device, weights, variant)
    raise ValueError(f"unsupported mining backend {backend!r}; expected rdd or loma")


def validate_lightglue_weights(weights: Path) -> None:
    if not weights.is_file():
        raise FileNotFoundError(f"LightGlue checkpoint not found: {weights}")
    state = torch.load(str(weights), map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    keys = set(state) if isinstance(state, dict) else set()
    if not any(key.startswith(("transformers.", "log_assignment.")) for key in keys):
        raise ValueError(f"{weights} does not look like an RDD LightGlue checkpoint")


def build_model(device: torch.device, weights: Path):
    validate_lightglue_weights(weights)
    config = {
        "name": "lightglue", "input_dim": 256, "descriptor_dim": 256,
        "add_scale_ori": False, "n_layers": 9, "num_heads": 4,
        "flash": True, "mp": False, "filter_threshold": 0.01,
        "depth_confidence": -1, "width_confidence": -1, "weights": str(weights),
    }
    return LightGlueMasked("rdd", **config).to(device).eval()


def select_diverse_topk(candidates: list[dict], k: int) -> list[dict]:
    by_collection: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        by_collection[candidate["collection"]].append(candidate)
    for values in by_collection.values():
        values.sort(key=lambda item: item["score"], reverse=True)
    order = sorted(by_collection, key=lambda name: by_collection[name][0]["score"], reverse=True)
    selected: list[dict] = []
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
    parser.add_argument("--lg_weights", type=Path, required=True)
    parser.add_argument("--backend", choices=["rdd", "loma"], default="rdd")
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--dataset_id", default="wildlife-reid-10k")
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
    if args.backend == "loma":
        validate_loma_cache(args.cache_dir, args.variant, args.lg_weights)
    model = build_backend_model(args.backend, device, args.lg_weights, args.variant)

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
    chunks = []
    for start in range(0, len(gallery_features), 32):
        gallery_chunk = gallery_features[start:start + 32]
        if args.backend == "loma":
            from wildmatch.mining.loma_backend import score_all_loma

            chunk = score_all_loma(model, query_features, gallery_chunk, device, batch_size=32)
        else:
            chunk = sequence_score_per_video_and_per_frame(
                model, query_features, gallery_chunk, device
            )
        chunks.append(chunk)
    scores = torch.cat(chunks, dim=1)
    frames = []
    for query_index, query_path in enumerate(query_paths):
        positives, negatives = [], []
        for column, score in enumerate(scores[query_index].tolist()):
            if gallery_paths[column] == query_path:
                continue
            candidate = {
                "score": float(score), "frame": str(gallery_paths[column]),
                "identity": gallery_ids[column], "collection": gallery_names[column],
            }
            (positives if candidate["identity"] == query.identity else negatives).append(candidate)
        selected_pos = select_diverse_topk(positives, args.top_k_frames)
        selected_neg = select_diverse_topk(negatives, args.top_k_frames)
        values = [item["score"] for item in selected_pos + selected_neg]
        frames.append({
            "query_frame": str(query_path), "query_frame_index": query_index,
            "selection_score": max(values) if values else 0.0,
            "positives": selected_pos, "negatives": selected_neg,
        })
    selected = sorted(frames, key=lambda item: item["selection_score"], reverse=True)[:args.top_m]
    selected.sort(key=lambda item: item["query_frame_index"])
    output = {
        "dataset": args.dataset_id, "query": query.name, "query_identity": query.identity,
        "query_split": args.split, "query_id": args.query_id,
        "backend": args.backend,
        "variant": args.variant if args.backend == "loma" else None,
        "weights": str(args.lg_weights.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "frames_per_collection": args.frames_per_collection,
        "top_k_frames": args.top_k_frames, "top_m": args.top_m,
        "gallery_collections": len(gallery), "selected_frames": selected,
        "all_frames": frames, "elapsed_s": time() - started,
    }
    args.dump_report.parent.mkdir(parents=True, exist_ok=True)
    output_path = Path(f"{args.dump_report}_{args.split}_{args.query_id}.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {output_path} ({len(selected)} selected query frames)")


if __name__ == "__main__":
    main()
