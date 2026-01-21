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
import json
import math
import os
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RDD.RDD import build as build_rdd
from RDD.matchers import LightGlue


@dataclass
class FrameFeat:
    keypoints: np.ndarray  # [K, 2]
    descriptors: np.ndarray  # [K, 256]
    scores: np.ndarray  # [K]
    image_size: np.ndarray  # [2] (H, W)


@dataclass
class SequenceEntry:
    split: str
    lynx_id: str
    site: str
    sequence_id: str
    frame_paths: List[Path]

    @property
    def name(self) -> str:
        return f"{self.lynx_id}/{self.site}/{self.sequence_id}"


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
    parser.add_argument("--config_path", type=Path, default=Path("./configs/default.yaml"), help="RDD config path.")
    parser.add_argument("--weights", type=Path, default=Path("./weights/RDD-v2.pth"), help="RDD weights path.")
    parser.add_argument("--frames_per_seq", type=int, default=10, help="Max frames sampled per sequence.")
    parser.add_argument(
        "--resize_max",
        type=int,
        default=1024,
        help="Resize longest side before RDD extract (matches demo). Use 0 to disable.",
    )
    parser.add_argument("--top_k", type=int, default=2048, help="Top-K keypoints for RDD soft detection.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for model/matcher.")
    parser.add_argument("--top_m_pool", type=int, default=5, help="Top-M frame scores used for sequence pooling.")
    parser.add_argument("--matcher_threshold", type=float, default=0.0, help="LightGlue conf threshold for counting matches.")
    parser.add_argument("--dump_report", type=Path, default=Path("./outputs/lynx_report"), help="Where to save metrics.")
    parser.add_argument("--metrics_plot", type=Path, default=Path("./outputs/lynx_metrics.png"), help="Where to save the per-lynx_id metrics plot.")
    parser.add_argument("--limit_seqs", type=int, default=0, help="Debug: limit number of query sequences.")
    return parser.parse_args()


def list_sequences(root: Path, split: str) -> List[SequenceEntry]:
    entries: List[SequenceEntry] = []
    split_dir = root / split
    for lynx_dir in sorted(split_dir.iterdir()):
        if not lynx_dir.is_dir():
            continue
        lynx_id = lynx_dir.name
        for site_dir in sorted(lynx_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            site = site_dir.name
            for seq_dir in sorted(site_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("frame_*.jpg"))
                if not frames:
                    continue
                entries.append(
                    SequenceEntry(
                        split=split,
                        lynx_id=lynx_id,
                        site=site,
                        sequence_id=seq_dir.name,
                        frame_paths=frames,
                    )
                )
    return entries


def sample_frames(frame_paths: List[Path], max_frames: int) -> List[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    idxs = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=int)
    return [frame_paths[i] for i in idxs]


def load_image(path: Path, resize_max: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if resize_max and resize_max > 0:
        w, h = img.size
        scale = resize_max / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor


def ensure_cache(path: Path, feat: FrameFeat):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=feat.keypoints,
        descriptors=feat.descriptors,
        scores=feat.scores,
        image_size=feat.image_size,
    )


def load_cached_feat(path: Path) -> FrameFeat:
    data = np.load(path)
    return FrameFeat(
        keypoints=data["keypoints"],
        descriptors=data["descriptors"],
        scores=data["scores"],
        image_size=data["image_size"],
    )


@torch.no_grad()
def extract_frame(model, img: torch.Tensor, device: torch.device, top_k: int) -> FrameFeat:
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)
    img = img.to(device)
    out = model.extract(img)[0]
    return FrameFeat(
        keypoints=out["keypoints"].cpu().numpy(),
        descriptors=out["descriptors"].cpu().numpy(),
        scores=out["scores"].cpu().numpy(),
        image_size=np.array(img.shape[-2:], dtype=np.int32),  # (H, W)
    )


def score_pair_lightglue(lg, fa: FrameFeat, fb: FrameFeat, device: torch.device) -> Tuple[float, int]:
    # Prepare batch-1 inputs
    k0 = torch.from_numpy(fa.keypoints).to(device).unsqueeze(0)
    k1 = torch.from_numpy(fb.keypoints).to(device).unsqueeze(0)
    d0 = torch.from_numpy(fa.descriptors).to(device).unsqueeze(0)
    d1 = torch.from_numpy(fb.descriptors).to(device).unsqueeze(0)
    size0 = torch.tensor(fa.image_size[::-1].copy(), device=device).unsqueeze(0)  # (W,H)
    size1 = torch.tensor(fb.image_size[::-1].copy(), device=device).unsqueeze(0)

    pred = lg(
        {
            "image0": {"keypoints": k0, "descriptors": d0, "image_size": size0},
            "image1": {"keypoints": k1, "descriptors": d1, "image_size": size1},
        }
    )
    if pred["scores"][0].numel() == 0:
        return 0.0, 0
    conf = pred["scores"][0]
    sum_conf = conf.sum().item()
    norm = min(max(1, fa.keypoints.shape[0]), max(1, fb.keypoints.shape[0]))
    score = sum_conf / norm
    return score, int((conf > 0).sum().item())


def aggregate_sequence_score(
    lg,
    q_frames: List[FrameFeat],
    g_frames: List[FrameFeat],
    device: torch.device,
    top_m: int,
) -> Dict[str, float]:
    per_q_best: List[float] = []
    per_q_matches: List[int] = []
    for qf in q_frames:
        best_score = 0.0
        best_matches = 0
        for gf in g_frames:
            score, nm = score_pair_lightglue(lg, qf, gf, device)
            if score > best_score:
                best_score = score
                best_matches = nm
        per_q_best.append(best_score)
        per_q_matches.append(best_matches)
    if not per_q_best:
        return {"score": 0.0, "avg_matches": 0.0}
    top_vals = sorted(per_q_best, reverse=True)[:top_m]
    seq_score = float(sum(top_vals) / len(top_vals))
    avg_matches = float(sum(per_q_matches) / len(per_q_matches))
    return {"score": seq_score, "avg_matches": avg_matches}

def compute_average_precision(relevance: List[bool]) -> float:
    num_rel = sum(relevance)
    if num_rel == 0:
        return 0.0
    hit = 0
    precisions = []
    for idx, is_rel in enumerate(relevance, start=1):
        if is_rel:
            hit += 1
            precisions.append(hit / idx)
    return float(sum(precisions) / num_rel)

def compute_retrieval_metrics(
    query_records: List[Dict[str, object]],
    n_gallery: int,
    threshold: float = None,
) -> Dict[str, float]:
    correct_top1 = 0
    correct_top5 = 0
    ap_sum = 0.0
    for rec in query_records:
        gt_id = rec["gt_id"]
        scores = rec["scores"]
        if threshold is not None:
            scores = [s for s in scores if s["score"] >= threshold]
        top1_id = scores[0]["lynx_id"] if scores else None
        top5_ids = [s["lynx_id"] for s in scores[:5]]
        correct_top1 += int(top1_id == gt_id)
        correct_top5 += int(gt_id in top5_ids)
        relevance = [s["lynx_id"] == gt_id for s in scores]
        ap_sum += compute_average_precision(relevance)
    n_queries = max(1, len(query_records))
    return {
        "top1_acc": correct_top1 / n_queries,
        "top5_acc": correct_top5 / n_queries,
        "mAP": ap_sum / n_queries,
        "n_queries": len(query_records),
        "n_gallery": n_gallery,
    }

def plot_per_lynx_metrics(per_lynx_id: Dict[str, Dict[str, float]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    lynx_ids = list(per_lynx_id.keys())
    if not lynx_ids:
        return
    top1_vals = [per_lynx_id[l]["top1_acc"] for l in lynx_ids]
    top5_vals = [per_lynx_id[l]["top5_acc"] for l in lynx_ids]
    map_vals = [per_lynx_id[l]["mAP"] for l in lynx_ids]

    x = np.arange(len(lynx_ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(lynx_ids) * 0.6), 5))
    ax.bar(x - width, top1_vals, width, label="top1")
    ax.bar(x, top5_vals, width, label="top5")
    ax.bar(x + width, map_vals, width, label="mAP")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Metrics per lynx_id")
    ax.set_xticks(x)
    ax.set_xticklabels(lynx_ids, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_models(config_path: Path, weights: Path, device: torch.device, top_k: int):
    rdd_conf = None
    model = build_rdd(rdd_conf, weights=str(weights))
    model.to(device)
    model.eval()
    model.top_k = top_k
    model.set_softdetect(top_k=top_k)

    lg_conf = {
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
        "weights": "./weights/RDD_lg-v2.pth",
    }
    lg = LightGlue("rdd", **lg_conf).to(device).eval()
    return model, lg


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
    rdd_model, lg_model = build_models(args.config_path, args.weights, device, args.top_k)

    def feat_cache_path(frame_path: Path) -> Path:
        rel = frame_path.relative_to(args.dataset_root)
        return cache_dir / rel.parent / f"{frame_path.stem}.npz"

    # Extract + cache features
    all_sequences = train_seqs + test_seqs
    for seq in tqdm(all_sequences, desc="Extracting features"):
        sampled = sample_frames(seq.frame_paths, args.frames_per_seq)
        for fp in sampled:
            cp = feat_cache_path(fp)
            if cp.exists():
                continue
            img = load_image(fp, args.resize_max)
            feat = extract_frame(rdd_model, img, device, args.top_k)
            ensure_cache(cp, feat)

    # Load cached features into memory (keeps script simple; acceptable for modest dataset sizes)
    def load_seq_feats(seq: SequenceEntry) -> List[FrameFeat]:
        sampled = sample_frames(seq.frame_paths, args.frames_per_seq)
        return [load_cached_feat(feat_cache_path(fp)) for fp in sampled]

    print("Scoring query sequences against gallery...")
    results = []
    query_records = []

    for q in tqdm(test_seqs, desc="Query sequences"):
        q_feats = load_seq_feats(q)
        scores = []
        for g in train_seqs:
            g_feats = load_seq_feats(g)
            agg = aggregate_sequence_score(lg_model, q_feats, g_feats, device, args.top_m_pool)
            scores.append({"score": agg["score"], "lynx_id": g.lynx_id, "name": g.name})
        scores.sort(key=lambda x: x["score"], reverse=True)

        top1 = scores[0] if scores else None
        top1_id = top1["lynx_id"] if top1 else None
        top5_names = [s["name"] for s in scores[:5]]

        results.append(
            {
                "query": q.name,
                "gt_id": q.lynx_id,
                "top1": top1["name"] if top1 else None,
                "top1_id": top1_id,
                "top1_score": top1["score"] if top1 else None,
                "top5": top5_names,
            }
        )
        query_records.append(
            {
                "query": q.name,
                "gt_id": q.lynx_id,
                "scores": scores,
            }
        )

    metrics = compute_retrieval_metrics(query_records, n_gallery=len(train_seqs))
    metrics_thresholded = compute_retrieval_metrics(
        query_records,
        n_gallery=len(train_seqs),
        threshold=args.matcher_threshold,
    )
    metrics_thresholded["threshold"] = args.matcher_threshold

    per_lynx_id = {}
    per_lynx_id_thresholded = {}
    for lynx_id in sorted({r["gt_id"] for r in query_records}):
        split_recs = [r for r in query_records if r["gt_id"] == lynx_id]
        per_lynx_id[lynx_id] = compute_retrieval_metrics(split_recs, n_gallery=len(train_seqs))
        split_thr = compute_retrieval_metrics(
            split_recs,
            n_gallery=len(train_seqs),
            threshold=args.matcher_threshold,
        )
        split_thr["threshold"] = args.matcher_threshold
        per_lynx_id_thresholded[lynx_id] = split_thr

    args.dump_report.parent.mkdir(parents=True, exist_ok=True)
    with open(f'{args.dump_report}_{time.strftime("%Y_%m_%d-%H_%M_%S")}.json', "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "metrics_thresholded": metrics_thresholded,
                "metrics_per_lynx_id": per_lynx_id,
                "metrics_per_lynx_id_thresholded": per_lynx_id_thresholded,
                "results": results,
            },
            f,
            indent=2,
        )

    plot_per_lynx_metrics(per_lynx_id, args.metrics_plot)

    print("Done.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
