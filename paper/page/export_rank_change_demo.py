#!/usr/bin/env python3
"""Export the project-page rank-change explorer: for sampled CzechLynx test queries, the
top 5 under cosine retrieval, default LoMa and LoMa + WildMatch, with the true individual
marked, plus whole-split counts of rescued, still-wrong, regressed and already-right queries.

Sources (the paper's CzechLynx closed-split runs at k=250):
* cosine retrieval: MegaDescriptor-L embeddings re-read from the probe's own feature cache
  through the probe's extraction code and the cosine run's ``config.snapshot.yaml`` (the
  cosine run did not persist its dense 11,924 x 27,836 score matrix); similarity is the
  probe's normalised dot product.
* default and fine-tuned LoMa: each run's ``scores.npz`` (finite scores of the 250
  MegaDescriptor-L candidates per query). A true individual outside the shortlist has no
  rank (``null``), which the page states.

Ranking uses the shared stable rule (descending score, lower gallery index first). The
true rank is the first position whose gallery photo has the query's identity. Categories
compare rank-1 correctness: ``rescued`` = fine-tuned right, default wrong; ``still_wrong``
= fine-tuned wrong; ``regressed`` = default right, fine-tuned wrong; ``already_right`` =
both right. Counts are over every query; the exported examples are a random sample per
category (seeded), never chosen by appearance. Photos are the raw CzechLynx frames.
Needs the ex-reid environment; a GPU is only needed if the feature cache misses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

PROBE_ROOT = REPO_ROOT / "experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/megadescriptor-l"
RUNS = {
    "cosine": PROBE_ROOT / "cosine/default/20260919T223430Z_5463f66e",
    "default": PROBE_ROOT / "vismatch/loma/20260920T131759Z_daf95fda",
    "finetuned": PROBE_ROOT / "vismatch/loma/20260920T131759Z_612b4791",
}
METHODS = [{"key": "cosine", "label": "Cosine retrieval (MegaDescriptor-L)"},
           {"key": "default", "label": "Default LoMa"},
           {"key": "finetuned", "label": "LoMa + WildMatch"}]
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "rank_change"
WEB_LONG_SIDE = 320
SAMPLE = {"rescued": 8, "still_wrong": 8, "regressed": 4, "already_right": 4}
ATTRIBUTION = "Photographs from CzechLynx (Picek et al.), time-closed split."


def stable_top(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores, ties broken by lower index (the shared rule)."""
    from wildmatch.evaluate.ranking import stable_rank_1d
    return stable_rank_1d(scores)[:n]


def true_rank(order: np.ndarray, labels: np.ndarray, identity: str) -> Optional[int]:
    hits = np.flatnonzero(labels[order] == identity)
    return int(hits[0]) + 1 if len(hits) else None


def categorize(cos_ok: bool, def_ok: bool, ft_ok: bool) -> str:
    if ft_ok and not def_ok:
        return "rescued"
    if def_ok and not ft_ok:
        return "regressed"
    if ft_ok and def_ok:
        return "already_right"
    return "still_wrong"


def shortlist_rankings(run_dir: Path, n_query: int, n_db: int) -> List[np.ndarray]:
    """Per query, the shortlisted gallery indices in stable score order (full order within the shortlist)."""
    with np.load(run_dir / "scores.npz") as data:
        shape = tuple(int(v) for v in data["shape"])
        if shape != (n_query, n_db):
            raise ValueError(f"{run_dir}: scores.npz shape {shape} does not match the split ({n_query}, {n_db})")
        rows, cols, values = data["rows"], data["cols"], data["values"].astype(np.float64)
    order = np.lexsort((cols, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    starts = np.searchsorted(rows, np.arange(n_query + 1))
    out: List[np.ndarray] = []
    for q in range(n_query):
        c, v = cols[starts[q]:starts[q + 1]], values[starts[q]:starts[q + 1]]
        if len(c) == 0:
            out.append(np.empty(0, dtype=np.int64)); continue
        rank = np.lexsort((c, -v))        # descending score, then lower gallery index
        out.append(c[rank])
    return out


def cosine_rankings(cfg_path: Path, device: str):
    """Query/database datasets and the full stable cosine order per query, via the probe's code and cache."""
    import torch
    from omegaconf import OmegaConf
    from wildmatch.evaluate import probe_runner as pr
    from wildmatch.utils.fingerprints import file_digest_cache

    cfg = OmegaConf.load(cfg_path)
    dataset, dataset_database, dataset_query = pr.load_dataset_splits(cfg)
    model, _, mean, std, img_size, _, _, checkpoint_path = pr.load_backbone(cfg)
    _, transform_model, _ = pr.build_transforms(mean, std, img_size)
    view_db = pr.make_dataset_view(cfg, dataset_database, transform_model)
    view_q = pr.make_dataset_view(cfg, dataset_query, transform_model)
    dev = torch.device("cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    model = model.to(dev).eval()
    cache = pr.FeatureCache(enabled=bool(cfg.benchmark.cache.enabled), cache_dir=Path(cfg.benchmark.cache.dir),
                            fmt=str(cfg.benchmark.cache.format))
    with file_digest_cache():
        feats_db = pr.extract_deep_features_with_cache(view_db, "database", model, dev, int(cfg.model.batch_size),
                                                       int(cfg.model.num_workers), cache, cfg, "cosine", checkpoint_path)
        feats_q = pr.extract_deep_features_with_cache(view_q, "query", model, dev, int(cfg.model.batch_size),
                                                      int(cfg.model.num_workers), cache, cfg, "cosine", checkpoint_path)
    print(f"[rank-change] cosine features: cache hits {cache.hits}, misses {cache.misses}", flush=True)
    sim = pr._cosine_similarity_matrix(np.asarray(feats_q), np.asarray(feats_db))
    return dataset_query, dataset_database, sim


def photo_tag(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


def build_payload(queries: List[Dict[str, Any]], photos: Dict[str, Any], counts: Dict[str, int], k: int) -> Dict[str, Any]:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_rank_change_demo.py",
        "dataset": "CzechLynx closed", "k": k, "attribution": ATTRIBUTION,
        "methods": METHODS, "runs": {key: path.name for key, path in RUNS.items()},
        "counts": counts, "sample": SAMPLE, "queries": queries, "photos": photos,
    }
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"demo payload contains a private path fragment {needle!r}")
    return payload


def export(out_dir: Path, device: str, seed: int, sample: Dict[str, int] = SAMPLE) -> Dict[str, Any]:
    dataset_query, dataset_database, sim = cosine_rankings(RUNS["cosine"] / "config.snapshot.yaml", device)
    q_meta, db_meta = dataset_query.metadata.reset_index(drop=True), dataset_database.metadata.reset_index(drop=True)
    q_labels = q_meta["unique_name"].astype(str).to_numpy(); db_labels = db_meta["unique_name"].astype(str).to_numpy()
    n_q, n_db = len(q_labels), len(db_labels)
    k = 250
    orders = {"default": shortlist_rankings(RUNS["default"], n_q, n_db),
              "finetuned": shortlist_rankings(RUNS["finetuned"], n_q, n_db)}
    with np.load(RUNS["default"] / "scores.npz") as d0, np.load(RUNS["finetuned"] / "scores.npz") as d1:
        score_lookup = {"default": dict(zip(zip(d0["rows"].tolist(), d0["cols"].tolist()), d0["values"].tolist())),
                        "finetuned": dict(zip(zip(d1["rows"].tolist(), d1["cols"].tolist()), d1["values"].tolist()))}

    per_query: List[Dict[str, Any]] = []
    counts = {"total": n_q, "cosine_top1": 0, "default_top1": 0, "finetuned_top1": 0,
              "rescued": 0, "still_wrong": 0, "regressed": 0, "already_right": 0}
    for q in range(n_q):
        ident = q_labels[q]
        cos_order = np.argsort(-sim[q], kind="stable")
        ranks = {"cosine": true_rank(cos_order, db_labels, ident),
                 "default": true_rank(orders["default"][q], db_labels, ident),
                 "finetuned": true_rank(orders["finetuned"][q], db_labels, ident)}
        ok = {m: ranks[m] == 1 for m in ranks}
        for m in ranks:
            counts[f"{m}_top1"] += int(ok[m])
        cat = categorize(ok["cosine"], ok["default"], ok["finetuned"])
        counts[cat] += 1
        per_query.append({"q": q, "identity": ident, "category": cat, "ranks": ranks, "cos_order": cos_order[:5]})
    print(f"[rank-change] counts: {counts}", flush=True)

    rng = random.Random(seed)
    chosen: List[Dict[str, Any]] = []
    for cat, n in sample.items():
        pool = [p for p in per_query if p["category"] == cat]
        rng.shuffle(pool)
        chosen.extend(pool[:n])

    out_dir.mkdir(parents=True, exist_ok=True)
    photos: Dict[str, Any] = {}
    root = Path(dataset_query.root)

    def add_photo(rel: str) -> str:
        tag = photo_tag(rel)
        if tag not in photos:
            image = Image.open(root / rel).convert("RGB")
            scale = min(1.0, WEB_LONG_SIDE / max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            (image.resize(size, Image.LANCZOS) if scale < 1.0 else image).save(out_dir / f"{tag}.jpg", format="JPEG",
                                                                              quality=82, optimize=True, progressive=True)
            photos[tag] = {"file": f"{tag}.jpg", "width": size[0], "height": size[1]}
        return tag

    queries: List[Dict[str, Any]] = []
    for p in chosen:
        q = p["q"]
        rankings: Dict[str, Any] = {}
        for m in ("cosine", "default", "finetuned"):
            top = p["cos_order"] if m == "cosine" else orders[m][q][:5]
            entries = []
            for g in top:
                g = int(g)
                score = float(sim[q, g]) if m == "cosine" else float(score_lookup[m].get((q, g), float("nan")))
                entries.append({"tag": add_photo(str(db_meta["path"].iloc[g])), "identity": db_labels[g],
                                "score": round(score, 4), "correct": bool(db_labels[g] == p["identity"])})
            rankings[m] = {"true_rank": p["ranks"][m], "top5": entries}
        queries.append({"tag": add_photo(str(q_meta["path"].iloc[q])), "identity": p["identity"], "category": p["category"],
                        "rankings": rankings})
        print(f"[rank-change] {p['category']:13s} {p['identity']}: cosine r{p['ranks']['cosine']} default r{p['ranks']['default']} "
              f"fine-tuned r{p['ranks']['finetuned']}", flush=True)
    payload = build_payload(queries, photos, counts, k)
    (out_dir / "rank_change.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[rank-change] wrote {out_dir / 'rank_change.json'} ({len(queries)} queries, {len(photos)} photos)")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    for key, path in RUNS.items():
        if not (path / "run_manifest.json").is_file():
            print(f"error: run {key} not found at {path}", file=sys.stderr)
            return 2
    export(args.out, args.device, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
