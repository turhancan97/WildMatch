#!/usr/bin/env python3
"""Export the project-page "rare and common individuals" view: retrieval accuracy by how
many gallery images the query's individual has, default versus fine-tuned LoMa, for every
paper dataset at the paper's main budget (k=250).

Sources: the same k=250 default and fine-tuned LoMa runs as the score-separation view
(pinned by the paper repository's results snapshot), each run's ``scores.npz`` (scores of the
250 MegaDescriptor-L candidates per query) and the identity order from the run's
``config.snapshot.yaml`` metadata filter. Gallery counts come from the database identities
of that same metadata.

Per query and matcher: Top-1 and Top-5 correctness under the shared stable ranking rule and
whether the shortlist contains the individual at all (identical for both matchers). Queries
are binned by the number of gallery images of their individual with fixed edges shared by
every dataset (1, 2-4, 5-9, 10-29, 30+); queries whose individual has no gallery image are
counted separately because no matcher can retrieve them. Each bin reports Top-1, Top-5 and
shortlist share per matcher and the fine-tuning gain with a paired bootstrap 95 % interval
(resampling queries within the bin, seeded).

Fail-closed checks are inherited from the score-separation exporter (completed manifest with
the paper's run id, matrix shape, scores in [0, 1], recomputed Top-1 equal to the recorded
value). CPU only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper.page import export_score_separation as sep  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "data" / "frequency_bins.json"
BIN_EDGES = [1, 2, 5, 10, 30]  # lower edges; the last bin is open
BIN_LABELS = ["1", "2–4", "5–9", "10–29", "30+"]
SMALL_BIN = 20
BOOTSTRAP = 2000
SEED = 0
MATCHERS = sep.MATCHERS
DATASETS = sep.DATASETS


def bin_index(gallery_count: np.ndarray, edges: Sequence[int] = BIN_EDGES) -> np.ndarray:
    """Bin index per query (0..len(edges)-1); -1 for queries whose individual has no gallery image."""
    idx = np.searchsorted(np.asarray(edges), gallery_count, side="right") - 1
    idx[gallery_count < edges[0]] = -1
    return idx


def stable_topn(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_query: int, n: int) -> np.ndarray:
    """Per query, the first ``n`` candidates in the shared stable order (score desc, index asc); -1 padded."""
    order = np.lexsort((cols, -values.astype(np.float64), rows))
    sorted_rows, sorted_cols = rows[order], cols[order]
    out = np.full((n_query, n), -1, dtype=np.int64)
    starts = np.searchsorted(sorted_rows, np.arange(n_query), side="left")
    ends = np.searchsorted(sorted_rows, np.arange(n_query), side="right")
    for q in range(n_query):
        take = sorted_cols[starts[q]:min(ends[q], starts[q] + n)]
        out[q, : len(take)] = take
    return out


def correctness(topn: np.ndarray, db_labels: np.ndarray, q_labels: np.ndarray) -> np.ndarray:
    """Boolean per query: any of the listed candidates has the query's identity."""
    valid = topn >= 0
    labels = db_labels[np.clip(topn, 0, None)]
    return np.any(valid & (labels == q_labels[:, None]), axis=1)


def bootstrap_gain(default: np.ndarray, finetuned: np.ndarray, n_resamples: int = BOOTSTRAP, seed: int = SEED) -> Tuple[float, float]:
    """Paired bootstrap 95 % interval of mean(finetuned) - mean(default) over queries."""
    n = len(default)
    if n == 0:
        return float("nan"), float("nan")
    diff = finetuned.astype(np.float64) - default.astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_bin(mask: np.ndarray, hit: Dict[str, Dict[str, np.ndarray]], in_shortlist: np.ndarray) -> Dict[str, Any]:
    n = int(mask.sum())
    out: Dict[str, Any] = {"n": n, "small": n < SMALL_BIN,
                           "shortlist_share": float(in_shortlist[mask].mean()) if n else None}
    for metric in ("top_1", "top_5"):
        d, f = hit["default"][metric][mask], hit["finetuned"][metric][mask]
        lo, hi = bootstrap_gain(d, f)
        out[metric] = {"default": float(d.mean()) if n else None, "finetuned": float(f.mean()) if n else None,
                       "gain": float(f.mean() - d.mean()) if n else None, "gain_ci95": [lo, hi] if n else None}
    return out


def export_dataset(stem: str, key: str, label: str, paper_repo: Path) -> Dict[str, Any]:
    runs = sep.paper_runs(paper_repo, stem)
    labels: Optional[Tuple[np.ndarray, np.ndarray]] = None
    hit: Dict[str, Dict[str, np.ndarray]] = {}
    in_shortlist: Optional[np.ndarray] = None
    for mkey, run in runs.items():
        run_dir = REPO_ROOT / Path(run["manifest_path"]).parent
        rows, cols, values, info = sep.load_run(run_dir, run["run_id"])
        if labels is None:
            labels = sep.split_labels(run_dir)
        q_labels, db_labels = labels
        n_query, n_db = len(q_labels), len(db_labels)
        if info["shape"] != (n_query, n_db):
            raise ValueError(f"{run_dir}: scores.npz shape {info['shape']} != split sizes {(n_query, n_db)}")
        top5 = stable_topn(rows, cols, values, n_query, 5)
        top1 = correctness(top5[:, :1], db_labels, q_labels)
        recorded = float(info["metrics"]["top_1"])
        if abs(top1.mean() - recorded) > 1e-9:
            raise ValueError(f"{run_dir}: recomputed Top-1 {top1.mean():.6f} != recorded {recorded:.6f}")
        hit[mkey] = {"top_1": top1, "top_5": correctness(top5, db_labels, q_labels)}
        same = q_labels[rows] == db_labels[cols]
        shortlist = np.zeros(n_query, dtype=bool)
        shortlist[rows[same]] = True
        if in_shortlist is None:
            in_shortlist = shortlist
        elif not np.array_equal(in_shortlist, shortlist):
            raise ValueError(f"{stem}: the two runs do not share a shortlist")
    assert labels is not None and in_shortlist is not None
    q_labels, db_labels = labels
    _, counts = np.unique(db_labels, return_counts=True)
    count_of = dict(zip(np.unique(db_labels), counts))
    gallery = np.array([count_of.get(ident, 0) for ident in q_labels], dtype=np.int64)
    bins = bin_index(gallery)
    result: Dict[str, Any] = {
        "key": key, "label": label, "stem": stem, "k": sep.MAIN_K,
        "runs": {m: {"run_id": r["run_id"], "top_1": r["top_1"], "top_5": r["top_5"]} for m, r in runs.items()},
        "n_queries": int(len(q_labels)), "n_database": int(len(db_labels)),
        "n_identities_database": int(len(count_of)),
        "n_queries_without_gallery_image": int((bins < 0).sum()),
        "bins": [], "overall": summarise_bin(bins >= 0, hit, in_shortlist),
    }
    for i, name in enumerate(BIN_LABELS):
        entry = summarise_bin(bins == i, hit, in_shortlist)
        entry["label"] = name
        entry["lower"] = BIN_EDGES[i]
        entry["upper"] = (BIN_EDGES[i + 1] - 1) if i + 1 < len(BIN_EDGES) else None
        entry["n_identities"] = int(len(np.unique(q_labels[bins == i])))
        result["bins"].append(entry)
    return result


def build_payload(datasets: List[Dict[str, Any]], paper_commit: Optional[str]) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_frequency_bins.py",
        "paper_commit": paper_commit,
        "k": sep.MAIN_K,
        "matchers": [{"key": k, "label": label} for k, (_, label) in MATCHERS.items()],
        "bin_labels": BIN_LABELS, "bin_edges": BIN_EDGES, "small_bin": SMALL_BIN,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "interval": "paired percentile 95 %"},
        "definitions": {
            "gallery_images": "number of database (gallery) images of the query's individual.",
            "top_1": "query's individual is the top-ranked candidate (shared stable rule).",
            "top_5": "query's individual appears among the top 5 candidates.",
            "shortlist_share": "share of the bin's queries whose individual is among the k MegaDescriptor-L "
                               "candidates; identical for both matchers and a ceiling for Top-1 and Top-5.",
            "gain": "fine-tuned minus default accuracy in the bin; the interval is a paired bootstrap over the "
                    "bin's queries.",
            "without_gallery_image": "queries whose individual has no gallery image cannot be retrieved and are "
                                     "excluded from the bins (counted per dataset).",
        },
        "datasets": datasets,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-repo", type=Path, default=sep.DEFAULT_PAPER_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args(argv)
    selected = [d for d in DATASETS if args.datasets is None or d[1] in args.datasets]
    datasets = []
    for stem, key, label in selected:
        result = export_dataset(stem, key, label, args.paper_repo)
        gains = " ".join(f"{b['label']}:{b['top_1']['gain'] * 100:+.1f}(n={b['n']})" if b["n"] else f"{b['label']}:–"
                         for b in result["bins"])
        print(f"[frequency] {label}: Top-1 gain by gallery count {gains}; "
              f"{result['n_queries_without_gallery_image']} queries without gallery image", flush=True)
        datasets.append(result)
    payload = build_payload(datasets, sep.paper_commit(args.paper_repo))
    text = json.dumps(payload, indent=1)
    for fragment in ("/shared/", "/home/"):
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in export")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"[frequency] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
