#!/usr/bin/env python3
"""Export the project-page score-separation view: histograms of LoMa image scores for
same-individual and different-individual query/candidate pairs, default versus fine-tuned,
for every paper dataset at the paper's main budget (k=250), plus per-query margins.

Sources: the paper repository's ``results/<stem>_ablation.csv`` snapshot names the exact
default and matcher-only fine-tuned LoMa runs behind the manuscript (``run_id`` and
``manifest_path``); each run's ``scores.npz`` holds the finite scores of the 250
MegaDescriptor-L candidates per query, and its ``config.snapshot.yaml`` names the metadata
CSV whose ``split_col`` filters give the query and database identity order (the same
filter ``load_dataset_splits`` applies). Both matchers rank the identical shortlist, so the
two histograms describe the same pairs.

Population: the scored pairs are the shortlist pairs, so "different individual" means the
hard candidates the matcher actually ranks, not random pairs. The per-query margin is the
query's best same-individual score minus its best different-individual score, over queries
whose shortlist holds at least one same-individual candidate; a positive margin is what the
triplet objective asks for.

Fail-closed checks: the manifest must be completed with the recorded ``run_id``, the score
matrix shape must equal the split sizes, and the Top-1 recomputed from ``scores.npz`` with
the shared stable rule must equal the run's ``metrics.json`` ``top_1`` (which also proves
the identity order is right). CPU only; needs pandas, numpy and PyYAML.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_REPO = REPO_ROOT.parent / "ECIR-Animal-ReID-Paper"
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "score_separation"
MAIN_K = 250
BINS = 50
MARGIN_BINS = 50
MATCHERS = {"default": ("default", "Default LoMa"), "finetuned": ("fine-tuned", "LoMa + WildMatch")}
# Same order and labels as paper/page/export_project_page_data.py.
DATASETS: List[Tuple[str, str, str]] = [
    ("CzechLynx_split-time_closed", "czechlynx", "CzechLynx"),
    ("HyenaID2022_split", "hyena", "Hyena"),
    ("LeopardID2022_split", "leopard", "Leopard"),
    ("NyalaData_split", "nyala", "Nyala"),
    ("SalamanderID2025_split", "salamander", "Salamander"),
    ("SeaStarReID2023_split", "sea_star", "Sea star"),
    ("WhaleSharkID_split", "whale_shark", "Whale shark"),
    ("ZindiTurtleRecall_split", "turtle", "Turtle"),
]


# --------------------------------------------------------------------------- statistics
def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a random same-individual score exceeds a random different-individual
    score, ties counted half (Mann-Whitney U with average ranks)."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative]).astype(np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    # average ranks for ties
    boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(values)]])
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2.0  # 1-based average rank
    rank_sum = ranks[: len(positive)].sum()
    n_pos, n_neg = len(positive), len(negative)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts per bin; values at the upper edge fall into the last bin."""
    clipped = np.clip(values, edges[0], np.nextafter(edges[-1], edges[0]))
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(np.int64)


def overlap_coefficient(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """Shared area of two histograms after normalising each to unit mass (1 = identical)."""
    if counts_a.sum() == 0 or counts_b.sum() == 0:
        return float("nan")
    return float(np.minimum(counts_a / counts_a.sum(), counts_b / counts_b.sum()).sum())


def stable_top1(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_query: int) -> np.ndarray:
    """Per query, the candidate with the highest score, lower database index first on ties
    (the shared ranking rule); -1 for queries without scored candidates."""
    order = np.lexsort((cols, -values.astype(np.float64), rows))  # row asc, value desc, col asc
    first = np.ones(len(order), dtype=bool)
    first[1:] = rows[order][1:] != rows[order][:-1]
    top = np.full(n_query, -1, dtype=np.int64)
    top[rows[order][first]] = cols[order][first]
    return top


def per_query_margins(rows: np.ndarray, values: np.ndarray, same: np.ndarray, n_query: int) -> np.ndarray:
    """Best same-individual score minus best different-individual score per query; NaN when
    a query has no same-individual candidate (or no different-individual candidate)."""
    best_same = np.full(n_query, -np.inf)
    best_diff = np.full(n_query, -np.inf)
    np.maximum.at(best_same, rows[same], values[same].astype(np.float64))
    np.maximum.at(best_diff, rows[~same], values[~same].astype(np.float64))
    margin = best_same - best_diff
    margin[~np.isfinite(best_same) | ~np.isfinite(best_diff)] = np.nan
    return margin


# --------------------------------------------------------------------------- sources
def paper_runs(paper_repo: Path, stem: str) -> Dict[str, Dict[str, Any]]:
    """The paper's default and fine-tuned LoMa rows at k=250 for one dataset stem."""
    table = pd.read_csv(paper_repo / "results" / f"{stem}_ablation.csv")
    runs: Dict[str, Dict[str, Any]] = {}
    for key, (label, _) in MATCHERS.items():
        rows = table[(table["method"].str.lower() == "vismatch") & (table["matcher"].str.lower() == "loma")
                     & (table["checkpoint"].str.lower() == label) & (table["candidate_k"].astype(float) == MAIN_K)]
        if len(rows) != 1:
            raise ValueError(f"{stem}: expected one {label} LoMa row at k={MAIN_K}, found {len(rows)}")
        row = rows.iloc[0]
        runs[key] = {"run_id": str(row["run_id"]), "manifest_path": str(row["manifest_path"]),
                     "top_1": float(row["top_1"]), "top_5": float(row["top_5"]),
                     "balanced_top_1": float(row["balanced_top_1"])}
    return runs


def split_labels(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    cfg = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))["dataset"]
    metadata = pd.read_csv(Path(cfg["root"]) / cfg["metadata_file"])
    split = metadata[cfg["split_col"]]
    database = metadata.loc[split == cfg["database_split_value"], cfg["label_col"]].astype(str).to_numpy()
    query = metadata.loc[split == cfg["query_split_value"], cfg["label_col"]].astype(str).to_numpy()
    return query, database


def load_run(run_dir: Path, expected_run_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("run_id") != expected_run_id:
        raise ValueError(f"{run_dir}: manifest status/run_id {manifest.get('status')}/{manifest.get('run_id')} "
                         f"does not match the paper's {expected_run_id}")
    with np.load(run_dir / "scores.npz") as data:
        return data["rows"].astype(np.int64), data["cols"].astype(np.int64), data["values"].astype(np.float32), {
            "shape": tuple(int(x) for x in data["shape"]), "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))}


def summarise(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, same: np.ndarray, n_query: int,
              edges: np.ndarray, margin_edges: np.ndarray) -> Dict[str, Any]:
    pos, neg = values[same], values[~same]
    counts_same, counts_diff = histogram(pos, edges), histogram(neg, edges)
    margin = per_query_margins(rows, values, same, n_query)
    finite = margin[np.isfinite(margin)]
    return {
        "same": counts_same.tolist(), "different": counts_diff.tolist(),
        "margin": histogram(finite, margin_edges).tolist(),
        "stats": {
            "n_same": int(len(pos)), "n_different": int(len(neg)),
            "median_same": float(np.median(pos)) if len(pos) else None,
            "median_different": float(np.median(neg)) if len(neg) else None,
            "mean_same": float(pos.mean()) if len(pos) else None,
            "mean_different": float(neg.mean()) if len(neg) else None,
            "auroc": auroc(pos, neg), "overlap": overlap_coefficient(counts_same, counts_diff),
            "n_queries_with_margin": int(len(finite)),
            "median_margin": float(np.median(finite)) if len(finite) else None,
            "positive_margin_fraction": float((finite > 0).mean()) if len(finite) else None,
        },
    }


def export_dataset(stem: str, key: str, label: str, paper_repo: Path, edges: np.ndarray,
                   margin_edges: np.ndarray) -> Dict[str, Any]:
    runs = paper_runs(paper_repo, stem)
    out: Dict[str, Any] = {"key": key, "label": label, "stem": stem, "k": MAIN_K, "runs": {}, "matchers": {}}
    labels: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for mkey, run in runs.items():
        run_dir = REPO_ROOT / Path(run["manifest_path"]).parent
        rows, cols, values, info = load_run(run_dir, run["run_id"])
        if labels is None:
            labels = split_labels(run_dir)
        q_labels, db_labels = labels
        n_query, n_db = len(q_labels), len(db_labels)
        if info["shape"] != (n_query, n_db):
            raise ValueError(f"{run_dir}: scores.npz shape {info['shape']} != split sizes {(n_query, n_db)}")
        if values.max() > 1.0 or values.min() < 0.0:
            raise ValueError(f"{run_dir}: scores outside [0, 1]: {values.min()}..{values.max()}")
        same = q_labels[rows] == db_labels[cols]
        top = stable_top1(rows, cols, values, n_query)
        top1 = float(np.mean((top >= 0) & (db_labels[np.clip(top, 0, None)] == q_labels)))
        recorded = float(info["metrics"]["top_1"])
        if abs(top1 - recorded) > 1e-9:
            raise ValueError(f"{run_dir}: recomputed Top-1 {top1:.6f} != recorded {recorded:.6f}; identity order unverified")
        if abs(recorded - run["top_1"]) > 1e-9:
            raise ValueError(f"{run_dir}: metrics.json Top-1 {recorded:.6f} != paper CSV {run['top_1']:.6f}")
        out["runs"][mkey] = {**run, "top_1_recomputed": top1, "n_pairs": int(len(values))}
        out["matchers"][mkey] = summarise(rows, cols, values, same, n_query, edges, margin_edges)
        out["n_queries"], out["n_database"] = n_query, n_db
        out["n_queries_with_same_candidate"] = int(len(np.unique(rows[same])))
    return out


def build_payload(datasets: List[Dict[str, Any]], edges: np.ndarray, margin_edges: np.ndarray,
                  paper_commit: Optional[str]) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_score_separation.py",
        "paper_commit": paper_commit,
        "k": MAIN_K,
        "matchers": [{"key": key, "label": label} for key, (_, label) in MATCHERS.items()],
        "bin_edges": [round(float(x), 6) for x in edges],
        "margin_bin_edges": [round(float(x), 6) for x in margin_edges],
        "score_definition": "LoMa image score: summed confidence of the mutual-nearest matches above the "
                            "threshold, divided by the smaller keypoint count; computed on background-removed inputs.",
        "population": f"all query/candidate pairs of the {MAIN_K} MegaDescriptor-L candidates per query (the pairs the "
                      "matcher ranks); 'different individual' therefore means hard shortlist candidates, not random pairs. "
                      "Both matchers rank the identical shortlist.",
        "margin_definition": "per query: best same-individual score minus best different-individual score, over "
                             "queries whose shortlist holds at least one same-individual candidate.",
        "datasets": datasets,
    }


def paper_commit(paper_repo: Path) -> Optional[str]:
    head = paper_repo / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref:"):
            return (paper_repo / ".git" / ref.split(" ", 1)[1]).read_text(encoding="utf-8").strip()[:12]
        return ref[:12]
    except OSError:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper-repo", type=Path, default=DEFAULT_PAPER_REPO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="*", default=None, help="page keys to export (default: all eight)")
    args = parser.parse_args(argv)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    margin_edges = np.linspace(-1.0, 1.0, MARGIN_BINS + 1)
    selected = [d for d in DATASETS if args.datasets is None or d[1] in args.datasets]
    datasets = []
    for stem, key, label in selected:
        result = export_dataset(stem, key, label, args.paper_repo, edges, margin_edges)
        stats = {m: result["matchers"][m]["stats"] for m in result["matchers"]}
        print(f"[score-separation] {label}: AUROC default {stats['default']['auroc']:.3f} -> fine-tuned "
              f"{stats['finetuned']['auroc']:.3f}; overlap {stats['default']['overlap']:.3f} -> "
              f"{stats['finetuned']['overlap']:.3f}; positive margin {stats['default']['positive_margin_fraction']:.3f} "
              f"-> {stats['finetuned']['positive_margin_fraction']:.3f}", flush=True)
        datasets.append(result)
    payload = build_payload(datasets, edges, margin_edges, paper_commit(args.paper_repo))
    text = json.dumps(payload, indent=1)
    for fragment in ("/shared/", "/home/"):
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in export")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "score_separation.json").write_text(text + "\n", encoding="utf-8")
    print(f"[score-separation] wrote {args.out_dir / 'score_separation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
