"""Where Vismatch runs spend their time, from the timing records completed runs already keep.

Every Vismatch probe run writes ``timings.json`` (model setup, feature extraction compute, cache
lookups, matching, batch sizes) and ``metrics.json`` (number of scored candidate pairs). This tool
reads them for every completed Vismatch run under ``--root`` and prints, per matcher, checkpoint
source and GPU: the median matching cost per candidate pair, the extraction cost per image (runs
that extracted features, i.e. had cache misses), the cache lookup cost per hit, the model setup, and
the share of the run spent matching. Nothing is run on a GPU and nothing is written unless
``--output`` names a CSV for the per-run rows.

    python paper/tools/profile_vismatch.py --root experiments/probe [--dataset SalamanderID2025] [--output reports/vismatch_profile.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROW_FIELDS = [
    "run_dir",
    "dataset",
    "matcher",
    "checkpoint_source",
    "device",
    "candidate_k",
    "num_query",
    "num_database",
    "pairs",
    "model_setup_sec",
    "extract_compute_sec",
    "cache_misses",
    "cache_hits",
    "cache_lookup_sec",
    "match_sec",
    "match_batches",
    "effective_match_batch_size",
    "total_run_sec",
    "match_ms_per_pair",
    "extract_ms_per_image",
    "lookup_ms_per_hit",
    "match_share",
]


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _ratio(numerator: Optional[float], denominator: Optional[float], scale: float = 1.0) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return scale * numerator / denominator


def run_row(run_dir: Path) -> Optional[Dict[str, Any]]:
    """The cost record of one completed Vismatch run, or None for any other run."""
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("method") != "vismatch" or manifest.get("status") != "completed":
        return None
    pairs = _float(metrics.get("num_candidate_pairs"))
    match_sec = _float(timings.get("vismatch_rerank_sec", timings.get("matcher_runtime_sec")))
    extract_sec = _float(timings.get("feature_extraction_compute_sec"))
    misses = _float(timings.get("feature_cache_misses"))
    hits = _float(timings.get("feature_cache_hits"))
    lookup = _float(timings.get("feature_cache_lookup_sec"))
    total = _float(timings.get("total_run_sec"))
    return {
        "run_dir": str(run_dir),
        "dataset": manifest.get("animal", ""),
        "matcher": manifest.get("variant", ""),
        "checkpoint_source": manifest.get("checkpoint_source", ""),
        "device": manifest.get("vismatch_device") or "unknown",
        "candidate_k": _float(timings.get("vismatch_candidate_k", timings.get("benchmark_candidate_k"))),
        "num_query": manifest.get("num_query"),
        "num_database": manifest.get("num_database"),
        "pairs": pairs,
        "model_setup_sec": _float(timings.get("vismatch_model_build_sec", timings.get("model_setup_sec"))),
        "extract_compute_sec": extract_sec,
        "cache_misses": misses,
        "cache_hits": hits,
        "cache_lookup_sec": lookup,
        "match_sec": match_sec,
        "match_batches": _float(timings.get("vismatch_match_batches")),
        "effective_match_batch_size": _float(timings.get("vismatch_effective_match_batch_size")),
        "total_run_sec": total,
        "match_ms_per_pair": _ratio(match_sec, pairs, 1000.0),
        "extract_ms_per_image": _ratio(extract_sec, misses, 1000.0),
        "lookup_ms_per_hit": _ratio(lookup, hits, 1000.0),
        "match_share": _ratio(match_sec, total),
    }


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for current, _, files in os.walk(root, followlinks=True):
        if "timings.json" in files and "run_manifest.json" in files:
            yield Path(current)


def collect(root: Path, dataset: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = [row for row in (run_row(path) for path in sorted(iter_run_dirs(root))) if row is not None]
    return [row for row in rows if dataset is None or row["dataset"] == dataset]


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Medians per (matcher, checkpoint source, device)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["matcher"], row["checkpoint_source"], row["device"])].append(row)
    summary = []
    for (matcher, source, device), members in sorted(groups.items()):
        summary.append(
            {
                "matcher": matcher,
                "checkpoint_source": source,
                "device": device,
                "runs": len(members),
                "match_ms_per_pair": _median(row["match_ms_per_pair"] for row in members),
                "extract_ms_per_image": _median(row["extract_ms_per_image"] for row in members),
                "extracting_runs": sum(1 for row in members if row["extract_ms_per_image"] is not None),
                "lookup_ms_per_hit": _median(row["lookup_ms_per_hit"] for row in members),
                "model_setup_sec": _median(row["model_setup_sec"] for row in members),
                "match_share": _median(row["match_share"] for row in members),
            }
        )
    return summary


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("experiments/probe"))
    parser.add_argument("--dataset", default=None, help="animal name, e.g. SalamanderID2025")
    parser.add_argument("--output", type=Path, default=None, help="write the per-run rows (CSV)")
    args = parser.parse_args(argv)
    rows = collect(args.root, args.dataset)
    print(f"{len(rows)} completed Vismatch runs under {args.root}")
    header = (
        f"{'matcher':<15} {'checkpoint':<10} {'device':<36} {'runs':>5} {'ms/pair':>8} "
        f"{'ms/image':>9} {'(n)':>5} {'ms/hit':>7} {'setup s':>8} {'match %':>8}"
    )
    print(header)
    for item in summarize(rows):
        share = None if item["match_share"] is None else 100.0 * item["match_share"]
        print(
            f"{item['matcher']:<15} {item['checkpoint_source']:<10} {item['device'][:36]:<36} {item['runs']:>5} "
            f"{_fmt(item['match_ms_per_pair']):>8} {_fmt(item['extract_ms_per_image'], 1):>9} "
            f"{item['extracting_runs']:>5} {_fmt(item['lookup_ms_per_hit']):>7} "
            f"{_fmt(item['model_setup_sec'], 1):>8} {_fmt(share, 1):>8}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-run rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
