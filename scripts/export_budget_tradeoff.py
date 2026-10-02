#!/usr/bin/env python3
"""Export the project-page candidate-budget trade-off: for every paper dataset and every
measured budget k, the share of queries whose individual is inside the k-image shortlist,
Top-5 of the default and the fine-tuned LoMa matcher, and the matching time, normalised to
minutes per 1,000 queries and milliseconds per scored pair.

Sources (all measured, nothing interpolated):
* the paper repository's ``results/<stem>_ablation.csv`` pins the default and fine-tuned
  LoMa run of each budget (``run_id``, ``manifest_path``, ``top_5``);
* each run's ``metrics.json`` gives ``candidate_recall_at_k`` (identity-level: the share of
  queries with at least one same-individual image among the k MegaDescriptor-L candidates;
  it is the same for both matchers because they rank the identical shortlist) and
  ``num_queries``;
* each run's ``timings.json`` gives ``vismatch_rerank_sec``, the Vismatch feature-matching
  timer over all query x candidate pairs (feature extraction, candidate selection, setup
  and cache I/O excluded; identical to ``primary_compute_runtime_sec`` where that exists);
* ``logs/parallel_run`` task records (via ``logs/index.csv``) give each run's Slurm job and,
  through ``sacct`` when available, the partition it ran on. Runs that predate the task
  records have no partition and are reported as "not recorded".

Hardware is part of the output because minutes depend on the GPU: per budget the export
keeps both runs' times and partitions, and ``display`` picks the time to show (mean of the
two when both ran on the same partition, else the run on the dataset's most common
partition). The page states the hardware next to every time. Fails closed on a missing run,
a recall mismatch between the two matchers, or a private path in the output.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_REPO = REPO_ROOT.parent / "ECIR-Animal-ReID-Paper"
DEFAULT_OUT = REPO_ROOT / "docs" / "data" / "budget_tradeoff.json"
LOG_INDEX = REPO_ROOT / "logs" / "index.csv"
BUDGETS = [10, 50, 100, 250, 500, 1000]
MAIN_K = 250
MATCHERS = {"default": ("default", "Default LoMa"), "finetuned": ("fine-tuned", "LoMa + WildMatch")}
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
GPU_LABELS = {"rtx4090_batch": "RTX 4090", "rtx4090": "RTX 4090", "dgxh100": "H100", "dgxa100": "A100",
              "dgx": "V100"}
NOT_RECORDED = "not recorded"


# --------------------------------------------------------------------------- helpers
def gpu_label(partition: Optional[str]) -> str:
    if not partition:
        return NOT_RECORDED
    return GPU_LABELS.get(partition, partition)


def per_pair_ms(matching_sec: float, n_queries: int, k: int) -> float:
    return 1000.0 * matching_sec / (n_queries * k)


def minutes_per_1000(matching_sec: float, n_queries: int) -> float:
    return matching_sec / 60.0 * 1000.0 / n_queries


def choose_display(times: Dict[str, float], partitions: Dict[str, Optional[str]], dominant: Optional[str]) -> Dict[str, Any]:
    """Which matching time to show for a budget shared by both matchers."""
    keys = list(times)
    parts = {k: partitions.get(k) for k in keys}
    if len({parts[k] for k in keys}) == 1:
        return {"matching_sec": sum(times.values()) / len(times), "partition": parts[keys[0]],
                "source": "mean of both runs"}
    preferred = [k for k in keys if parts[k] == dominant] or keys
    chosen = preferred[0]
    return {"matching_sec": times[chosen], "partition": parts[chosen], "source": f"{chosen} run"}


def job_lookup(log_index: Path) -> Dict[str, str]:
    """experiment run directory (repo-relative) -> Slurm job id ``<array>_<task>``."""
    if not log_index.is_file():
        return {}
    lookup: Dict[str, str] = {}
    with log_index.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            run_dir = row.get("experiment_run_directory") or ""
            if run_dir and row.get("job_id"):
                lookup[run_dir] = f"{row['job_id']}_{row['task_id']}"
    return lookup


def sacct_partitions(job_ids: Sequence[str]) -> Dict[str, str]:
    """``sacct`` partition per job id; empty when sacct is unavailable."""
    if not job_ids or shutil.which("sacct") is None:
        return {}
    result = subprocess.run(["sacct", "-j", ",".join(sorted(set(job_ids))), "-X", "--format=JobID,Partition", "-P", "-n"],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}
    partitions: Dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 2 and parts[0]:
            partitions[parts[0]] = parts[1]
    return partitions


# --------------------------------------------------------------------------- collection
def paper_rows(paper_repo: Path, stem: str) -> pd.DataFrame:
    table = pd.read_csv(paper_repo / "results" / f"{stem}_ablation.csv")
    rows = table[(table["method"].str.lower() == "vismatch") & (table["matcher"].str.lower() == "loma")]
    return rows


def collect_dataset(stem: str, key: str, label: str, paper_repo: Path, jobs: Dict[str, str],
                    partitions: Dict[str, str]) -> Dict[str, Any]:
    rows = paper_rows(paper_repo, stem)
    budgets: List[Dict[str, Any]] = []
    run_partitions: List[Optional[str]] = []
    n_queries: Optional[int] = None
    for k in BUDGETS:
        entry: Dict[str, Any] = {"k": k, "runs": {}, "top_5": {}, "matching_sec": {}, "partition": {}}
        recalls = {}
        for mkey, (ck_label, _) in MATCHERS.items():
            sel = rows[(rows["checkpoint"].str.lower() == ck_label) & (rows["candidate_k"].astype(float) == k)]
            if len(sel) != 1:
                raise ValueError(f"{stem}: expected one {ck_label} LoMa row at k={k}, found {len(sel)}")
            row = sel.iloc[0]
            run_dir_rel = str(Path(row["manifest_path"]).parent)
            run_dir = REPO_ROOT / run_dir_rel
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("status") != "completed" or manifest.get("run_id") != str(row["run_id"]):
                raise ValueError(f"{run_dir}: manifest does not match the paper's run {row['run_id']}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
            nq = int(round(float(metrics["num_queries"])))
            if n_queries is None:
                n_queries = nq
            elif nq != n_queries:
                raise ValueError(f"{stem}: query count differs between runs ({nq} vs {n_queries})")
            if int(timings.get("vismatch_candidate_k") or timings.get("benchmark_candidate_k") or k) != k:
                raise ValueError(f"{run_dir}: timings record a different candidate budget than {k}")
            recalls[mkey] = float(metrics["candidate_recall_at_k"])
            job = jobs.get(run_dir_rel)
            partition = partitions.get(job) if job else None
            entry["runs"][mkey] = str(row["run_id"])
            entry["top_5"][mkey] = float(row["top_5"])
            entry["matching_sec"][mkey] = float(timings["vismatch_rerank_sec"])
            entry["partition"][mkey] = partition
            run_partitions.append(partition)
        if abs(recalls["default"] - recalls["finetuned"]) > 1e-12:
            raise ValueError(f"{stem} k={k}: shortlist recall differs between matchers; the runs do not share a shortlist")
        entry["shortlist_share"] = recalls["default"]
        budgets.append(entry)
    assert n_queries is not None
    recorded = [p for p in run_partitions if p]
    dominant = Counter(recorded).most_common(1)[0][0] if recorded else None
    for entry in budgets:
        display = choose_display(entry["matching_sec"], entry["partition"], dominant)
        k = entry["k"]
        entry["display"] = {
            **display,
            "gpu": gpu_label(display["partition"]),
            "minutes_per_1000_queries": minutes_per_1000(display["matching_sec"], n_queries),
            "ms_per_pair": per_pair_ms(display["matching_sec"], n_queries, k),
            "pairs": n_queries * k,
        }
        entry["gpu"] = {m: gpu_label(p) for m, p in entry["partition"].items()}
    return {
        "key": key, "label": label, "stem": stem, "n_queries": n_queries,
        "dominant_gpu": gpu_label(dominant), "runs_with_recorded_hardware": len(recorded), "runs_total": len(run_partitions),
        "budgets": budgets,
    }


def build_payload(datasets: List[Dict[str, Any]], paper_commit: Optional[str], sacct_used: bool) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "scripts/export_budget_tradeoff.py",
        "paper_commit": paper_commit,
        "budgets": BUDGETS, "main_k": MAIN_K,
        "matchers": [{"key": k, "label": label} for k, (_, label) in MATCHERS.items()],
        "definitions": {
            "shortlist_share": "share of queries with at least one image of their individual among the k "
                               "MegaDescriptor-L candidates (candidate_recall_at_k); identical for both matchers.",
            "top_5": "Top-5 accuracy of the matcher's ranking of the k candidates (paper tables).",
            "matching_sec": "Vismatch feature-matching time over all query x candidate pairs (vismatch_rerank_sec); "
                            "feature extraction, candidate selection, model setup and cache I/O excluded.",
            "minutes_per_1000_queries": "matching_sec / 60 x 1000 / queries.",
            "ms_per_pair": "1000 x matching_sec / (queries x k).",
            "hardware": "Slurm partition of the run (sacct) mapped to a GPU name; 'not recorded' for runs that predate "
                        "the launcher's task records." + ("" if sacct_used else " sacct was unavailable at export time."),
        },
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-index", type=Path, default=LOG_INDEX)
    parser.add_argument("--no-sacct", action="store_true", help="do not query Slurm accounting for partitions")
    args = parser.parse_args(argv)
    jobs = job_lookup(args.log_index)
    partitions = {} if args.no_sacct else sacct_partitions(list(jobs.values()))
    datasets = [collect_dataset(stem, key, label, args.paper_repo, jobs, partitions) for stem, key, label in DATASETS]
    for d in datasets:
        main_entry = next(b for b in d["budgets"] if b["k"] == MAIN_K)
        print(f"[budget] {d['label']}: k={MAIN_K} shortlist {main_entry['shortlist_share']:.3f}, Top-5 "
              f"{main_entry['top_5']['default']:.3f} -> {main_entry['top_5']['finetuned']:.3f}, "
              f"{main_entry['display']['minutes_per_1000_queries']:.1f} min/1000 queries, "
              f"{main_entry['display']['ms_per_pair']:.2f} ms/pair on {main_entry['display']['gpu']} "
              f"({d['runs_with_recorded_hardware']}/{d['runs_total']} runs with recorded hardware)", flush=True)
    payload = build_payload(datasets, paper_commit(args.paper_repo), bool(partitions))
    text = json.dumps(payload, indent=1)
    for fragment in ("/shared/", "/home/"):
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in export")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"[budget] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
