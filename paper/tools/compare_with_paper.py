#!/usr/bin/env python3
"""Compare new probe runs with the paper's frozen results snapshot.

For every completed run of a sweep (``--job <array id>``, read from ``logs/index.csv``) or of
explicit run directories (``--run DIR``), find the paper row with the same identity in the paper
repository's ``results/<animal>_<split>_ablation.csv``: method, matcher, backbone, checkpoint label,
classifier-probe mode and weighting, and candidate budget (shortlist methods only). Prints a
table of Top-1, Top-5 and balanced Top-1 (percent) with the differences, and writes it as CSV
with ``--output``. Rows without a paper counterpart are listed with empty paper values.

Used to measure how far results move on the SAM 3 WildlifeReID-10k inputs (sweep
``wildlife_sam3``, 2026-10-04). Run from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

METRICS = ("top_1", "top_5", "balanced_top_1")
METHOD_LABELS = {"cosine": "Cosine", "wildfusion": "WildFusion", "vismatch": "Vismatch",
                 "linear_probe": "Linear Probe", "efficient_probe": "Efficient Probe",
                 "local_lightglue": "Local LightGlue"}
SHORTLIST_METHODS = {"wildfusion", "vismatch", "local_lightglue"}
PROBE_METHODS = {"linear_probe", "efficient_probe"}
TRAIN_MODE_LABELS = {"classifier": "frozen", "partial": "partial fine-tuned", "all": "full fine-tuned"}
WEIGHTING_LABELS = {"inverse_frequency": "weighted", "none": "unweighted"}
CHECKPOINT_LABELS = {"default": "default", "custom": "fine-tuned", "matcher-fine-tuned": "fine-tuned"}


def run_identity(run_dir: Path) -> Dict[str, str]:
    """The paper-table identity of one run, from its config snapshot."""
    config = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    benchmark, dataset = config["benchmark"], config["dataset"]
    method = str(benchmark["method"])
    methods = benchmark.get("methods", {})
    identity = {
        "animal": str(dataset["animal"]),
        "split": str(dataset["split_col"]),
        "method": METHOD_LABELS.get(method, method),
        "matcher": "-",
        "backbone": str(config["model"]["type"]),
        "checkpoint": "default",
        "class_weighting": "",
        "candidate_k": str(benchmark["candidate_k"]) if method in SHORTLIST_METHODS else "",
    }
    if method == "vismatch":
        settings = methods["vismatch"]
        identity["matcher"] = str(settings["matcher"])
        if str(settings.get("checkpoint_source", "default")) == "custom":
            identity["checkpoint"] = "fine-tuned"
    if method in PROBE_METHODS:
        settings = methods[method]
        weighting = WEIGHTING_LABELS.get(str(settings.get("class_weighting")), "unknown")
        identity["class_weighting"] = weighting
        identity["checkpoint"] = f"{TRAIN_MODE_LABELS.get(str(settings.get('train_mode')), 'unknown')} ({weighting})"
    return identity


def paper_rows(results_dir: Path, animal: str, split: str) -> List[Dict[str, str]]:
    path = results_dir / f"{animal}_{split}_ablation.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_paper_row(rows: Iterable[Dict[str, str]], identity: Dict[str, str]) -> Optional[Dict[str, str]]:
    for row in rows:
        if all(str(row.get(key, "")) == identity[key]
               for key in ("method", "matcher", "backbone", "checkpoint", "class_weighting", "candidate_k")):
            return row
    return None


def runs_of_job(index_path: Path, job: str) -> List[Path]:
    with index_path.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["job_id"] == job and r["status"] == "completed"]
    return [Path(r["experiment_run_directory"]) for r in sorted(rows, key=lambda r: int(r["task_id"]))]


def compare(runs: Iterable[Path], results_dir: Path) -> List[Dict[str, object]]:
    table = []
    for run_dir in runs:
        identity = run_identity(run_dir)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        paper = find_paper_row(paper_rows(results_dir, identity["animal"], identity["split"]), identity)
        row: Dict[str, object] = {**identity, "run": run_dir.name, "paper_run": paper["run_id"] if paper else ""}
        for metric in METRICS:
            new = float(metrics[metric])
            row[f"new_{metric}"] = round(100 * new, 2)
            row[f"paper_{metric}"] = round(100 * float(paper[metric]), 2) if paper else ""
            row[f"delta_{metric}"] = round(100 * (new - float(paper[metric])), 2) if paper else ""
        table.append(row)
    return table


def render(table: List[Dict[str, object]]) -> str:
    def triple(row, prefix):
        values = [row[f"{prefix}_{m}"] for m in METRICS]
        return " / ".join(f"{v:.1f}" if v != "" else "n/a" for v in values)

    lines = ["| Dataset | Method | k | Paper T1 / T5 / bT1 | New T1 / T5 / bT1 | dT1 | dbT1 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        method = row["method"] + ("" if row["matcher"] == "-" else f" {row['matcher']}")
        if row["checkpoint"] != "default":
            method += f" {row['checkpoint']}"
        delta = lambda key: f"{row[key]:+.1f}" if row[key] != "" else ""
        lines.append(f"| {row['animal']} | {method} | {row['candidate_k'] or '-'} | {triple(row, 'paper')} | "
                     f"{triple(row, 'new')} | {delta('delta_top_1')} | {delta('delta_balanced_top_1')} |")
    return "\n".join(lines)


def default_results_dir() -> Optional[Path]:
    try:
        from wildmatch.paths import path
    except ImportError:
        return None
    repo = path("external.paper_repo")
    return repo / "results" if repo else None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job", help="sweep array id; completed tasks are read from --index")
    source.add_argument("--run", type=Path, action="append", help="run directory (repeatable)")
    parser.add_argument("--index", type=Path, default=Path("logs/index.csv"))
    parser.add_argument("--paper-results", type=Path, default=None,
                        help="the paper repository's results/ (default: paths external.paper_repo)")
    parser.add_argument("--output", type=Path, help="also write the table as CSV")
    args = parser.parse_args(argv)
    results_dir = args.paper_results or default_results_dir()
    if results_dir is None or not results_dir.is_dir():
        parser.error("paper results not found: pass --paper-results")
    runs = runs_of_job(args.index, args.job) if args.job else args.run
    if not runs:
        parser.error("no completed runs to compare")
    table = compare(runs, results_dir)
    print(render(table))
    missing = sum(1 for row in table if not row["paper_run"])
    if missing:
        print(f"\n{missing} run(s) have no paper counterpart", file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
