#!/usr/bin/env python3
"""Export the data of the project page's "Beyond the paper" section (results the paper does not report).

Three parts, each read from completed run folders under ``experiments/`` (run ids kept, paths dropped):

``rdd``
    RDD-LightGlue retrained with the shared recipe (``experiments/rdd-relaxed/probe/``, sweeps ``rdd_relaxed`` and
    ``rdd_relaxed_czechlynx``) next to the paper's fine-tuned and default RDD-LightGlue. Paper values come from
    the paper runs' own ``metrics.json`` (found through the frozen results snapshot's ``manifest_path``), not
    from the snapshot's numbers, because the snapshot's six SalamanderID2025 fine-tuned rows were rounded.
``sam3``
    The paper's methods rerun on the SAM 3 masks (``dataset.metadata_file`` under ``metadata_sam3/``; sweeps
    ``wildlife_sam3`` and ``wildlife_sam3_grid``), each paired with the paper run of the same identity.
``jaguar``
    JaguarReID (benchmark-only dataset, not in the paper): default and fine-tuned LoMa and RDD-LightGlue, cosine,
    WildFusion and the frozen weighted linear probe (sweeps ``jaguar_grid`` and ``jaguar_finetuned``).

Fails closed on a missing paper run or a private path in the output.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from paper.tools.compare_with_paper import find_paper_row, paper_rows, run_identity  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "docs" / "data" / "beyond.json"
METRICS = ["top_1", "top_5", "balanced_top_1"]
KS = [10, 50, 100, 250, 500, 1000]
RDD_DATASETS = [  # (animal, split, page key, label)
    ("HyenaID2022", "split", "hyena", "Hyena"),
    ("LeopardID2022", "split", "leopard", "Leopard"),
    ("SeaStarReID2023", "split", "sea_star", "Sea star"),
    ("WhaleSharkID", "split", "whale_shark", "Whale shark"),
    ("ZindiTurtleRecall", "split", "turtle", "Turtle"),
    ("SalamanderID2025", "split", "salamander", "Salamander"),
    ("CzechLynx", "unseen_eval_split", "czechlynx_unseen", "CzechLynx (unseen individuals)"),
]
SAM3_DATASETS = [  # the six paper WildlifeReID-10k datasets
    ("HyenaID2022", "hyena", "Hyena"),
    ("LeopardID2022", "leopard", "Leopard"),
    ("NyalaData", "nyala", "Nyala"),
    ("SeaStarReID2023", "sea_star", "Sea star"),
    ("WhaleSharkID", "whale_shark", "Whale shark"),
    ("ZindiTurtleRecall", "turtle", "Turtle"),
]
PRIVATE = ("/shared/", "/home/")


def completed_runs(root: Path) -> List[Path]:
    runs = []
    for manifest in sorted(root.rglob("run_manifest.json")):
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed":
                runs.append(manifest.parent)
        except (OSError, json.JSONDecodeError):
            continue  # an unreadable manifest (e.g. the empty file of 2026-10-04) is skipped
    return runs


def metrics_of(run: Path) -> Dict[str, float]:
    data = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return {m: round(float(data[m]), 6) for m in METRICS}


def identity_key(identity: Dict[str, str]) -> tuple:
    return tuple(
        identity[k]
        for k in ("animal", "split", "method", "matcher", "backbone", "checkpoint", "class_weighting", "candidate_k")
    )


def newest(runs: List[Path]) -> Dict[tuple, Path]:
    out: Dict[tuple, Path] = {}
    for run in runs:
        key = identity_key(run_identity(run))
        if key not in out or run.name > out[key].name:
            out[key] = run
    return out


def paper_metrics(results: Path, identity: Dict[str, str]) -> Dict[str, Any]:
    row = find_paper_row(paper_rows(results, identity["animal"], identity["split"]), identity)
    if row is None:
        raise ValueError(f"no paper row for {identity}")
    run = REPO_ROOT / Path(row["manifest_path"]).parent
    return {**metrics_of(run), "run_id": row["run_id"]}


def export_rdd(results: Path) -> Dict[str, Any]:
    retrained = newest(completed_runs(REPO_ROOT / "experiments" / "rdd-relaxed" / "probe"))
    datasets = {}
    for animal, split, key, label in RDD_DATASETS:
        ks = [10, 50, 100, 160] if split == "unseen_eval_split" else KS
        series: Dict[str, Dict[str, Any]] = {"rdd_default": {}, "rdd_paper": {}, "rdd_retrained": {}}
        for k in ks:
            base = {
                "animal": animal,
                "split": split,
                "method": "Vismatch",
                "matcher": "rdd-lightglue",
                "backbone": "megadescriptor-l",
                "class_weighting": "",
                "candidate_k": str(k),
            }
            series["rdd_default"][str(k)] = paper_metrics(results, {**base, "checkpoint": "default"})
            series["rdd_paper"][str(k)] = paper_metrics(results, {**base, "checkpoint": "fine-tuned"})
            run = retrained.get(identity_key({**base, "checkpoint": "fine-tuned"}))
            if run is None:
                raise ValueError(f"no retrained RDD run for {animal} k={k}")
            series["rdd_retrained"][str(k)] = {**metrics_of(run), "run_id": run.name}
        datasets[key] = {"label": label, "ks": ks, "series": series}
    return {
        "series_order": ["rdd_retrained", "rdd_paper", "rdd_default"],
        "series_labels": {
            "rdd_retrained": "RDD-LightGlue, shared recipe (retrained)",
            "rdd_paper": "RDD-LightGlue, fine-tuned (paper)",
            "rdd_default": "RDD-LightGlue (default)",
        },
        "datasets": datasets,
    }


def export_sam3(results: Path) -> Dict[str, Any]:
    root = REPO_ROOT / "experiments" / "probe" / "WildlifeReID-10k"
    runs = []
    for run in completed_runs(root):
        cfg = yaml.safe_load((run / "config.snapshot.yaml").read_text(encoding="utf-8"))
        if str(cfg["dataset"]["metadata_file"]).startswith("metadata_sam3/"):
            runs.append(run)
    datasets: Dict[str, Any] = {}
    for animal, key, label in SAM3_DATASETS:
        rows = []
        for ident, run in sorted(newest([r for r in runs if run_identity(r)["animal"] == animal]).items()):
            identity = run_identity(run)
            paper = paper_metrics(results, identity)
            method = identity["method"] + ("" if identity["matcher"] == "-" else f" {identity['matcher']}")
            if identity["checkpoint"] != "default":
                method += f" ({identity['checkpoint']})"
            rows.append(
                {
                    "method": method,
                    "candidate_k": identity["candidate_k"],
                    "paper": paper,
                    "sam3": {**metrics_of(run), "run_id": run.name},
                }
            )
        at_250 = [r for r in rows if r["candidate_k"] in ("250", "")]
        datasets[key] = {
            "label": label,
            "rows": rows,
            "mean_delta_top_1_k250": round(
                sum(r["sam3"]["top_1"] - r["paper"]["top_1"] for r in at_250) / len(at_250), 6
            ),
            "n_k250": len(at_250),
        }
    return {"datasets": datasets}


def export_jaguar() -> Dict[str, Any]:
    rows = []
    for ident, run in sorted(newest(completed_runs(REPO_ROOT / "experiments" / "probe" / "JaguarReID")).items()):
        identity = run_identity(run)
        if identity["split"] != "split_v2":
            continue
        rows.append(
            {
                **{k: identity[k] for k in ("method", "matcher", "checkpoint", "candidate_k")},
                **metrics_of(run),
                "run_id": run.name,
            }
        )
    return {"rows": rows}


def check_no_private_paths(payload: Any) -> None:
    text = json.dumps(payload)
    for fragment in PRIVATE:
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in the export")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from paper.tools.compare_with_paper import default_results_dir

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=default_results_dir(), help="the paper's results/ snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "metrics": METRICS,
        "metric_labels": {"top_1": "Top-1", "top_5": "Top-5", "balanced_top_1": "Balanced top-1"},
        "rdd": export_rdd(args.results),
        "sam3": export_sam3(args.results),
        "jaguar": export_jaguar(),
    }
    check_no_private_paths(payload)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
