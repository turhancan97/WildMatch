#!/usr/bin/env python3
"""Export the project-page matcher ablation: ALIKED-LightGlue and SuperPoint-LightGlue with their
default weights next to the paper's default LoMa and RDD-LightGlue, on the eight paper datasets and the
paper's budget grid k = 10, 50, 100, 250, 500, 1000.

Sources:
* the default LoMa and RDD-LightGlue points come from the committed ``docs/data/curves.json`` (built by
  ``export_project_page_data.py`` from the paper's frozen results snapshot), so the page shows exactly the
  numbers it shows elsewhere;
* the ALIKED-LightGlue and SuperPoint-LightGlue points come from the completed runs of the sweeps
  ``matcher_ablation`` (k = 250) and ``matcher_ablation_grid`` (the other budgets) under
  ``<experiments>/matcher-ablation/probe/`` (paper input tables, same MegaDescriptor-L candidates, default
  weights, RTX 4090). The newest completed run per dataset, matcher and budget is used; its run id and GPU
  are kept, its path is not.

The mean over the eight datasets is given only where every dataset has a value. Fails closed on a missing
run, an unexpected dataset or split, or a private path in the output. Not part of the paper.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CURVES = REPO_ROOT / "docs" / "data" / "curves.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "data" / "matcher_ablation.json"
DEFAULT_RUNS = REPO_ROOT / "experiments" / "matcher-ablation" / "probe"

KS = [10, 50, 100, 250, 500, 1000]
METRICS = ["top_1", "top_5", "top_10", "balanced_top_1"]
# run config animal -> curves.json dataset key (CzechLynx: the closed split only)
ANIMALS = {
    "CzechLynx": "czechlynx",
    "HyenaID2022": "hyena",
    "LeopardID2022": "leopard",
    "NyalaData": "nyala",
    "SalamanderID2025": "salamander",
    "SeaStarReID2023": "sea_star",
    "WhaleSharkID": "whale_shark",
    "ZindiTurtleRecall": "turtle",
}
NEW_MATCHERS = {"aliked-lightglue": "aliked_default", "superpoint-lightglue": "superpoint_default"}
SERIES = [
    ("loma_default", "LoMa (default)"),
    ("rdd_default", "RDD-LightGlue (default)"),
    ("superpoint_default", "SuperPoint-LightGlue (default)"),
    ("aliked_default", "ALIKED-LightGlue (default)"),
]
PRIVATE = ("/shared/", "/home/")


def collect_runs(runs_root: Path) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """{dataset key: {series key: {k: point}}} for the two ablation matchers."""
    newest: Dict[tuple, tuple] = {}
    for metrics_path in sorted(runs_root.rglob("metrics.json")):
        run = metrics_path.parent
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        cfg = yaml.safe_load((run / "config.snapshot.yaml").read_text(encoding="utf-8"))
        vismatch = cfg["benchmark"]["methods"]["vismatch"]
        matcher = vismatch["matcher"]
        if matcher not in NEW_MATCHERS or vismatch.get("checkpoint_source", "default") != "default":
            continue
        animal = cfg["dataset"]["animal"]
        if animal not in ANIMALS:
            raise ValueError(f"{run.name}: unexpected animal {animal!r}")
        if animal == "CzechLynx" and cfg["dataset"].get("split_col") != "split-time_closed":
            raise ValueError(f"{run.name}: CzechLynx run is not on the closed split")
        key = (ANIMALS[animal], NEW_MATCHERS[matcher], int(cfg["benchmark"]["candidate_k"]))
        if key not in newest or run.name > newest[key][0]:
            newest[key] = (run.name, run, manifest)
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for (dataset, series, k), (name, run, manifest) in newest.items():
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        timings = json.loads((run / "timings.json").read_text(encoding="utf-8"))
        pairs = metrics.get("num_candidate_pairs") or 0
        seconds = timings.get("primary_compute_runtime_sec")
        point = {m: round(float(metrics[m]), 6) for m in METRICS}
        point["ms_per_pair"] = round(1000.0 * seconds / pairs, 3) if seconds and pairs else None
        point["run_id"] = name
        point["device"] = manifest.get("vismatch_device") or metrics.get("vismatch_device")
        out.setdefault(dataset, {}).setdefault(series, {})[str(k)] = point
    return out


def build(curves: Dict[str, Any], runs: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    for key in ANIMALS.values():
        source = curves["datasets"][key]
        series: Dict[str, Any] = {}
        for skey, label in SERIES:
            if skey in ("loma_default", "rdd_default"):
                points = {
                    k: {m: source["series"][skey]["points"][k].get(m) for m in METRICS}
                    for k in map(str, KS)
                    if k in source["series"][skey]["points"]
                }
            else:
                points = runs.get(key, {}).get(skey, {})
            missing = [k for k in KS if str(k) not in points]
            if missing:
                raise ValueError(f"{key} / {skey}: no run for k = {missing}")
            series[skey] = {"label": label, "points": points}
        datasets[key] = {"label": source["label"], "series": series}
    mean_series: Dict[str, Any] = {}
    for skey, label in SERIES:
        points = {}
        for k in map(str, KS):
            values = {m: [datasets[d]["series"][skey]["points"][k].get(m) for d in datasets] for m in METRICS}
            points[k] = {m: round(sum(v) / len(v), 6) if None not in v else None for m, v in values.items()}
        mean_series[skey] = {"label": label, "points": points}
    devices = sorted(
        {
            p["device"]
            for d in datasets.values()
            for s in d["series"].values()
            for p in s["points"].values()
            if p.get("device")
        }
    )
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "paper_commit": curves.get("paper_commit"),
        "note": "Extension, not in the paper: default weights, paper input tables, same MegaDescriptor-L candidates.",
        "ks": KS,
        "metrics": METRICS,
        "metric_labels": curves["metric_labels"],
        "series_order": [skey for skey, _ in SERIES],
        "ablation_devices": devices,
        "datasets": datasets,
        "mean": {"label": "Mean of the eight datasets", "series": mean_series},
    }


def check_no_private_paths(payload: Any) -> None:
    text = json.dumps(payload)
    for fragment in PRIVATE:
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in the export")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="matcher-ablation probe root")
    parser.add_argument("--curves", type=Path, default=CURVES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    curves = json.loads(args.curves.read_text(encoding="utf-8"))
    payload = build(curves, collect_runs(args.runs))
    check_no_private_paths(payload)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(payload['datasets'])} datasets, devices {payload['ablation_devices']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
