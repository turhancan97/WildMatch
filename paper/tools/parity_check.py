"""Check that refactored code reproduces reference probe runs (see notes/parity_reference.md).

Pairs every completed run under ``--candidate-root`` with the newest completed run under
``--reference-root`` that has the same identity (dataset, animal, split protocol, backbone,
method, matcher/variant, checkpoint variant, candidate budget and, for classifier probes,
train mode and class weighting), then compares them:

- the scored entries of ``scores.npz`` must be the same set (same shortlist), and the scores
  must agree within the method's tolerance: 1e-2 for LoMa (bfloat16 autocast), 1e-4 otherwise;
- the Top-1 database index of every query must be identical (shared stable tie rule);
- the headline metrics must agree within 1e-6, or within 0.01 for classifier probes, whose
  GPU training is not bit-reproducible (their scores are reported but not gated).

Exit status is non-zero when any pair fails or a candidate run has no reference.

    python paper/tools/parity_check.py \\
        --reference-root /home/.../explainable_individual_reidentification/experiments/probe \\
        --candidate-root experiments/probe --dataset SalamanderID2025
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

METRICS = ("top_1", "top_5", "top_10", "balanced_top_1", "mAP_at_k")
PROBE_METHODS = {"linear_probe", "efficient_probe"}
SCORE_TOLERANCE = {"loma": 1e-2}
DEFAULT_SCORE_TOLERANCE = 1e-4
METRIC_TOLERANCE = 1e-6
PROBE_METRIC_TOLERANCE = 0.01


@dataclass(frozen=True)
class Run:
    path: Path
    identity: Tuple[str, ...]
    run_id: str
    method: str
    variant: str


def _identity(run_dir: Path) -> Optional[Run]:
    manifest_path = run_dir / "run_manifest.json"
    snapshot_path = run_dir / "config.snapshot.yaml"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return None
    config = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    benchmark = config.get("benchmark", {})
    method = str(manifest.get("method"))
    method_cfg = (benchmark.get("methods") or {}).get(method, {}) or {}
    probe_keys = (str(method_cfg.get("train_mode")), str(method_cfg.get("class_weighting"))) if method in PROBE_METHODS else ("-", "-")
    identity = (
        str(manifest.get("dataset")), str(manifest.get("animal")), str(manifest.get("split_protocol")),
        str(manifest.get("model")), method, str(manifest.get("variant")),
        str(manifest.get("checkpoint_variant")), str(benchmark.get("candidate_k")), *probe_keys,
    )
    return Run(run_dir, identity, str(manifest.get("run_id", run_dir.name)), method, str(manifest.get("variant")))


def collect(root: Path, dataset: Optional[str] = None) -> Dict[Tuple[str, ...], Run]:
    """Newest completed run per identity under ``root`` (run ids sort by UTC timestamp)."""
    newest: Dict[Tuple[str, ...], Run] = {}
    for manifest in sorted(root.rglob("run_manifest.json")):
        run = _identity(manifest.parent)
        if run is None or (dataset and run.identity[0] != dataset):
            continue
        if run.identity not in newest or run.run_id > newest[run.identity].run_id:
            newest[run.identity] = run
    return newest


def _load_scores(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    path = run_dir / "scores.npz"
    if not path.is_file():
        return None
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _top1(shape: Tuple[int, int], rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Top-1 column per row with the shared rule: highest score, lowest index on ties."""
    dense = np.full(shape, -np.inf)
    dense[rows, cols] = values
    best = dense.max(axis=1, keepdims=True)
    return np.argmax(dense == best, axis=1)


def compare(reference: Run, candidate: Run) -> Dict[str, object]:
    result: Dict[str, object] = {"reference": reference.run_id, "candidate": candidate.run_id, "problems": []}
    problems: List[str] = result["problems"]  # type: ignore[assignment]
    is_probe = candidate.method in PROBE_METHODS
    ref_metrics = json.loads((reference.path / "metrics.json").read_text(encoding="utf-8"))
    new_metrics = json.loads((candidate.path / "metrics.json").read_text(encoding="utf-8"))
    tolerance = PROBE_METRIC_TOLERANCE if is_probe else METRIC_TOLERANCE
    diffs = {}
    for key in METRICS:
        a, b = ref_metrics.get(key), new_metrics.get(key)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        if np.isnan(a) and np.isnan(b):
            continue
        diffs[key] = float(b) - float(a)
        if abs(diffs[key]) > tolerance:
            problems.append(f"{key} differs by {diffs[key]:+.6f} (tolerance {tolerance})")
    result["metric_diffs"] = diffs

    ref_scores, new_scores = _load_scores(reference.path), _load_scores(candidate.path)
    if ref_scores is None or new_scores is None:
        result["scores"] = "not persisted"
        return result
    if tuple(ref_scores["shape"]) != tuple(new_scores["shape"]):
        problems.append(f"score matrix shape {tuple(new_scores['shape'])} != {tuple(ref_scores['shape'])}")
        return result
    shape = tuple(int(x) for x in ref_scores["shape"])
    ref_keys = ref_scores["rows"].astype(np.int64) * shape[1] + ref_scores["cols"]
    new_keys = new_scores["rows"].astype(np.int64) * shape[1] + new_scores["cols"]
    ref_order, new_order = np.argsort(ref_keys), np.argsort(new_keys)
    same_entries = ref_keys.shape == new_keys.shape and np.array_equal(ref_keys[ref_order], new_keys[new_order])
    if not same_entries:
        problems.append(f"scored entries differ ({len(ref_keys)} vs {len(new_keys)}); the shortlist changed")
    else:
        delta = np.abs(ref_scores["values"][ref_order] - new_scores["values"][new_order])
        result["max_score_diff"] = float(delta.max()) if delta.size else 0.0
        score_tolerance = SCORE_TOLERANCE.get(candidate.variant, DEFAULT_SCORE_TOLERANCE)
        if not is_probe and result["max_score_diff"] > score_tolerance:
            problems.append(f"max score difference {result['max_score_diff']:.2e} > {score_tolerance}")
    ref_top1 = _top1(shape, ref_scores["rows"], ref_scores["cols"], ref_scores["values"])
    new_top1 = _top1(shape, new_scores["rows"], new_scores["cols"], new_scores["values"])
    result["top1_disagreements"] = int((ref_top1 != new_top1).sum())
    if result["top1_disagreements"] and not is_probe:
        problems.append(f"{result['top1_disagreements']} queries change their Top-1 database image")
    return result


def run(reference_root: Path, candidate_root: Path, dataset: Optional[str]) -> int:
    references = collect(reference_root, dataset)
    candidates = collect(candidate_root, dataset)
    if not candidates:
        print(f"no completed candidate runs under {candidate_root}")
        return 1
    failures = 0
    for identity, candidate in sorted(candidates.items()):
        label = "/".join(part for part in identity if part not in ("-", "None"))
        reference = references.get(identity)
        if reference is None:
            print(f"MISSING  {label}: no reference run")
            failures += 1
            continue
        outcome = compare(reference, candidate)
        status = "FAIL" if outcome["problems"] else "PASS"
        failures += status == "FAIL"
        extra = f" max|dscore|={outcome['max_score_diff']:.1e}" if "max_score_diff" in outcome else ""
        extra += f" top1_changes={outcome.get('top1_disagreements', '-')}"
        print(f"{status:8s} {label}: {outcome['reference']} -> {outcome['candidate']}{extra}")
        for problem in outcome["problems"]:
            print(f"         - {problem}")
    print(f"{len(candidates) - failures}/{len(candidates)} pairs pass")
    return 1 if failures else 0


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, required=True, help="experiments/probe of the paper-v1 checkout")
    parser.add_argument("--candidate-root", type=Path, required=True, help="experiments/probe of the refactor checkout")
    parser.add_argument("--dataset", help="only pairs of this dataset (e.g. SalamanderID2025)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    sys.exit(run(args.reference_root, args.candidate_root, args.dataset))


if __name__ == "__main__":
    main()
