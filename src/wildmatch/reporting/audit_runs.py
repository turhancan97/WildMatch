"""Read-only audit of run metadata: which runs and index rows predate which provenance fields.

Run manifests and the run index gained fields over time (the ``code`` identity block, checkpoint
SHA-256s, ``vismatch_device``, primary timing fields, ...). Historical artifacts are never rewritten
(AGENTS.md), so this tool only reports, per gap, how many runs or rows lack the field and which
ones; readers label missing values ``unknown`` instead of guessing. Nothing on disk is changed
unless ``--output`` names a CSV to write the per-run listing to.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from wildmatch.reporting.artifacts import ARTIFACT_SCHEMA_VERSION

# gap -> (what it means, when the field appeared). Order is the report order.
MANIFEST_GAPS: Dict[str, Tuple[str, str]] = {
    "unreadable": ("run_manifest.json cannot be parsed", "-"),
    "schema_version": (f"no or old schema_version (current {ARTIFACT_SCHEMA_VERSION})", "-"),
    "git_commit": ("no git_commit", "2026-08"),
    "code_identity": ("no code block (commit, dirty flag, diff hash)", "2026-10-04"),
    "model": ("no model field (tables label the backbone unknown)", "2026-08"),
    "split_protocol": ("no split_protocol (legacy unsuffixed outputs)", "2026-08"),
    "primary_timing": ("probe without primary_compute_runtime_sec (tables show --)", "2026-08"),
    "checkpoint_sha256": ("fine-tuned checkpoint recorded without SHA-256", "2026-10-04"),
    "vismatch_device": ("Vismatch run without the GPU model (vismatch_device)", "2026-10-05"),
}
INDEX_COLUMNS = ("model", "split_protocol", "git_commit", "primary_compute_runtime_sec")


def _has_sha256(value: Any) -> bool:
    return "sha256" in json.dumps(value, default=str) and '"sha256": null' not in json.dumps(value, default=str)


def manifest_gaps(manifest: Dict[str, Any]) -> List[str]:
    """The gap names (keys of ``MANIFEST_GAPS``) of one parsed run manifest."""
    gaps: List[str] = []
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        gaps.append("schema_version")
    if not manifest.get("git_commit"):
        gaps.append("git_commit")
    if not isinstance(manifest.get("code"), dict):
        gaps.append("code_identity")
    if not manifest.get("model"):
        gaps.append("model")
    if not manifest.get("split_protocol"):
        gaps.append("split_protocol")
    workflow = str(manifest.get("workflow", ""))
    timings = manifest.get("timings") if isinstance(manifest.get("timings"), dict) else {}
    if workflow == "probe" and "primary_compute_runtime_sec" not in timings:
        gaps.append("primary_timing")
    backbone = manifest.get("checkpoint")
    fine_tuned_backbone = isinstance(backbone, dict) and backbone.get("path")
    fine_tuned_matcher = str(manifest.get("checkpoint_source", "")) == "custom"
    finetune_identity = manifest.get("checkpoint_identity")
    if (
        (fine_tuned_backbone and not _has_sha256(backbone))
        or (fine_tuned_matcher and not _has_sha256(manifest.get("vismatch_checkpoint")))
        or (
            isinstance(finetune_identity, dict)
            and finetune_identity.get("exists")
            and not _has_sha256(finetune_identity)
        )
    ):
        gaps.append("checkpoint_sha256")
    if str(manifest.get("method", "")) == "vismatch" and "vismatch_device" not in manifest:
        gaps.append("vismatch_device")
    return gaps


def iter_manifests(root: Path) -> Iterable[Path]:
    for current, _, files in os.walk(root, followlinks=True):
        if "run_manifest.json" in files:
            yield Path(current) / "run_manifest.json"


def audit_manifests(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(iter_manifests(root)):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            gaps = manifest_gaps(manifest)
        except (OSError, ValueError):
            manifest, gaps = {}, ["unreadable"]
        records.append(
            {
                "run_dir": str(path.parent),
                "run_id": manifest.get("run_id", path.parent.name),
                "status": manifest.get("status", ""),
                "workflow": manifest.get("workflow", ""),
                "method": manifest.get("method", ""),
                "run_utc": manifest.get("run_utc", ""),
                "gaps": gaps,
            }
        )
    return records


def audit_index(index: Path, columns: Sequence[str] = INDEX_COLUMNS) -> Dict[str, Any]:
    """Row count, per-status counts and empty-cell counts of the key columns of a run-index CSV."""
    with index.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    return {
        "rows": len(rows),
        "columns": len(header),
        "status": dict(Counter(row.get("status", "") for row in rows)),
        "absent_columns": [column for column in columns if column not in header],
        "empty": {column: sum(1 for row in rows if not row.get(column)) for column in columns if column in header},
    }


def summarize(records: List[Dict[str, Any]]) -> List[Tuple[str, int, str, str]]:
    counts = Counter(gap for record in records for gap in record["gaps"])
    return [(gap, counts.get(gap, 0), *MANIFEST_GAPS[gap]) for gap in MANIFEST_GAPS]


def parse_args(argv=None, prog=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("experiments"), help="tree of run directories")
    parser.add_argument("--index", type=Path, default=Path("reports/runs.csv"), help="run index to check")
    parser.add_argument("--output", type=Path, default=None, help="write one row per run with its gaps (CSV)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    args = parse_args(argv, prog)
    records = audit_manifests(args.root) if args.root.is_dir() else []
    with_gaps = sum(1 for record in records if record["gaps"])
    print(f"{len(records)} run manifests under {args.root}; {with_gaps} lack at least one field")
    print(f"{'gap':<18} {'runs':>6}  {'since':<10}  meaning")
    for gap, count, meaning, since in summarize(records):
        print(f"{gap:<18} {count:>6}  {since:<10}  {meaning}")
    if args.index.is_file():
        report = audit_index(args.index)
        print(f"\n{args.index}: {report['rows']} rows, {report['columns']} columns, status {report['status']}")
        if report["absent_columns"]:
            print(f"  columns absent from the header: {', '.join(report['absent_columns'])}")
        for column, count in report["empty"].items():
            print(f"  empty {column}: {count}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["run_dir", "run_id", "status", "workflow", "method", "run_utc", "gaps"]
            )
            writer.writeheader()
            for record in records:
                writer.writerow({**record, "gaps": ";".join(record["gaps"])})
        print(f"\nPer-run listing: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
