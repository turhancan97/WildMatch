"""Per-task log records of sweep tasks and the ``logs/index.csv`` index built from them.

Each task writes ``<logs_root>/<dataset>/<animal>/<split>/job-<id>/task-NNN__....json`` next to
its ``.out``, ``.err`` and ``.combined.log``. ``write_index`` rebuilds the index atomically
under a file lock from every record. Moved from ``scripts/probe_log_metadata.py`` and
``scripts/summarize_logs.py`` (2026-10-04); record fields and index columns are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from wildmatch.sweep.manifest import atomic_json

FIELDS = [
    "job_id",
    "task_id",
    "submission_id",
    "manifest_path",
    "profile_id",
    "dataset",
    "animal",
    "split_protocol",
    "method",
    "matcher",
    "class_weighting",
    "checkpoint",
    "checkpoint_owner",
    "checkpoint_sha256",
    "validation_status",
    "validation_error",
    "candidate_k",
    "status",
    "start_time",
    "end_time",
    "experiment_run_directory",
    "stdout_path",
    "stderr_path",
    "combined_path",
    "error_summary",
    "command",
]
STATUSES = ("running", "completed", "failed")


def error_summary(error_file: Optional[Path]) -> str:
    if not error_file:
        return ""
    path = Path(error_file)
    if not path.is_file():
        return ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return ""
    pattern = re.compile(r"(traceback|error|exception|oom|out.of.memory|killed|failed|permission)", re.IGNORECASE)
    matching = [line for line in lines if pattern.search(line)]
    return (matching[-1] if matching else lines[-1])[:1000]


def init_record(path: Path, *, status: str = "running", job_id: str, task_id: int, dataset: str, animal: str,
                split_protocol: str = "", method: str, matcher: str, train_mode: str = "-",
                class_weighting: str = "-", checkpoint: str, checkpoint_path: str, candidate_k: int,
                command: str, start_time: str, stdout_path: str, stderr_path: str, combined_path: str,
                submission_id: str = "", manifest_path: str = "", profile_id: str = "",
                checkpoint_owner: str = "", checkpoint_sha256: str = "", validation_status: str = "") -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    atomic_json(path, {
        "schema_version": 1,
        "status": status,
        "job_id": job_id,
        "task_id": int(task_id),
        "dataset": dataset,
        "animal": animal,
        "split_protocol": split_protocol,
        "method": method,
        "matcher": matcher,
        "train_mode": train_mode,
        "class_weighting": class_weighting,
        "checkpoint": checkpoint,
        "checkpoint_path": None if checkpoint_path in {"", "-", None} else checkpoint_path,
        "candidate_k": int(candidate_k),
        "command": command,
        "start_time": start_time,
        "end_time": "",
        "experiment_run_directory": "",
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "combined_path": combined_path,
        "error_summary": "",
        "submission_id": submission_id,
        "manifest_path": manifest_path,
        "profile_id": profile_id,
        "checkpoint_owner": checkpoint_owner,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_status": validation_status,
        "validation_error": "",
    })


def update_record(path: Path, *, status: str, end_time: str, experiment_run_directory: str = "",
                  error_file: Optional[Path] = None, validation_status: str = "", validation_error: str = "") -> None:
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if not path.is_file():
        raise ValueError(f"metadata file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["end_time"] = end_time
    payload["experiment_run_directory"] = experiment_run_directory or ""
    # Only failures get a summary: a successful task's last stderr line is usually a progress bar.
    payload["error_summary"] = error_summary(error_file) if status == "failed" else ""
    if validation_status:
        payload["validation_status"] = validation_status
    if validation_error:
        payload["validation_error"] = validation_error
    atomic_json(path, payload)


def load_records(logs_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not logs_root.is_dir():
        return records
    for path in sorted(logs_root.rglob("task-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record["_metadata_path"] = str(path)
        records.append(record)
    return records


def _sort_key(record: Mapping[str, Any], field: str) -> tuple:
    value = record.get(field, "")
    if value in (None, ""):
        return (1, "")
    if field == "candidate_k":
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, "")
    return (0, str(value).lower())


def sort_records(records: Iterable[Mapping[str, Any]], field: str) -> List[Mapping[str, Any]]:
    return sorted(records, key=lambda record: _sort_key(record, field))


def _display_path(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(value)


def _index_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    row = {field: record.get(field, "") for field in FIELDS}
    for field in ("stdout_path", "stderr_path", "combined_path", "experiment_run_directory"):
        row[field] = _display_path(row[field])
    return row


def write_index(logs_root: Path, index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_name(index_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        # Read while holding the lock. Otherwise two finishing tasks can each
        # scan an incomplete view and the later atomic replace can omit a row.
        records = load_records(logs_root)
        fd, temporary = tempfile.mkstemp(prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                for record in sort_records(records, "start_time"):
                    writer.writerow(_index_row(record))
            os.replace(temporary, index_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _markdown(records: Sequence[Mapping[str, Any]]) -> str:
    columns = ["job_id", "task_id", "dataset", "animal", "split_protocol", "method", "matcher",
               "checkpoint", "candidate_k", "status", "error_summary"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in records:
        values = [str(record.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _csv(records: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(_index_row(record))
    return output.getvalue().rstrip("\n")


_FILTERS = ("dataset", "animal", "split_protocol", "method", "matcher", "checkpoint", "status")


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Summarize and index sweep task logs.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs/parallel_run"))
    parser.add_argument("--index-path", type=Path)
    for key in _FILTERS:
        parser.add_argument(f"--{key.replace('_', '-')}")
    parser.add_argument("--sort-by", default="start_time", choices=FIELDS)
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--write-index", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    args = build_parser(prog).parse_args(argv)
    all_records = load_records(args.logs_root)
    if args.write_index:
        write_index(args.logs_root, args.index_path or args.logs_root.parent / "index.csv")
    selected = [r for r in all_records
                if all(not getattr(args, key) or str(r.get(key, "")) == getattr(args, key) for key in _FILTERS)]
    records = sort_records(selected, args.sort_by)
    if not args.quiet:
        print(_csv(records) if args.format == "csv" else _markdown(records))
    return 0
