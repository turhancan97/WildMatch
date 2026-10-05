"""Check that every row of reports/runs.csv points at a run directory that exists.

Run directories are sometimes moved (the Salamander parity runs went to
``experiments/parity-reference/`` on 2026-10-04) or deleted (the first-split JaguarReID runs, old
superseded runs), and the index keeps the stale rows. This tool reports them and, with ``--fix``,
repoints moved runs (found under ``--root`` by run id, with a manifest of the same run id) and marks
the rest ``missing``. Only ``completed`` rows are checked: a ``failed`` row keeps its status (readers
skip it anyway, and its run directory often never existed). It never deletes a row. The index is rewritten under the same lock as the
sweep writers, after a timestamped backup copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wildmatch.utils.io import file_lock, write_csv_atomically

PATH_COLUMNS = ("run_dir", "manifest_path", "metrics_path", "visualization_dir")
MISSING = "missing"


def parse_args(argv=None, prog=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog, description="Find runs.csv rows whose run directory is gone")
    parser.add_argument("--index", type=Path, default=Path("reports/runs.csv"))
    parser.add_argument("--root", type=Path, default=Path("experiments"), help="where moved runs are searched")
    parser.add_argument("--fix", action="store_true", help="repoint moved runs, mark the rest missing")
    return parser.parse_args(argv)


def _run_dirs_by_name(root: Path) -> Dict[str, List[Path]]:
    found: Dict[str, List[Path]] = {}
    for current, _, files in os.walk(root, followlinks=True):
        if "run_manifest.json" in files:
            path = Path(current)
            found.setdefault(path.name, []).append(path)
    return found


def _manifest_run_id(run_dir: Path) -> Optional[str]:
    try:
        return str(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")).get("run_id", ""))
    except (OSError, ValueError):
        return None


def plan(rows: List[Dict[str, str]], root: Path) -> List[Tuple[int, str, str]]:
    """``(row index, action, detail)`` for every completed row whose run directory is gone."""
    found: Optional[Dict[str, List[Path]]] = None
    actions: List[Tuple[int, str, str]] = []
    for index, row in enumerate(rows):
        run_dir = row.get("run_dir", "")
        if row.get("status") != "completed" or not run_dir or Path(run_dir).is_dir():
            continue
        if found is None:
            found = _run_dirs_by_name(root)
        candidates = [path for path in found.get(row.get("run_id", ""), []) if _manifest_run_id(path) == row["run_id"]]
        if len(candidates) == 1:
            actions.append((index, "moved", candidates[0].as_posix()))
        elif candidates:
            actions.append((index, "ambiguous", ", ".join(path.as_posix() for path in candidates)))
        else:
            actions.append((index, MISSING, ""))
    return actions


def apply(rows: List[Dict[str, str]], actions: List[Tuple[int, str, str]]) -> None:
    for index, action, detail in actions:
        row = rows[index]
        if action == "moved":
            old = row["run_dir"].rstrip("/")
            for column in PATH_COLUMNS:
                if row.get(column, "").startswith(old):
                    row[column] = detail + row[column][len(old) :]
        elif action == MISSING:
            row["status"] = MISSING


def _read(index_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def main(argv=None, prog=None) -> int:
    args = parse_args(argv, prog)
    with file_lock(args.index):
        fieldnames, rows = _read(args.index)
        actions = plan(rows, args.root)
        for index, action, detail in actions:
            row = rows[index]
            print(
                f"{action:9s} {row.get('run_id', '')}  {row.get('run_dir', '')}" + (f"  -> {detail}" if detail else "")
            )
        counts = {name: sum(action == name for _, action, _ in actions) for name in ("moved", MISSING, "ambiguous")}
        print(f"{len(rows)} rows; " + ", ".join(f"{name} {count}" for name, count in counts.items()))
        if not args.fix or not (counts["moved"] or counts[MISSING]):
            if actions and not args.fix:
                print("dry run; pass --fix to update the index")
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = args.index.with_name(f"{args.index.name}.bak-{stamp}")
        shutil.copy2(args.index, backup)
        apply(rows, actions)
        write_csv_atomically(args.index, fieldnames, rows)
        print(f"updated {args.index} (backup {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
