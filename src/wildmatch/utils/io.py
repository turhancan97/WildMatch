import csv
import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def ensure_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``<path>.lock`` around a read-modify-write of a shared file.

    Parallel sweep tasks update the same CSVs (``reports/runs.csv``, the legacy benchmark CSV);
    without the lock two tasks can each read the old file and the later write drops the other's row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_csv_atomically(csv_path: Path, header: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write a CSV through a unique temporary file in the same folder, then replace the target.

    The old fixed ``<name>.tmp`` path was shared by every writer, so two processes could write into
    one temporary file. The target keeps its permissions (a new file gets the umask default).
    """
    fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", suffix=".tmp", dir=csv_path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(_sanitize_row(row))
        os.chmod(temporary, _target_mode(csv_path))
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _target_mode(path: Path) -> int:
    try:
        return path.stat().st_mode & 0o777
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    with file_lock(csv_path):
        _append_csv_row_locked(csv_path, row)


def _append_csv_row_locked(csv_path: Path, row: Dict[str, Any]) -> None:
    if not csv_path.is_file():
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_header = reader.fieldnames or []
        existing_rows = list(reader)

    if not existing_header:
        fieldnames = list(row.keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        return

    new_keys = [k for k in row.keys() if k not in existing_header]
    if new_keys:
        print(f"CSV schema normalized: {csv_path} (added {len(new_keys)} column{'s' if len(new_keys) != 1 else ''})")
        _rewrite_csv_with_header(
            csv_path=csv_path,
            rows=existing_rows,
            header=existing_header + new_keys,
        )
        existing_header = existing_header + new_keys

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=existing_header, extrasaction="ignore")
        writer.writerow(row)


def update_csv_rows(
    csv_path: Path,
    match: Dict[str, Any],
    updates: Dict[str, Any],
) -> int:
    """Update rows matching all key/value pairs and rewrite the CSV atomically."""
    with file_lock(csv_path):
        return _update_csv_rows_locked(csv_path, match, updates)


def _update_csv_rows_locked(
    csv_path: Path,
    match: Dict[str, Any],
    updates: Dict[str, Any],
) -> int:

    if not csv_path.is_file():
        return 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header:
        return 0

    new_keys = [key for key in updates if key not in header]
    header.extend(new_keys)
    updated_count = 0
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in match.items()):
            row.update({key: value for key, value in updates.items()})
            updated_count += 1
    if updated_count:
        _rewrite_csv_with_header(csv_path=csv_path, rows=rows, header=header)
    return updated_count


def _rewrite_csv_with_header(csv_path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    write_csv_atomically(csv_path, header, rows)


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # csv.DictReader uses None as a key when a row has more values than the header.
    # We drop these orphan values so schema rewrites stay robust.
    return {k: v for k, v in row.items() if k is not None}
