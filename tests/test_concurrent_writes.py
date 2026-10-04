"""Shared CSVs and feature-cache files survive parallel writers (parallel sweep tasks)."""

import csv
import multiprocessing
import os
import stat
import tempfile
import unittest
from pathlib import Path

from wildmatch.reporting.artifacts import upsert_run_index
from wildmatch.utils.io import append_csv_row, update_csv_rows

WORKERS = 6
ROWS_PER_WORKER = 15


def _upsert_many(index_path: str, worker: int) -> None:
    for number in range(ROWS_PER_WORKER):
        upsert_run_index(Path(index_path), {"run_id": f"w{worker}-r{number}", "status": "completed"})


def _append_many(csv_path: str, worker: int) -> None:
    for number in range(ROWS_PER_WORKER):
        row = {"run": f"w{worker}-r{number}"}
        if number == 5:
            row[f"extra_{worker}"] = "x"  # schema growth rewrites the file under the lock
        append_csv_row(Path(csv_path), row)


def _run_workers(target, path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=target, args=(str(path), worker)) for worker in range(WORKERS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0


def _ids(path: Path, column: str) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row[column] for row in csv.DictReader(handle)]


def _no_leftovers(folder: Path) -> list:
    return [p.name for p in folder.iterdir() if p.name.endswith(".tmp") or ".tmp." in p.name]


class ConcurrentWriteTests(unittest.TestCase):
    def test_parallel_run_index_updates_keep_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "runs.csv"
            _run_workers(_upsert_many, index)
            ids = _ids(index, "run_id")
            self.assertEqual(len(ids), WORKERS * ROWS_PER_WORKER)
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(_no_leftovers(Path(tmp)), [])

    def test_parallel_appends_keep_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "benchmark_results.csv"
            _run_workers(_append_many, target)
            self.assertEqual(
                sorted(_ids(target, "run")), sorted(f"w{w}-r{n}" for w in range(WORKERS) for n in range(15))
            )
            self.assertEqual(_no_leftovers(Path(tmp)), [])

    def test_rewrites_keep_the_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "results.csv"
            append_csv_row(target, {"run": "a", "value": "1"})
            os.chmod(target, 0o640)
            self.assertEqual(update_csv_rows(target, {"run": "a"}, {"value": "2"}), 1)
            upsert_run_index(Path(tmp) / "runs.csv", {"run_id": "a"})
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            self.assertEqual(_ids(target, "value"), ["2"])
            self.assertNotEqual(stat.S_IMODE((Path(tmp) / "runs.csv").stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
