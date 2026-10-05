"""wildmatch check-index: repoint moved runs, mark deleted ones, never touch the others."""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from wildmatch.reporting.check_index import main

HEADER = ["run_id", "status", "run_dir", "manifest_path", "metrics_path", "visualization_dir", "top_1"]


def _run(root: Path, relative: str, run_id: str) -> None:
    run = root / relative
    run.mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")


def _row(run_id: str, run_dir: str, status: str = "completed") -> dict:
    return {
        "run_id": run_id,
        "status": status,
        "run_dir": run_dir,
        "manifest_path": f"{run_dir}/run_manifest.json",
        "metrics_path": f"{run_dir}/metrics.json",
        "visualization_dir": f"{run_dir}/visualizations",
        "top_1": "0.5",
    }


class CheckIndexTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.addCleanup(os.chdir, self._cwd)
        root = Path("experiments")
        _run(root, "probe/A/r_ok", "r_ok")
        _run(root, "parity-reference/probe/A/r_moved", "r_moved")
        _run(root, "other/r_wrong_id", "something_else")  # same folder name, different run: not a match
        self.index = Path("reports/runs.csv")
        self.index.parent.mkdir()
        rows = [
            _row("r_ok", "experiments/probe/A/r_ok"),
            _row("r_moved", "experiments/probe/A/r_moved"),
            _row("r_gone", "experiments/probe/A/r_gone"),
            _row("r_wrong_id", "experiments/probe/A/r_wrong_id"),
            _row("r_failed", "experiments/probe/A/r_failed", status="failed"),  # gone, but failed: untouched
        ]
        with self.index.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        self.before = self.index.read_text(encoding="utf-8")

    def _main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(list(argv)), 0)
        return out.getvalue()

    def _rows(self):
        with self.index.open(newline="", encoding="utf-8") as handle:
            return {row["run_id"]: row for row in csv.DictReader(handle)}

    def test_dry_run_reports_and_writes_nothing(self):
        out = self._main()
        self.assertIn("moved     r_moved", out)
        self.assertIn("missing   r_gone", out)
        self.assertIn("missing   r_wrong_id", out)
        self.assertIn("5 rows; moved 1, missing 2, ambiguous 0", out)
        self.assertNotIn("r_failed", out)
        self.assertEqual(self.index.read_text(encoding="utf-8"), self.before)
        self.assertEqual(list(Path("reports").glob("runs.csv.bak-*")), [])

    def test_fix_repoints_moved_marks_missing_keeps_the_rest(self):
        self._main("--fix")
        rows = self._rows()
        moved = rows["r_moved"]
        self.assertEqual(moved["status"], "completed")
        self.assertEqual(moved["run_dir"], "experiments/parity-reference/probe/A/r_moved")
        self.assertEqual(moved["metrics_path"], "experiments/parity-reference/probe/A/r_moved/metrics.json")
        self.assertEqual(moved["visualization_dir"], "experiments/parity-reference/probe/A/r_moved/visualizations")
        self.assertEqual(rows["r_gone"]["status"], "missing")
        self.assertEqual(rows["r_gone"]["run_dir"], "experiments/probe/A/r_gone")
        self.assertEqual(rows["r_wrong_id"]["status"], "missing")
        self.assertEqual(rows["r_ok"], _row("r_ok", "experiments/probe/A/r_ok"))
        self.assertEqual(rows["r_failed"], _row("r_failed", "experiments/probe/A/r_failed", status="failed"))
        self.assertEqual(list(rows), ["r_ok", "r_moved", "r_gone", "r_wrong_id", "r_failed"])
        (backup,) = Path("reports").glob("runs.csv.bak-*")
        self.assertEqual(backup.read_text(encoding="utf-8"), self.before)

    def test_second_fix_is_a_no_op(self):
        self._main("--fix")
        after = self.index.read_text(encoding="utf-8")
        out = self._main("--fix")
        self.assertIn("moved 0, missing 0, ambiguous 0", out)
        self.assertEqual(self.index.read_text(encoding="utf-8"), after)
        self.assertEqual(len(list(Path("reports").glob("runs.csv.bak-*"))), 1)


if __name__ == "__main__":
    unittest.main()
