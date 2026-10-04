"""Tests for paper/tools/parity_check.py on synthetic run directories."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import yaml

from wildmatch.evaluate.ranking import stable_rank_1d

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("parity_check", REPO_ROOT / "paper" / "tools" / "parity_check.py")
PC = importlib.util.module_from_spec(_SPEC)
sys.modules["parity_check"] = PC  # dataclasses resolve their module through sys.modules
_SPEC.loader.exec_module(PC)


def _write_run(
    root: Path,
    run_id: str,
    *,
    method="vismatch",
    variant="rdd-lightglue",
    k=50,
    values=None,
    rows=None,
    cols=None,
    metrics=None,
    status="completed",
):
    run_dir = root / "SalamanderID2025" / "split" / method / variant / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": status,
                "run_id": run_id,
                "dataset": "SalamanderID2025",
                "animal": "SalamanderID2025",
                "split_protocol": "split",
                "model": "megadescriptor-l",
                "method": method,
                "variant": variant,
                "checkpoint_variant": "default",
            }
        )
    )
    (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump({"benchmark": {"candidate_k": k, "methods": {}}}))
    (run_dir / "metrics.json").write_text(json.dumps(metrics or {"top_1": 0.5, "top_5": 0.8}))
    rows = np.array([0, 0, 1, 1]) if rows is None else rows
    cols = np.array([0, 1, 0, 2]) if cols is None else cols
    values = np.array([0.9, 0.2, 0.1, 0.7]) if values is None else values
    np.savez(run_dir / "scores.npz", shape=np.array([2, 3]), rows=rows, cols=cols, values=values)
    return run_dir


def _run(reference: Path, candidate: Path) -> int:
    with redirect_stdout(io.StringIO()):
        return PC.run(reference, candidate, None)


class ParityCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ref = Path(self.tmp.name) / "ref"
        self.new = Path(self.tmp.name) / "new"

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_runs_pass(self):
        _write_run(self.ref, "20260101T000000Z_a")
        _write_run(self.new, "20260102T000000Z_b")
        self.assertEqual(_run(self.ref, self.new), 0)

    def test_rdd_score_shift_above_tolerance_fails(self):
        _write_run(self.ref, "20260101T000000Z_a")
        _write_run(self.new, "20260102T000000Z_b", values=np.array([0.9, 0.2, 0.1, 0.701]))
        self.assertEqual(_run(self.ref, self.new), 1)

    def test_loma_has_a_looser_score_tolerance(self):
        _write_run(self.ref, "20260101T000000Z_a", variant="loma")
        _write_run(self.new, "20260102T000000Z_b", variant="loma", values=np.array([0.9, 0.2, 0.1, 0.705]))
        self.assertEqual(_run(self.ref, self.new), 0)

    def test_changed_shortlist_fails(self):
        _write_run(self.ref, "20260101T000000Z_a")
        _write_run(self.new, "20260102T000000Z_b", cols=np.array([0, 2, 0, 2]))
        self.assertEqual(_run(self.ref, self.new), 1)

    def test_metric_difference_fails_and_probe_tolerance_is_looser(self):
        _write_run(self.ref, "20260101T000000Z_a", metrics={"top_1": 0.5})
        _write_run(self.new, "20260102T000000Z_b", metrics={"top_1": 0.505})
        self.assertEqual(_run(self.ref, self.new), 1)
        ref2, new2 = Path(self.tmp.name) / "ref2", Path(self.tmp.name) / "new2"
        _write_run(ref2, "20260101T000000Z_a", method="linear_probe", variant="default", metrics={"top_1": 0.5})
        _write_run(
            new2,
            "20260102T000000Z_b",
            method="linear_probe",
            variant="default",
            metrics={"top_1": 0.505},
            values=np.array([0.1, 0.9, 0.5, 0.4]),
        )
        self.assertEqual(_run(ref2, new2), 0)

    def test_missing_reference_and_newest_run_selection(self):
        _write_run(self.ref, "20260101T000000Z_a", k=50)
        _write_run(self.new, "20260102T000000Z_b", k=100)
        self.assertEqual(_run(self.ref, self.new), 1)
        runs = PC.collect(self.ref)
        _write_run(self.ref, "20260105T000000Z_c", k=50)
        _write_run(self.ref, "20260106T000000Z_d", k=50, status="failed")
        newest = PC.collect(self.ref)
        self.assertEqual(len(runs), 1)
        self.assertEqual([r.run_id for r in newest.values()], ["20260105T000000Z_c"])

    def test_top1_rule_matches_the_shared_stable_ranking(self):
        rng = np.random.default_rng(0)
        dense = rng.integers(0, 3, size=(20, 7)).astype(float)  # many ties
        dense[3] = -np.inf
        rows, cols = np.nonzero(np.isfinite(dense))
        top1 = PC._top1(dense.shape, rows, cols, dense[rows, cols])
        expected = [int(stable_rank_1d(row)[0]) for row in dense]
        self.assertEqual(list(top1), expected)


if __name__ == "__main__":
    unittest.main()
