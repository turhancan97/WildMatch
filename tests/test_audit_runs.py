import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from wildmatch.reporting.artifacts import ARTIFACT_SCHEMA_VERSION
from wildmatch.reporting.audit_runs import MANIFEST_GAPS, audit_index, audit_manifests, main, manifest_gaps

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "legacy_experiments"

COMPLETE = {
    "schema_version": ARTIFACT_SCHEMA_VERSION,
    "status": "completed",
    "run_id": "r1",
    "workflow": "probe",
    "git_commit": "abc",
    "code": {"commit": "abc", "dirty": False},
    "model": "megadescriptor-l",
    "split_protocol": "split",
    "method": "vismatch",
    "timings": {"primary_compute_runtime_sec": 1.0},
    "checkpoint": None,
    "checkpoint_source": "custom",
    "vismatch_checkpoint": {"components": [{"sha256": "f" * 64}]},
    "vismatch_device": "cuda:NVIDIA GeForce RTX 4090:sm89",
}


class AuditRunsTest(unittest.TestCase):
    def test_complete_manifest_has_no_gaps(self):
        self.assertEqual(manifest_gaps(COMPLETE), [])

    def test_old_manifest_gaps(self):
        old = {key: value for key, value in COMPLETE.items() if key not in {"code", "vismatch_device", "model"}}
        old["timings"] = {}
        old["vismatch_checkpoint"] = {"components": [{"path": "x"}]}
        self.assertEqual(
            manifest_gaps(old), ["code_identity", "model", "primary_timing", "checkpoint_sha256", "vismatch_device"]
        )
        self.assertTrue(set(manifest_gaps(old)) <= set(MANIFEST_GAPS))

    def test_fixture_runs_and_unreadable_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiments"
            run = root / "probe" / "a" / "run1"
            run.mkdir(parents=True)
            (run / "run_manifest.json").write_text("")
            records = {record["run_id"]: record for record in audit_manifests(root)}
            self.assertEqual(records["run1"]["gaps"], ["unreadable"])
        records = audit_manifests(FIXTURES)
        self.assertEqual(len(records), 1)
        self.assertIn("code_identity", records[0]["gaps"])

    def test_index_report_and_cli_write_nothing_but_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "runs.csv"
            with index.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "model", "git_commit"])
                writer.writeheader()
                writer.writerow({"run_id": "a", "status": "completed", "model": "m", "git_commit": ""})
                writer.writerow({"run_id": "b", "status": "failed", "model": "", "git_commit": ""})
            report = audit_index(index)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["status"], {"completed": 1, "failed": 1})
            self.assertEqual(report["empty"], {"model": 1, "git_commit": 2})
            self.assertEqual(report["absent_columns"], ["split_protocol", "primary_compute_runtime_sec"])
            before = {path: os.stat(path).st_mtime_ns for path in FIXTURES.rglob("*") if path.is_file()}
            before[index] = os.stat(index).st_mtime_ns
            output = Path(tmp) / "audit.csv"
            self.assertEqual(main(["--root", str(FIXTURES), "--index", str(index), "--output", str(output)]), 0)
            self.assertEqual(before, {path: os.stat(path).st_mtime_ns for path in before})
            rows = list(csv.DictReader(output.open()))
            self.assertEqual(len(rows), 1)
            self.assertIn("code_identity", rows[0]["gaps"].split(";"))

    def test_manifest_json_is_valid_after_atomic_write(self):
        from wildmatch.utils.io import atomic_write

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            atomic_write(path, lambda temporary: temporary.write_text(json.dumps(COMPLETE)))
            self.assertEqual(json.loads(path.read_text()), COMPLETE)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["run_manifest.json"])


if __name__ == "__main__":
    unittest.main()
