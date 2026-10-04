"""The table and figure exporters keep the paper's runs apart from runs on the SAM 3 inputs."""

import json
import tempfile
import unittest
from pathlib import Path

from wildmatch.reporting import export_figures, export_tables
from wildmatch.reporting.paper_tables import (
    discover_records,
    filter_records_by_inputs,
    registry_input_tables,
    select_latest_records,
)

TABLES = {"HyenaID2022": {"paper": "metadata_old/hyena.csv", "current": "metadata_sam3/hyena.csv"}}


def _write_run(root: Path, animal: str, run_utc: str, metadata_file: str | None, top_1: float) -> None:
    run = root / "probe" / "WildlifeReID-10k" / animal / "split" / "megadescriptor-l" / "cosine" / "default" / run_utc
    run.mkdir(parents=True)
    manifest = {
        "status": "completed",
        "workflow": "probe",
        "method": "cosine",
        "animal": animal,
        "split_protocol": "split",
        "model": "megadescriptor-l",
        "run_id": run_utc,
        "run_utc": run_utc,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({"top_1": top_1}), encoding="utf-8")
    if metadata_file is not None:
        (run / "config.snapshot.yaml").write_text(f"dataset:\n  metadata_file: {metadata_file}\n", encoding="utf-8")


class ExportInputsTests(unittest.TestCase):
    def test_paper_runs_are_not_replaced_by_newer_sam3_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "HyenaID2022", "20260901T000000Z", "metadata_old/hyena.csv", 0.5)
            _write_run(root, "HyenaID2022", "20261004T000000Z", "metadata_sam3/hyena.csv", 0.6)
            _write_run(root, "LeopardID2022", "20260901T000000Z", "anything.csv", 0.7)
            records = discover_records(root)
            self.assertEqual(
                sorted(record["metadata_file"] for record in records),
                ["anything.csv", "metadata_old/hyena.csv", "metadata_sam3/hyena.csv"],
            )

            def top_1(inputs):
                kept = filter_records_by_inputs(records, inputs, TABLES)
                return [record["top_1"] for record in select_latest_records(kept, "HyenaID2022")]

            self.assertEqual(top_1("paper"), [0.5])
            self.assertEqual(top_1("current"), [0.6])
            self.assertEqual(top_1("all"), [0.6])  # the old behaviour: newest run wins
            # animals without two input tables are never filtered
            for inputs in ("paper", "current"):
                kept = filter_records_by_inputs(records, inputs, TABLES)
                self.assertEqual([r["top_1"] for r in select_latest_records(kept, "LeopardID2022")], [0.7])
            with self.assertRaises(ValueError):
                filter_records_by_inputs(records, "sam3", TABLES)

    def test_run_without_config_snapshot_has_no_metadata_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "HyenaID2022", "20260901T000000Z", None, 0.5)
            (record,) = discover_records(root)
            self.assertEqual(record["metadata_file"], "")
            self.assertEqual(filter_records_by_inputs([record], "paper", TABLES), [])

    def test_registry_lists_the_wildlifereid10k_animals(self):
        tables = registry_input_tables()
        self.assertEqual(len(tables), 12)
        for animal, table in tables.items():
            self.assertNotEqual(table["paper"], table["current"], animal)
            self.assertTrue(table["current"].startswith("metadata_sam3/"), animal)

    def test_commands_default_to_paper_inputs(self):
        self.assertEqual(export_tables.parse_args([]).inputs, "paper")
        self.assertEqual(export_figures.parse_args([]).inputs, "paper")
        self.assertEqual(export_tables.parse_args(["--inputs", "current"]).inputs, "current")


if __name__ == "__main__":
    unittest.main()
