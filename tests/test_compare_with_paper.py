"""paper/tools/compare_with_paper.py on synthetic run directories and paper tables."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from paper.tools import compare_with_paper as C


def _run(root: Path, name: str, method: str, top_1: float, **methods) -> Path:
    run = root / name
    run.mkdir(parents=True)
    config = {
        "dataset": {"animal": "HyenaID2022", "split_col": "split"},
        "model": {"type": "megadescriptor-l"},
        "benchmark": {"method": method, "candidate_k": 250, "methods": methods},
    }
    (run / "config.snapshot.yaml").write_text(yaml.safe_dump(config))
    (run / "metrics.json").write_text(json.dumps({"top_1": top_1, "top_5": 0.9, "balanced_top_1": 0.8}))
    return run


class CompareWithPaperTests(unittest.TestCase):
    def test_identity_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = [
                "method",
                "matcher",
                "backbone",
                "checkpoint",
                "class_weighting",
                "candidate_k",
                "top_1",
                "top_5",
                "balanced_top_1",
                "run_id",
            ]
            rows = [
                ["Vismatch", "loma", "megadescriptor-l", "default", "", "250", "0.5", "0.9", "0.8", "p1"],
                ["Vismatch", "loma", "megadescriptor-l", "fine-tuned", "", "250", "0.6", "0.9", "0.8", "p2"],
                ["Vismatch", "loma", "megadescriptor-l", "fine-tuned", "", "50", "0.1", "0.9", "0.8", "p3"],
                [
                    "Linear Probe",
                    "-",
                    "megadescriptor-l",
                    "frozen (weighted)",
                    "weighted",
                    "",
                    "0.3",
                    "0.9",
                    "0.8",
                    "p4",
                ],
                ["Cosine", "-", "megadescriptor-l", "default", "", "", "0.2", "0.9", "0.8", "p5"],
            ]
            results = root / "results"
            results.mkdir()
            with (results / "HyenaID2022_split_ablation.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            runs = [
                _run(root, "a", "vismatch", 0.65, vismatch={"matcher": "loma", "checkpoint_source": "custom"}),
                _run(
                    root,
                    "b",
                    "linear_probe",
                    0.31,
                    linear_probe={"train_mode": "classifier", "class_weighting": "inverse_frequency"},
                ),
                _run(root, "c", "cosine", 0.2),
                _run(root, "d", "vismatch", 0.4, vismatch={"matcher": "rdd-lightglue", "checkpoint_source": "default"}),
            ]
            table = C.compare(runs, results)
            self.assertEqual([row["paper_run"] for row in table], ["p2", "p4", "p5", ""])
            self.assertEqual(table[0]["delta_top_1"], 5.0)
            self.assertEqual(table[1]["checkpoint"], "frozen (weighted)")
            self.assertEqual(table[2]["candidate_k"], "")  # full-gallery methods have no budget
            self.assertIn("| HyenaID2022 | Vismatch loma fine-tuned | 250 |", C.render(table))


if __name__ == "__main__":
    unittest.main()
