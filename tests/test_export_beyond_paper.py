import json
import unittest
from pathlib import Path

from paper.page import export_beyond_paper as bp

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "beyond"


class CommittedExportTests(unittest.TestCase):
    def setUp(self):
        if not bp.DEFAULT_OUT.is_file():
            self.skipTest("beyond.json not exported")
        self.data = json.loads(bp.DEFAULT_OUT.read_text(encoding="utf-8"))

    def test_rdd_has_every_dataset_series_and_budget(self):
        rdd = self.data["rdd"]
        self.assertEqual(list(rdd["datasets"]), [key for _, _, key, _ in bp.RDD_DATASETS])
        for key, ds in rdd["datasets"].items():
            self.assertEqual(ds["ks"], [10, 50, 100, 160] if key == "czechlynx_unseen" else bp.KS)
            for series in rdd["series_order"]:
                points = ds["series"][series]
                self.assertEqual(sorted(points, key=int), [str(k) for k in ds["ks"]], (key, series))
                for point in points.values():
                    self.assertTrue(point["run_id"])
                    for metric in bp.METRICS:
                        self.assertTrue(0.0 <= point[metric] <= 1.0)

    def test_sam3_pairs_every_rerun_with_a_paper_run(self):
        for key, ds in self.data["sam3"]["datasets"].items():
            self.assertTrue(ds["rows"], key)
            for row in ds["rows"]:
                self.assertTrue(row["paper"]["run_id"])
                self.assertTrue(row["sam3"]["run_id"])
                self.assertNotEqual(row["paper"]["run_id"], row["sam3"]["run_id"])

    def test_jaguar_has_default_and_fine_tuned_matchers(self):
        rows = self.data["jaguar"]["rows"]
        for matcher in ("loma", "rdd-lightglue"):
            for checkpoint in ("default", "fine-tuned"):
                ks = {r["candidate_k"] for r in rows if r["matcher"] == matcher and r["checkpoint"] == checkpoint}
                self.assertEqual(ks, {str(k) for k in bp.KS}, (matcher, checkpoint))

    def test_no_private_paths(self):
        bp.check_no_private_paths(self.data)
        for name in ("jaguar_matches.json", "seastar_masks.json"):
            bp.check_no_private_paths(json.loads((ASSETS / name).read_text(encoding="utf-8")))


class HelperTests(unittest.TestCase):
    def test_private_paths_fail(self):
        with self.assertRaises(ValueError):
            bp.check_no_private_paths({"x": "/home/someone/run"})

    def test_jaguar_sampling_is_seeded_and_ignores_scores(self):
        import numpy as np

        from paper.page import export_jaguar_examples as je

        correct = np.array([True, False, True, True, False, True])
        self.assertEqual(je.sample_queries(correct, 2, 0), je.sample_queries(correct, 2, 0))
        self.assertTrue(set(je.sample_queries(correct, 4, 0)) <= {0, 2, 3, 5})
        rows, cols = np.array([0, 0, 1, 1]), np.array([3, 1, 2, 0])
        values = np.array([0.5, 0.5, 0.2, 0.9])
        self.assertEqual(je.top1(rows, cols, values, 2).tolist(), [1, 0])  # tie -> lowest gallery index


if __name__ == "__main__":
    unittest.main()
