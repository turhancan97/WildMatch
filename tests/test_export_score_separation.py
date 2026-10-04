import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from paper.page import export_score_separation as sep  # noqa: E402


class StatisticsTests(unittest.TestCase):
    def test_auroc_extremes_and_ties(self):
        self.assertEqual(sep.auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])), 1.0)
        self.assertEqual(sep.auroc(np.array([0.1, 0.2]), np.array([0.9, 0.8])), 0.0)
        self.assertAlmostEqual(sep.auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])), 0.5)
        self.assertTrue(math.isnan(sep.auroc(np.array([]), np.array([0.1]))))
        # 3 of 4 ordered pairs favour the positives
        self.assertAlmostEqual(sep.auroc(np.array([0.3, 0.9]), np.array([0.5, 0.1])), 0.75)

    def test_histogram_keeps_upper_edge_and_total(self):
        edges = np.linspace(0, 1, 5)
        counts = sep.histogram(np.array([0.0, 0.1, 0.5, 1.0, 1.0]), edges)
        self.assertEqual(counts.sum(), 5)
        self.assertEqual(counts[-1], 2)
        self.assertEqual(counts[0], 2)

    def test_overlap_coefficient(self):
        a = np.array([10, 0, 0, 0])
        self.assertAlmostEqual(sep.overlap_coefficient(a, a), 1.0)
        self.assertAlmostEqual(sep.overlap_coefficient(a, np.array([0, 0, 0, 7])), 0.0)
        self.assertAlmostEqual(sep.overlap_coefficient(np.array([5, 5]), np.array([10, 0])), 0.5)

    def test_stable_top1_prefers_lower_index_on_ties(self):
        rows = np.array([0, 0, 0, 1, 1])
        cols = np.array([7, 3, 5, 2, 9])
        values = np.array([0.5, 0.5, 0.1, 0.2, 0.9], dtype=np.float32)
        top = sep.stable_top1(rows, cols, values, n_query=3)
        self.assertEqual(top.tolist(), [3, 9, -1])

    def test_per_query_margins(self):
        rows = np.array([0, 0, 0, 1, 1, 2])
        values = np.array([0.8, 0.3, 0.6, 0.2, 0.7, 0.4], dtype=np.float32)
        same = np.array([True, False, False, True, False, False])
        margin = sep.per_query_margins(rows, values, same, n_query=3)
        self.assertAlmostEqual(margin[0], 0.8 - 0.6, places=6)
        self.assertAlmostEqual(margin[1], 0.2 - 0.7, places=6)
        self.assertTrue(np.isnan(margin[2]))  # no same-individual candidate

    def test_summarise_counts_add_up(self):
        rows = np.repeat(np.arange(4), 3)
        cols = np.tile(np.arange(3), 4)
        values = np.linspace(0.05, 0.95, 12).astype(np.float32)
        same = cols == 0
        edges = np.linspace(0, 1, 11)
        result = sep.summarise(rows, cols, values, same, 4, edges, np.linspace(-1, 1, 5))
        self.assertEqual(sum(result["same"]), int(same.sum()))
        self.assertEqual(sum(result["different"]), int((~same).sum()))
        self.assertEqual(sum(result["margin"]), result["stats"]["n_queries_with_margin"])
        self.assertEqual(result["stats"]["n_queries_with_margin"], 4)


class CommittedExportTests(unittest.TestCase):
    PATH = sep.DEFAULT_OUT / "score_separation.json"

    def setUp(self):
        if not self.PATH.is_file():
            self.skipTest("score_separation.json not exported")
        self.data = json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_shape_and_population(self):
        self.assertEqual(self.data["k"], sep.MAIN_K)
        self.assertEqual([m["key"] for m in self.data["matchers"]], ["default", "finetuned"])
        self.assertEqual([d["key"] for d in self.data["datasets"]], [key for _, key, _ in sep.DATASETS])
        bins = len(self.data["bin_edges"]) - 1
        for d in self.data["datasets"]:
            for mkey in ("default", "finetuned"):
                side = d["matchers"][mkey]
                self.assertEqual(len(side["same"]), bins)
                self.assertEqual(sum(side["same"]), side["stats"]["n_same"])
                self.assertEqual(sum(side["different"]), side["stats"]["n_different"])
                self.assertEqual(side["stats"]["n_same"] + side["stats"]["n_different"], d["runs"][mkey]["n_pairs"])
                self.assertLessEqual(d["runs"][mkey]["n_pairs"], d["n_queries"] * d["k"])
                self.assertTrue(0.0 <= side["stats"]["auroc"] <= 1.0)
                self.assertTrue(0.0 <= side["stats"]["overlap"] <= 1.0)
                self.assertAlmostEqual(d["runs"][mkey]["top_1"], d["runs"][mkey]["top_1_recomputed"], places=9)
            # the same shortlist is scored by both matchers
            self.assertEqual(d["matchers"]["default"]["stats"]["n_same"], d["matchers"]["finetuned"]["stats"]["n_same"])

    def test_fine_tuning_separates_scores_on_every_dataset(self):
        for d in self.data["datasets"]:
            with self.subTest(dataset=d["key"]):
                default, finetuned = d["matchers"]["default"]["stats"], d["matchers"]["finetuned"]["stats"]
                self.assertGreater(finetuned["auroc"], default["auroc"])
                self.assertLess(finetuned["overlap"], default["overlap"])

    def test_no_private_paths(self):
        text = self.PATH.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text)
        self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
