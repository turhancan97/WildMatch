import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from paper.page import export_frequency_bins as fb  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_bin_index_fixed_edges(self):
        counts = np.array([0, 1, 2, 4, 5, 9, 10, 29, 30, 400])
        self.assertEqual(fb.bin_index(counts).tolist(), [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_stable_topn_order_and_padding(self):
        rows = np.array([0, 0, 0, 1])
        cols = np.array([7, 3, 5, 2])
        values = np.array([0.5, 0.5, 0.9, 0.1], dtype=np.float32)
        top = fb.stable_topn(rows, cols, values, n_query=3, n=2)
        self.assertEqual(top.tolist(), [[5, 3], [2, -1], [-1, -1]])

    def test_correctness(self):
        top = np.array([[0, 1], [2, -1], [-1, -1]])
        db = np.array(["a", "b", "c"])
        q = np.array(["b", "c", "a"])
        self.assertEqual(fb.correctness(top, db, q).tolist(), [True, True, False])
        self.assertEqual(fb.correctness(top[:, :1], db, q).tolist(), [False, True, False])

    def test_bootstrap_interval_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        default = rng.random(300) < 0.4
        finetuned = default | (rng.random(300) < 0.2)
        lo, hi = fb.bootstrap_gain(default, finetuned, n_resamples=500, seed=3)
        gain = finetuned.mean() - default.mean()
        self.assertLessEqual(lo, gain)
        self.assertGreaterEqual(hi, gain)
        self.assertGreater(lo, 0.0)
        self.assertTrue(np.isnan(fb.bootstrap_gain(np.array([]), np.array([]))[0]))


class CommittedExportTests(unittest.TestCase):
    PATH = fb.DEFAULT_OUT

    def setUp(self):
        if not self.PATH.is_file():
            self.skipTest("frequency_bins.json not exported")
        self.data = json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_bins_are_consistent(self):
        self.assertEqual(self.data["bin_labels"], fb.BIN_LABELS)
        self.assertEqual([d["key"] for d in self.data["datasets"]], [key for _, key, _ in fb.DATASETS])
        for d in self.data["datasets"]:
            with self.subTest(dataset=d["key"]):
                total = sum(b["n"] for b in d["bins"])
                self.assertEqual(total + d["n_queries_without_gallery_image"], d["n_queries"])
                self.assertEqual(d["overall"]["n"], total)
                for b in d["bins"] + [d["overall"]]:
                    if not b["n"]:
                        continue
                    for who in ("default", "finetuned"):
                        self.assertLessEqual(b["top_1"][who], b["top_5"][who] + 1e-12)
                        self.assertLessEqual(b["top_5"][who], b["shortlist_share"] + 1e-12)
                    for metric in ("top_1", "top_5"):
                        lo, hi = b[metric]["gain_ci95"]
                        self.assertLessEqual(lo, b[metric]["gain"] + 1e-12)
                        self.assertGreaterEqual(hi, b[metric]["gain"] - 1e-12)
                # the overall Top-1 over queries with a gallery image reproduces the recorded Top-1 when no query
                # lacks a gallery image; otherwise the recorded value counts those misses too
                for who in ("default", "finetuned"):
                    recorded = d["runs"][who]["top_1"]
                    reproduced = d["overall"]["top_1"][who] * d["overall"]["n"] / d["n_queries"]
                    self.assertAlmostEqual(reproduced, recorded, places=9)

    def test_no_private_paths(self):
        text = self.PATH.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text)
        self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
