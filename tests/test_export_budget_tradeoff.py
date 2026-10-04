import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from paper.page import export_budget_tradeoff as bt  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_normalisations(self):
        self.assertAlmostEqual(bt.minutes_per_1000(600.0, 2000), 5.0)
        self.assertAlmostEqual(bt.per_pair_ms(2.0, 100, 10), 2.0)

    def test_gpu_labels(self):
        self.assertEqual(bt.gpu_label("rtx4090_batch"), "RTX 4090")
        self.assertEqual(bt.gpu_label("dgxh100"), "H100")
        self.assertEqual(bt.gpu_label(None), bt.NOT_RECORDED)
        self.assertEqual(bt.gpu_label("mystery"), "mystery")

    def test_choose_display_same_partition_averages(self):
        display = bt.choose_display(
            {"default": 10.0, "finetuned": 12.0},
            {"default": "rtx4090_batch", "finetuned": "rtx4090_batch"},
            "rtx4090_batch",
        )
        self.assertAlmostEqual(display["matching_sec"], 11.0)
        self.assertEqual(display["partition"], "rtx4090_batch")

    def test_choose_display_prefers_dominant_partition(self):
        display = bt.choose_display(
            {"default": 10.0, "finetuned": 30.0}, {"default": "rtx4090_batch", "finetuned": "dgxa100"}, "rtx4090_batch"
        )
        self.assertEqual(display["matching_sec"], 10.0)
        self.assertEqual(display["source"], "default run")
        display = bt.choose_display(
            {"default": 10.0, "finetuned": 30.0}, {"default": None, "finetuned": "rtx4090_batch"}, "rtx4090_batch"
        )
        self.assertEqual(display["matching_sec"], 30.0)

    def test_job_lookup_missing_index(self):
        self.assertEqual(bt.job_lookup(Path("/nonexistent/index.csv")), {})


class CommittedExportTests(unittest.TestCase):
    PATH = bt.DEFAULT_OUT

    def setUp(self):
        if not self.PATH.is_file():
            self.skipTest("budget_tradeoff.json not exported")
        self.data = json.loads(self.PATH.read_text(encoding="utf-8"))

    def test_structure_and_monotone_shortlist(self):
        self.assertEqual(self.data["budgets"], bt.BUDGETS)
        self.assertEqual([d["key"] for d in self.data["datasets"]], [key for _, key, _ in bt.DATASETS])
        for d in self.data["datasets"]:
            ks = [b["k"] for b in d["budgets"]]
            self.assertEqual(ks, bt.BUDGETS)
            shares = [b["shortlist_share"] for b in d["budgets"]]
            self.assertEqual(shares, sorted(shares), f"{d['key']}: shortlist share must grow with k")
            for b in d["budgets"]:
                for m in ("default", "finetuned"):
                    # Top-5 cannot exceed the shortlist share: a correct identity outside the shortlist cannot rank
                    self.assertLessEqual(b["top_5"][m], b["shortlist_share"] + 1e-9, f"{d['key']} k={b['k']} {m}")
                    self.assertGreater(b["matching_sec"][m], 0)
                disp = b["display"]
                self.assertAlmostEqual(
                    disp["minutes_per_1000_queries"], bt.minutes_per_1000(disp["matching_sec"], d["n_queries"])
                )
                self.assertAlmostEqual(
                    disp["ms_per_pair"], bt.per_pair_ms(disp["matching_sec"], d["n_queries"], b["k"])
                )
                self.assertEqual(disp["pairs"], d["n_queries"] * b["k"])
                self.assertTrue(disp["gpu"])
            minutes = [b["display"]["minutes_per_1000_queries"] for b in d["budgets"]]
            self.assertEqual(minutes, sorted(minutes), f"{d['key']}: matching time must grow with k")

    def test_no_private_paths(self):
        text = self.PATH.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text)
        self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
