"""Tests for the GPU-free parts of paper/page/export_rank_change_demo.py."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "paper/page" / "export_rank_change_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_rank_change_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load()


class RankChangeHelpersTests(unittest.TestCase):
    def test_true_rank_and_categories(self):
        labels = np.array(["a", "b", "c", "a"])
        self.assertEqual(R.true_rank(np.array([1, 3, 0, 2]), labels, "a"), 2)
        self.assertIsNone(R.true_rank(np.array([1, 2]), labels, "a"))
        self.assertEqual(R.categorize(False, False, True), "rescued")
        self.assertEqual(R.categorize(True, True, False), "regressed")
        self.assertEqual(R.categorize(False, True, True), "already_right")
        self.assertEqual(R.categorize(True, False, False), "still_wrong")

    def test_shortlist_rankings_use_stable_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            # query 0: gallery 2 scores 0.9, gallery 1 and 3 tie at 0.5 (lower index first); query 1: nothing scored
            np.savez(run / "scores.npz", shape=np.array([2, 4]), rows=np.array([0, 0, 0]), cols=np.array([3, 1, 2]),
                     values=np.array([0.5, 0.5, 0.9], dtype=np.float32))
            orders = R.shortlist_rankings(run, 2, 4)
            self.assertEqual(orders[0].tolist(), [2, 1, 3])
            self.assertEqual(len(orders[1]), 0)
            with self.assertRaises(ValueError):
                R.shortlist_rankings(run, 3, 4)

    def test_payload_guard(self):
        payload = R.build_payload([], {}, {"total": 0}, 250)
        self.assertEqual(payload["k"], 250)
        json.dumps(payload)
        with self.assertRaises(ValueError):
            R.build_payload([{"tag": "x", "rankings": {}, "identity": "/shared/x"}], {}, {"total": 0}, 250)


class CommittedRankChangeTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "rank_change" / "rank_change.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("rank-change demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        c = data["counts"]
        self.assertEqual(c["rescued"] + c["still_wrong"] + c["regressed"] + c["already_right"], c["total"])
        self.assertEqual(c["finetuned_top1"], c["rescued"] + c["already_right"])
        self.assertEqual(c["default_top1"], c["regressed"] + c["already_right"])
        for q in data["queries"]:
            self.assertIn(q["tag"], data["photos"])
            self.assertTrue((self.DEMO.parent / data["photos"][q["tag"]]["file"]).is_file())
            for m in ("cosine", "default", "finetuned"):
                r = q["rankings"][m]
                self.assertEqual(len(r["top5"]), 5)
                self.assertEqual(r["true_rank"] == 1, r["top5"][0]["correct"])
                for e in r["top5"]:
                    self.assertTrue((self.DEMO.parent / data["photos"][e["tag"]]["file"]).is_file())
                    self.assertEqual(e["correct"], e["identity"] == q["identity"])
            ft_ok, def_ok = q["rankings"]["finetuned"]["true_rank"] == 1, q["rankings"]["default"]["true_rank"] == 1
            self.assertEqual(q["category"], R.categorize(q["rankings"]["cosine"]["true_rank"] == 1, def_ok, ft_ok))
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text); self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
