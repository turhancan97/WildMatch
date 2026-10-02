"""Tests for the GPU-free parts of scripts/export_mined_pairs_demo.py."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_mined_pairs_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_mined_pairs_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load()


class MinedPairsHelpersTests(unittest.TestCase):
    def test_raw_path_from_masked_path(self):
        self.assertEqual(M.raw_rel("CzechLynx_masked/foe_bohemia/lynx_041/22605_lynx_041.jpg"),
                         "CzechLynx/foe_bohemia/lynx_041/22605_lynx_041.jpg")
        with self.assertRaises(ValueError):
            M.raw_rel("CzechLynx/foe_bohemia/x.jpg")

    def test_identity_from_view_frame(self):
        self.assertEqual(M.identity_of("train/lynx_000/foe_carpaths/6969/frame_000000.jpg"), "lynx_000")

    def test_warm_share_counts_warm_animal_pixels_only(self):
        raw = np.zeros((10, 10, 3), np.uint8); raw[:, :5] = (200, 120, 40); raw[:, 5:] = (40, 80, 200)
        masked = np.zeros((10, 10, 3), np.uint8); masked[2:8, :] = raw[2:8, :]
        warm, fg = M.warm_share(Image.fromarray(raw, "RGB"), Image.fromarray(masked, "RGB"))
        self.assertAlmostEqual(fg, 0.6)
        self.assertAlmostEqual(warm, 0.5)

    def test_payload_guard_and_shape(self):
        anchors = [{"identity": "lynx_1", "site": "snpa", "selection_score": 0.1,
                    "anchor": {"tag": "a", "image": {"file": "a.jpg", "width": 1, "height": 1}, "source": "CzechLynx/x.jpg"},
                    "positives": [], "negatives": []}]
        payload = M.build_payload(anchors, {"seed": 1})
        self.assertIn("positives_per_anchor", payload["mining"])
        json.dumps(payload)
        anchors[0]["anchor"]["source"] = "/shared/x.jpg"
        with self.assertRaises(ValueError):
            M.build_payload(anchors, {})


class CommittedMinedPairsTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "mined_pairs" / "mined_pairs.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("mined-pairs demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        self.assertTrue(data["anchors"])
        for a in data["anchors"]:
            self.assertTrue((self.DEMO.parent / a["anchor"]["image"]["file"]).is_file())
            self.assertEqual(len(a["positives"]), 5); self.assertEqual(len(a["negatives"]), 5)
            self.assertTrue(all(p["identity"] == a["identity"] for p in a["positives"]))
            self.assertTrue(all(p["identity"] != a["identity"] for p in a["negatives"]))
            for p in a["positives"] + a["negatives"]:
                self.assertTrue((self.DEMO.parent / p["image"]["file"]).is_file())
                if "points" in p:
                    n = p["match_count"]
                    self.assertEqual(len(p["points"]["anchor"]), n); self.assertEqual(len(p["points"]["partner"]), n)
                    self.assertEqual(sorted(p["order_by_confidence"]), list(range(n)))
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text); self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
