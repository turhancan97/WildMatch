"""Tests for the GPU-free parts of scripts/export_synthetic_demo.py."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_synthetic_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_synthetic_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = _load()


class SyntheticDemoHelpersTests(unittest.TestCase):
    def test_identity_parsing(self):
        self.assertEqual(D.identity_of("CzechLynx_Synthetic/synthetic/synthetic_lynx_173/09954_synthetic_lynx_173.jpg"), "lynx_173")
        with self.assertRaises(ValueError):
            D.identity_of("not/a/render.jpg")

    def test_mask_background_blackens_only_the_flat_colour(self):
        array = np.full((8, 8, 3), D.TEASER_BACKGROUND, dtype=np.uint8)
        array[2:5, 2:5] = (120, 90, 40)      # animal
        array[0, 0] = (240, 240, 236)        # within tolerance -> background
        array[7, 7] = (200, 200, 200)        # outside tolerance -> kept
        out = np.asarray(D.mask_background(Image.fromarray(array, "RGB")))
        self.assertTrue((out[2:5, 2:5] == (120, 90, 40)).all())
        self.assertTrue((out[0, 0] == 0).all())
        self.assertTrue((out[7, 7] == (200, 200, 200)).all())
        self.assertTrue((out[6, 1] == 0).all())

    def test_normalized_to_pixels_uses_half_pixel_convention(self):
        pts = D.normalized_to_pixels(np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]), (520, 400))
        np.testing.assert_allclose(pts[0], [-0.5, -0.5])
        np.testing.assert_allclose(pts[1], [519.5, 399.5])
        np.testing.assert_allclose(pts[2], [259.5, 199.5])

    def test_ranking_and_payload(self):
        def result(score, confs):
            return SimpleNamespace(score=score, match_count=len(confs), confidences=np.array(confs),
                                   matched_kpts0=None, matched_kpts1=None)
        kq = np.array([[1.0, 2.0], [3.0, 4.0]]); kg = np.array([[5.0, 6.0], [7.0, 8.0]])
        cands = [
            D.candidate_record("L129_0", "x/synthetic_lynx_129/a.jpg", "lynx_173", (520, 520), result(0.02, [0.3, 0.9]), kq, kg),
            D.candidate_record("L173_1", "x/synthetic_lynx_173/b.jpg", "lynx_173", (520, 520), result(0.4, [0.5, 0.8]), kq, kg),
        ]
        payload = D.build_payload({"tag": "L173_0", "identity": "lynx_173", "image": {"file": "L173_0.jpg", "width": 520, "height": 520}},
                                  cands, {"checkpoint_sha256": "abc"})
        ranked = payload["candidates"]
        self.assertEqual([c["tag"] for c in ranked], ["L173_1", "L129_0"])
        self.assertEqual([c["rank"] for c in ranked], [1, 2])
        self.assertTrue(ranked[0]["same_individual"]); self.assertFalse(ranked[1]["same_individual"])
        self.assertEqual(ranked[1]["order_by_confidence"], [1, 0])
        self.assertTrue(payload["synthetic"])
        self.assertIn("CC BY 4.0", payload["attribution"])
        json.dumps(payload)

    def test_payload_rejects_private_paths(self):
        with self.assertRaises(ValueError):
            D.build_payload({"tag": "q", "identity": "lynx_1", "source": "/shared/x.jpg", "image": {}}, [], {})


class CommittedDemoTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "synthetic" / "synthetic_demo.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("synthetic demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        self.assertTrue(data["synthetic"])
        self.assertIn("Picek", data["attribution"])
        self.assertTrue((self.DEMO.parent / data["query"]["image"]["file"]).is_file())
        scores = [c["score"] for c in data["candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for c in data["candidates"]:
            self.assertTrue((self.DEMO.parent / c["image"]["file"]).is_file())
            n = c["match_count"]
            self.assertEqual(len(c["points"]["query"]), n)
            self.assertEqual(len(c["points"]["gallery"]), n)
            self.assertEqual(len(c["confidence"]), n)
            self.assertEqual(sorted(c["order_by_confidence"]), list(range(n)))
            for side in ("query", "gallery"):
                size = data["query"]["image"] if side == "query" else c["image"]
                for x, y in c["points"][side]:
                    self.assertTrue(-1 <= x <= size["width"] and -1 <= y <= size["height"], (c["tag"], side, x, y))
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text); self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
