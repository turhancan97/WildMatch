"""Tests for the GPU-free parts of scripts/export_before_after_demo.py."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_before_after_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_before_after_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B = _load()


class BeforeAfterHelpersTests(unittest.TestCase):
    def test_match_record_maps_points_and_orders_by_confidence(self):
        result = SimpleNamespace(score=0.5, match_count=2, confidences=np.array([0.2, 0.9]),
                                 matched_kpts0=np.array([[-1.0, -1.0], [0.0, 0.0]]), matched_kpts1=np.array([[1.0, 1.0], [0.0, 0.0]]))
        rec = B.match_record(result, (100, 50), (200, 80))
        self.assertEqual(rec["match_count"], 2)
        self.assertEqual(rec["points"]["query"][0], [-0.5, -0.5])
        self.assertEqual(rec["points"]["gallery"][0], [199.5, 79.5])
        self.assertEqual(rec["order_by_confidence"], [1, 0])

    def test_payload_shape_and_private_path_guard(self):
        pair = {"dataset": "hyena", "label": "Hyena", "query": {"image": {"file": "hyena_query.jpg", "width": 10, "height": 10}, "source": "x"},
                "gallery": {"image": {"file": "hyena_gallery.jpg", "width": 10, "height": 10}, "source": "y"},
                "checkpoint_sha256": "abc", "probe_score": 0.6,
                "results": {"default": {"score": 0.1, "match_count": 0, "points": {"query": [], "gallery": []}, "confidence": [], "order_by_confidence": []},
                            "finetuned": {"score": 0.6, "match_count": 0, "points": {"query": [], "gallery": []}, "confidence": [], "order_by_confidence": []}}}
        payload = B.build_payload([pair], {"loma_arch": "LoMa-B"})
        self.assertEqual(set(payload["matchers"]), {"default", "finetuned"})
        self.assertIn("background-removed", payload["inputs"])
        json.dumps(payload)
        pair["query"]["source"] = "/shared/secret.jpg"
        with self.assertRaises(ValueError):
            B.build_payload([pair], {})


class CommittedBeforeAfterTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "before_after" / "before_after.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("before/after demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        self.assertTrue(data["pairs"])
        for p in data["pairs"]:
            for side in ("query", "gallery"):
                self.assertTrue((self.DEMO.parent / p[side]["image"]["file"]).is_file())
            for key in ("default", "finetuned"):
                r = p["results"][key]
                n = r["match_count"]
                self.assertEqual(len(r["points"]["query"]), n)
                self.assertEqual(len(r["points"]["gallery"]), n)
                self.assertEqual(len(r["confidence"]), n)
                self.assertEqual(sorted(r["order_by_confidence"]), list(range(n)))
                for side in ("query", "gallery"):
                    size = p[side]["image"]
                    for x, y in r["points"][side]:
                        self.assertTrue(-1 <= x <= size["width"] and -1 <= y <= size["height"], (p["dataset"], key, side))
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text); self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
