"""Tests for the GPU-free parts of scripts/export_synthetic_demo.py."""

import importlib.util
import json
import sys
import unittest
import warnings
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

    def test_apply_mask_blackens_outside_the_mask(self):
        array = np.full((6, 5, 3), 200, dtype=np.uint8)
        mask = np.zeros((6, 5), dtype=np.uint8); mask[1:4, 1:3] = 1
        out = np.asarray(D.apply_mask(Image.fromarray(array, "RGB"), mask))
        self.assertTrue((out[1:4, 1:3] == 200).all())
        self.assertEqual(int(out.sum()), 200 * 3 * 6)
        with self.assertRaises(ValueError):
            D.apply_mask(Image.fromarray(array, "RGB"), np.zeros((5, 6), dtype=np.uint8))

    def test_decode_rle_round_trip(self):
        # pycocotools 2.x calls np.array(copy=False) internally; NumPy 2 warns. Not our code.
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pycocotools\.")
        from pycocotools import mask as mask_utils
        mask = np.zeros((7, 9), dtype=np.uint8); mask[2:5, 3:8] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        decoded = D.decode_rle(json.dumps(rle))
        np.testing.assert_array_equal(decoded, mask)

    def test_load_rows_fails_closed_on_missing_render(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta.csv").write_text("source,path,mask\nsynthetic,a/b.jpg,{}\n", encoding="utf-8")
            self.assertEqual(set(D.load_rows(root, "meta.csv", ["a/b.jpg"])), {"a/b.jpg"})
            with self.assertRaises(FileNotFoundError):
                D.load_rows(root, "meta.csv", ["a/b.jpg", "missing.jpg"])

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
            D.candidate_record("G_lynx_129", "x/synthetic_lynx_129/a.jpg", "lynx_173", (520, 520), result(0.02, [0.3, 0.9]), kq, kg),
            D.candidate_record("G_lynx_173", "x/synthetic_lynx_173/b.jpg", "lynx_173", (520, 520), result(0.4, [0.5, 0.8]), kq, kg),
        ]
        queries = [{"tag": "Q_lynx_173", "identity": "lynx_173", "image": {"file": "Q_lynx_173.jpg", "width": 520, "height": 520},
                    "candidates": cands}]
        gallery = [{"tag": "G_lynx_173", "identity": "lynx_173", "image": {}}, {"tag": "G_lynx_129", "identity": "lynx_129", "image": {}}]
        payload = D.build_payload(queries, gallery, {"checkpoint_sha256": "abc"})
        ranked = payload["queries"][0]["candidates"]
        self.assertEqual([c["tag"] for c in ranked], ["G_lynx_173", "G_lynx_129"])
        self.assertEqual([c["rank"] for c in ranked], [1, 2])
        self.assertEqual(payload["queries"][0]["correct_rank"], 1)
        self.assertTrue(ranked[0]["same_individual"]); self.assertFalse(ranked[1]["same_individual"])
        self.assertEqual(ranked[1]["order_by_confidence"], [1, 0])
        self.assertTrue(payload["synthetic"])
        self.assertIn("CC BY 4.0", payload["attribution"])
        self.assertEqual(len(payload["gallery"]), 2)
        json.dumps(payload)

    def test_individuals_table_is_consistent(self):
        paths = [p for sides in D.INDIVIDUALS.values() for p in sides.values()]
        self.assertEqual(len(paths), len(set(paths)))
        for identity, sides in D.INDIVIDUALS.items():
            self.assertEqual(set(sides), {"query", "gallery"})
            for rel in sides.values():
                self.assertEqual(D.identity_of(rel), identity)

    def test_payload_rejects_private_paths(self):
        with self.assertRaises(ValueError):
            D.build_payload([{"tag": "q", "identity": "lynx_1", "source": "/shared/x.jpg", "image": {}, "candidates": []}], [], {})


class CommittedDemoTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "synthetic" / "synthetic_demo.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("synthetic demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        self.assertTrue(data["synthetic"])
        self.assertIn("Picek", data["attribution"])
        gallery = {g["tag"]: g for g in data["gallery"]}
        self.assertEqual(len(gallery), len(data["queries"]))
        for g in gallery.values():
            self.assertTrue((self.DEMO.parent / g["image"]["file"]).is_file())
        for q in data["queries"]:
            self.assertTrue((self.DEMO.parent / q["image"]["file"]).is_file())
            scores = [c["score"] for c in q["candidates"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertEqual(len(q["candidates"]), len(gallery))
            self.assertEqual(sum(c["same_individual"] for c in q["candidates"]), 1)
            self.assertIsNotNone(q["correct_rank"])
            for c in q["candidates"]:
                self.assertIn(c["tag"], gallery)
                n = c["match_count"]
                self.assertEqual(len(c["points"]["query"]), n)
                self.assertEqual(len(c["points"]["gallery"]), n)
                self.assertEqual(len(c["confidence"]), n)
                self.assertEqual(sorted(c["order_by_confidence"]), list(range(n)))
                for side in ("query", "gallery"):
                    size = q["image"] if side == "query" else c["image"]
                    for x, y in c["points"][side]:
                        self.assertTrue(-1 <= x <= size["width"] and -1 <= y <= size["height"], (q["tag"], c["tag"], side, x, y))
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text); self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
