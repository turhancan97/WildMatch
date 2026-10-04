"""Tests for the CPU-only paper/page/export_masking_demo.py."""

import pytest
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "paper/page" / "export_masking_demo.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_masking_demo", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return json.dumps(rle)


class MaskingDemoHelpersTests(unittest.TestCase):
    def test_iou(self):
        a = np.zeros((4, 4), bool)
        a[:2, :] = True
        b = np.zeros((4, 4), bool)
        b[1:3, :] = True
        self.assertAlmostEqual(M.iou(a, b), 4 / 12)
        self.assertEqual(M.iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)

    def test_default_renders_follow_the_match_demo(self):
        renders = M.default_renders()
        self.assertEqual(len(renders), 2 * len(M.INDIVIDUALS))
        self.assertEqual(len({r["tag"] for r in renders}), len(renders))
        self.assertTrue(all(r["role"] in ("query", "gallery") for r in renders))

    def test_export_end_to_end_on_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "imgs").mkdir(parents=True)
            sam3 = Path(tmp) / "sam3"
            sam3.mkdir()
            out = Path(tmp) / "out"
            renders = [
                {"tag": "Q_lynx_1", "path": "imgs/synthetic_lynx_1/a.jpg", "identity": "lynx_1", "role": "query"},
                {"tag": "G_lynx_1", "path": "imgs/synthetic_lynx_1/b.jpg", "identity": "lynx_1", "role": "gallery"},
            ]
            h, w = 40, 60
            dataset_mask = np.zeros((h, w), np.uint8)
            dataset_mask[10:30, 10:50] = 1
            sam3_mask = np.zeros((h, w), np.uint8)
            sam3_mask[12:30, 10:50] = 1
            with (root / "meta.csv").open("w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=["path", "mask"])
                wr.writeheader()
                for r in renders:
                    (root / r["path"]).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(root / r["path"])
                    wr.writerow({"path": r["path"], "mask": _rle(dataset_mask)})
            with (sam3 / "masks.csv").open("w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "path",
                        "masked_path",
                        "n_detections",
                        "best_score",
                        "threshold_used",
                        "prompt_used",
                        "merge",
                        "fg_fraction",
                        "bbox",
                        "mask",
                    ],
                )
                wr.writeheader()
                for r in renders:
                    wr.writerow(
                        {
                            "path": r["path"],
                            "masked_path": "x",
                            "n_detections": "2",
                            "best_score": "0.91",
                            "threshold_used": "0.5",
                            "prompt_used": "Lynx",
                            "merge": "union",
                            "fg_fraction": str(sam3_mask.mean()),
                            "bbox": "",
                            "mask": _rle(sam3_mask),
                        }
                    )
            # The exporter prints a per-render report, and pycocotools 2.x triggers a NumPy 2
            # DeprecationWarning inside its RLE decoder; neither is our code under test.
            with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pycocotools\.")
                payload = M.export(root, "meta.csv", sam3, out, renders)
            self.assertEqual([g["key"] for g in payload["groups"]], ["synthetic"])
            items = payload["groups"][0]["items"]
            self.assertEqual(len(items), 2)
            item = items[0]
            self.assertEqual(item["sam3"]["prompt"], "Lynx")
            self.assertEqual(item["sam3"]["instances"], 2)
            self.assertAlmostEqual(item["iou_with_dataset_mask"], (18 * 40) / (20 * 40), places=3)
            self.assertTrue((out / "Q_lynx_1.jpg").is_file())
            self.assertTrue((out / "Q_lynx_1_sam3.png").is_file())
            self.assertTrue((out / "Q_lynx_1_dataset.png").is_file())
            with Image.open(out / "Q_lynx_1_sam3.png") as exported_mask:
                self.assertEqual(exported_mask.size, (w, h))
            self.assertIn("Lynx", payload["run"]["prompts"])
            text = (out / "masking_demo.json").read_text(encoding="utf-8")
            self.assertNotIn("/home/", text)

    @pytest.mark.data
    def test_real_items_resolve_raw_paths_and_references(self):
        if not M.MATCH_EXAMPLES_JSON.is_file() or not all(Path(p.metadata).is_file() for p in M.PAPER_PROFILES):
            self.skipTest("match examples or dataset metadata not available")
        items = M.real_items()
        self.assertEqual(len(items), 16)
        kinds = {i["dataset"]: (i["reference"] or {}).get("type") for i in items}
        self.assertEqual(kinds["lynx_closed"], "rle")
        self.assertIsNone(kinds["salamander"])
        self.assertIsNone(kinds["hyena"])
        self.assertTrue(all(not i["path"].startswith("masked_images/") for i in items))
        self.assertTrue(all((Path(i["root"]) / i["path"]).is_file() for i in items))

    def test_missing_sam3_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                M.load_sam3_rows(Path(tmp), ["a.jpg"])


class CommittedMaskingDemoTests(unittest.TestCase):
    DEMO = REPO_ROOT / "docs" / "assets" / "demo" / "masking" / "masking_demo.json"

    def test_committed_demo_is_consistent(self):
        if not self.DEMO.is_file():
            self.skipTest("masking demo not exported")
        data = json.loads(self.DEMO.read_text(encoding="utf-8"))
        self.assertEqual(data["groups"][0]["key"], "synthetic")
        self.assertTrue(data["groups"][0]["synthetic"])
        for group in data["groups"]:
            self.assertIn("Picek", group["attribution"])
            self.assertTrue(group["items"])
            for item in group["items"]:
                for key in ("image", "sam3_mask", "dataset_mask"):
                    file = item[key]["file"] if key == "image" else item[key]
                    if file is None:
                        continue
                    self.assertTrue((self.DEMO.parent / file).is_file(), file)
                with Image.open(self.DEMO.parent / item["sam3_mask"]) as sam3_mask:
                    self.assertEqual(sam3_mask.size, (item["image"]["width"], item["image"]["height"]))
                if item["dataset_mask"] is None:
                    self.assertIsNone(item["iou_with_dataset_mask"])
                    self.assertIsNone(item["reference_mask_source"])
                else:
                    self.assertTrue(0.0 <= item["iou_with_dataset_mask"] <= 1.0)
                    self.assertTrue(item["reference_mask_source"])
                if group["key"] == "real":
                    self.assertEqual(item["dataset_mask"] is not None, item["dataset"] == "lynx_closed")
        text = self.DEMO.read_text(encoding="utf-8")
        self.assertNotIn("/shared/", text)
        self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
