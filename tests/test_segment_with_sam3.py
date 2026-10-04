from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from wildmatch.data.prepare.sam3_masks import (
    MetadataError,
    build_masked_metadata,
    decode_mask,
    encode_mask,
    ensure_directory,
    merge_instances,
    parse_args,
    parse_mapping,
    parse_threshold_overrides,
)


class MergeInstancesTests(unittest.TestCase):
    def setUp(self):
        # Two pieces of one animal split by an occluding finger (columns 0-1 and 3-4).
        self.masks = np.zeros((2, 1, 3, 5), dtype=bool)
        self.masks[0, 0, :, 0:2] = True
        self.masks[1, 0, :, 3:5] = True
        self.scores = np.array([0.4, 0.9])

    def test_union_keeps_every_piece(self):
        mask, count, best = merge_instances(self.masks, self.scores, "union")
        self.assertEqual(count, 2)
        self.assertAlmostEqual(best, 0.9)
        self.assertEqual(int(mask.sum()), 12)

    def test_best_keeps_only_the_top_piece(self):
        mask, _, _ = merge_instances(self.masks, self.scores, "best")
        self.assertTrue(np.array_equal(mask, self.masks[1, 0]))

    def test_largest_keeps_the_piece_with_most_pixels(self):
        sizes = self.masks.reshape(self.masks.shape[0], -1).sum(axis=1)
        mask, _, best = merge_instances(self.masks, self.scores, "largest")
        self.assertTrue(np.array_equal(mask, self.masks[int(np.argmax(sizes)), 0]))
        self.assertAlmostEqual(best, 0.9)  # the reported score stays the best detection's

    def test_no_detection_is_reported_not_dropped(self):
        mask, count, best = merge_instances(np.zeros((0, 3, 5), dtype=bool), np.zeros(0), "union")
        self.assertIsNone(mask)
        self.assertEqual((count, best), (0, 0.0))

    def test_rle_roundtrip(self):
        # pycocotools 2.x calls np.array(copy=False) internally; NumPy 2 warns. Not our code.
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pycocotools\.")
        mask = np.zeros((7, 9), dtype=bool)
        mask[2:5, 1:8] = True
        self.assertTrue(np.array_equal(decode_mask(encode_mask(mask)), mask))


class ParsingTests(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(parse_mapping("database=train, query=test"), {"database": "train", "query": "test"})
        for bad in ("database", "a=b,a=c", "=x"):
            with self.assertRaises(ValueError):
                parse_mapping(bad)

    def test_threshold_overrides(self):
        self.assertEqual(parse_threshold_overrides(["query/images/a=b.jpg=0.1"]), {"query/images/a=b.jpg": 0.1})
        with self.assertRaises(ValueError):
            parse_threshold_overrides(["0.1"])

    def test_a_step_is_required(self):
        # argparse prints its usage message to stderr before exiting; keep test output clean.
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse_args(["--root", "r", "--csv", "c.csv"])


class MaskedMetadataTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = pd.DataFrame(
            {
                "image_id": [1, 2, 3],
                "identity": ["a", "a", "b"],
                "path": ["database/images/1.jpg", "query/images/2.jpg", "database/images/3.jpg"],
                "split": ["database", "query", "database"],
            }
        )
        masks = pd.DataFrame(
            {
                "path": source["path"],
                "masked_path": "masked_images/" + source["path"],
                "mask": [encode_mask(np.ones((2, 2), dtype=bool))] * 3,
                "best_score": [0.9, 0.8, 0.1],
                "fg_fraction": [0.3, 0.4, 0.2],
                "n_detections": [1, 2, 3],
                "threshold_used": [0.5, 0.5, 0.1],
                "prompt_used": ["Salamander"] * 3,
            }
        )
        for rel in masks["masked_path"]:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x")
        return source, masks

    def test_builds_pre_masked_metadata_with_mapped_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, masks = self._fixture(root)
            out = build_masked_metadata(
                source,
                masks,
                root,
                split_col="split",
                split_map={"database": "train", "query": "test"},
                split_map_column="split_train_test",
            )
        self.assertEqual(
            list(out.columns),
            [
                "image_id",
                "identity",
                "path",
                "split",
                "original_path",
                "mask",
                "sam3_score",
                "sam3_fg_fraction",
                "sam3_n_instances",
                "sam3_threshold",
                "sam3_prompt",
                "split_train_test",
            ],
        )
        self.assertTrue(out["path"].str.startswith("masked_images/").all())
        self.assertEqual(list(out["original_path"]), list(source["path"]))
        self.assertEqual(list(out["split"]), ["database", "query", "database"])  # untouched
        self.assertEqual(list(out["split_train_test"]), ["train", "test", "train"])
        self.assertEqual(list(out["sam3_threshold"]), [0.5, 0.5, 0.1])

    def test_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, masks = self._fixture(root)
            with self.assertRaisesRegex(MetadataError, "no mask"):
                build_masked_metadata(source, masks.iloc[:2], root)
            (root / masks["masked_path"].iloc[0]).unlink()
            with self.assertRaisesRegex(MetadataError, "do not exist"):
                build_masked_metadata(source, masks, root)
            self._fixture(root)
            with self.assertRaisesRegex(MetadataError, "without a mapping"):
                build_masked_metadata(
                    source,
                    masks,
                    root,
                    split_col="split",
                    split_map={"database": "train"},
                    split_map_column="split_train_test",
                )
            with self.assertRaisesRegex(MetadataError, "duplicate"):
                build_masked_metadata(pd.concat([source, source.iloc[:1]]), masks, root)


class EnsureDirectoryTests(unittest.TestCase):
    def test_retries_a_concurrent_creation(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "a", "b")
            real_mkdir = Path.mkdir
            calls = []

            def flaky(self, *args, **kwargs):
                calls.append(self)
                if len(calls) == 1:
                    raise FileExistsError(str(self))
                return real_mkdir(self, *args, **kwargs)

            with mock.patch.object(Path, "mkdir", flaky), mock.patch("time.sleep"):
                ensure_directory(target)
            self.assertTrue(target.is_dir())
            self.assertGreaterEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
