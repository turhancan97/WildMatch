"""Feature caches hash the masks applied at load time (dataset.no_background)."""

import hashlib
import unittest
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

from wildmatch.evaluate.probe_runner import cache_mask_col, dataset_digest
from wildmatch.matchers.vismatch import _cache_key
from wildmatch.utils.fingerprints import mask_digest

RLE_A = '{"size": [4, 4], "counts": "04"}'
RLE_B = '{"size": [4, 4], "counts": "13"}'


def _dataset(masks):
    return SimpleNamespace(df=pd.DataFrame({"path": ["a.jpg", "b.jpg"], "identity": [1, 2], "mask": masks}))


class MaskCacheIdentityTests(unittest.TestCase):
    def test_dataset_digest_without_mask_col_is_unchanged(self):
        expected = hashlib.sha256("a.jpg|1\nb.jpg|2".encode("utf-8")).hexdigest()
        self.assertEqual(dataset_digest(_dataset([RLE_A, RLE_A]), "identity"), expected)
        self.assertEqual(dataset_digest(_dataset([RLE_A, RLE_B]), "identity"), expected)

    def test_dataset_digest_with_mask_col_follows_the_masks(self):
        same = dataset_digest(_dataset([RLE_A, RLE_A]), "identity", mask_col="mask")
        self.assertEqual(same, dataset_digest(_dataset([RLE_A, RLE_A]), "identity", mask_col="mask"))
        self.assertNotEqual(same, dataset_digest(_dataset([RLE_A, RLE_B]), "identity", mask_col="mask"))
        self.assertNotEqual(same, dataset_digest(_dataset([RLE_A, RLE_A]), "identity"))

    def test_only_load_time_masking_hashes_masks(self):
        def cfg(no_background):
            return OmegaConf.create({"dataset": {"no_background": no_background, "mask_col": "mask"}})

        self.assertEqual(cache_mask_col(cfg(True)), "mask")
        self.assertIsNone(cache_mask_col(cfg(False)))

    def test_vismatch_key_adds_the_mask_only_when_given(self):
        args = dict(
            image_path="a.jpg", image_content_hash="c", split_name="db", resize_max=512, top_k=2048, cfg_tag="t"
        )
        old = hashlib.sha256("db|a.jpg|content=c|resize_max=512|top_k=2048|t".encode("utf-8")).hexdigest()
        self.assertEqual(_cache_key(**args), old)
        self.assertEqual(_cache_key(**args, mask_hash=None), old)
        with_a = _cache_key(**args, mask_hash=mask_digest(RLE_A))
        self.assertNotEqual(with_a, old)
        self.assertNotEqual(with_a, _cache_key(**args, mask_hash=mask_digest(RLE_B)))

    def test_mask_digest_handles_dicts_and_missing_cells(self):
        self.assertEqual(mask_digest(None), mask_digest(float("nan")))
        self.assertEqual(mask_digest({"b": 1, "a": 2}), mask_digest({"a": 2, "b": 1}))
        self.assertNotEqual(mask_digest(RLE_A), mask_digest(RLE_B))


if __name__ == "__main__":
    unittest.main()
