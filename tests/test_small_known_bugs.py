"""Regression tests for three known issues fixed on 2026-10-04 (none affected reported results)."""

import inspect
import unittest

import numpy as np

from wildmatch.data.dataset_view import BenchmarkDatasetView
from wildmatch.evaluate import probe_runner
from wildmatch.matchers.vismatch import scored_rank1


class KnownIssueTests(unittest.TestCase):
    def test_vismatch_drawings_skip_queries_without_a_scored_candidate(self):
        self.assertIsNone(scored_rank1(np.full(5, -np.inf)))
        row = np.array([-np.inf, 0.2, 0.9, -np.inf])
        self.assertEqual(scored_rank1(row), 2)
        # ties at the top keep the lowest index, as everywhere else
        self.assertEqual(scored_rank1(np.array([-np.inf, 0.5, 0.5])), 1)

    def test_probe_training_probabilities_build_no_graph(self):
        source = inspect.getsource(probe_runner)
        calls = [line for line in source.splitlines() if "train_probs = _predict_class_probabilities" in line]
        self.assertEqual(len(calls), 2)  # linear and efficient probe
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if "train_probs = _predict_class_probabilities" in line:
                self.assertIn("torch.no_grad()", lines[index - 1])

    def test_float_images_in_unit_range_are_scaled_not_binarised(self):
        to_uint8 = BenchmarkDatasetView._to_hwc_uint8
        unit = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        self.assertEqual(to_uint8(None, unit, 0).tolist(), [[[0, 127, 255]]])
        full_range = np.array([[[0.0, 127.0, 300.0]]], dtype=np.float32)
        self.assertEqual(to_uint8(None, full_range, 0).tolist(), [[[0, 127, 255]]])
        already = np.array([[[1, 2, 3]]], dtype=np.uint8)
        self.assertIs(to_uint8(None, already, 0).dtype, np.dtype(np.uint8))


if __name__ == "__main__":
    unittest.main()
