"""ALIKED-LightGlue and SuperPoint-LightGlue extraction through the Vismatch backend.

LightGlue's own extractors take one image per call (``assert img.shape[0] == 1``) and return
batch-shaped outputs, including ``keypoint_scores`` of shape (1, N). Before 2026-10-08 the backend
sent them batches of same-shape images and kept the scores' batch dimension, so neither matcher could
run. RDD-LightGlue and LoMa keep their batched extraction.
"""

import unittest

import numpy as np
import torch

try:
    from wildmatch.matchers.vismatch import BATCHED_EXTRACTION_MATCHERS, VismatchMatcherBackend

    HAS_VISMATCH_BACKEND = True
except ModuleNotFoundError:
    HAS_VISMATCH_BACKEND = False


class FakeLightGlueExtractor:
    """Mimics lightglue.utils.Extractor.extract: one image per call, batch-shaped outputs."""

    def __init__(self):
        self.calls = 0

    def extract(self, img):
        if img.dim() == 3:
            img = img[None]
        assert img.dim() == 4 and img.shape[0] == 1
        self.calls += 1
        value = float(img.mean())
        return {
            "keypoints": torch.full((1, 3, 2), value),
            "descriptors": torch.full((1, 3, 8), value),
            "keypoint_scores": torch.full((1, 3), 0.5),
            "image_size": torch.tensor([[img.shape[-1], img.shape[-2]]], dtype=torch.float32),
        }


def make_backend(matcher, extractor):
    backend = object.__new__(VismatchMatcherBackend)
    backend.matcher_name = matcher
    backend.resize_max = 512
    backend.preprocessing_divisor = 32
    backend.device = torch.device("cpu")
    backend.extractor = extractor
    return backend


@unittest.skipUnless(HAS_VISMATCH_BACKEND, "Vismatch backend dependencies are not available")
class LightGlueExtractorTest(unittest.TestCase):
    def test_lightglue_extractors_go_image_by_image(self):
        for matcher in ("aliked-lightglue", "superpoint-lightglue"):
            extractor = FakeLightGlueExtractor()
            backend = make_backend(matcher, extractor)
            images = [torch.full((3, 40, 20), i / 4) for i in range(3)]  # same processed shape, values in [0, 1]
            features = backend.extract_frames_batch(images)
            self.assertEqual(extractor.calls, 3, matcher)
            for i, feature in enumerate(features):
                self.assertEqual(feature.keypoints.shape, (3, 2))
                self.assertEqual(feature.scores.shape, (3,))
                np.testing.assert_allclose(feature.keypoints, i / 4, rtol=1e-6)  # no item mix-up

    def test_single_image_scores_lose_their_batch_dimension(self):
        backend = make_backend("aliked-lightglue", FakeLightGlueExtractor())
        feature = backend.extract_frame(torch.zeros((3, 40, 20)))
        self.assertEqual(feature.scores.shape, (3,))
        np.testing.assert_allclose(feature.scores, 0.5)

    def test_rdd_and_loma_keep_batched_extraction(self):
        self.assertEqual(BATCHED_EXTRACTION_MATCHERS, frozenset({"rdd-lightglue", "loma"}))

        class FakeRDDExtractor:
            batch_sizes = []

            def extract(self, images):
                self.batch_sizes.append(int(images.shape[0]))
                return [
                    {
                        "keypoints": torch.zeros((1, 2)),
                        "descriptors": torch.ones((1, 4)),
                        "scores": torch.ones(1),
                    }
                    for _ in range(int(images.shape[0]))
                ]

        extractor = FakeRDDExtractor()
        backend = make_backend("rdd-lightglue", extractor)
        backend.extract_frames_batch([torch.zeros((3, 40, 20)) for _ in range(3)])
        self.assertEqual(extractor.batch_sizes, [3])


if __name__ == "__main__":
    unittest.main()
