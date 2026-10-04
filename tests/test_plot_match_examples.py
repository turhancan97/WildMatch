from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from wildmatch.reporting.paper_datasets import PAPER_PROFILES
from paper.figures.plot_match_examples import (
    EXAMPLES,
    PAPER_ORDER,
    RUNS,
    Example,
    correct_top1_pairs,
    crop_box_around,
    dim_background,
    padding_box,
    points_outside_boxes,
    spread_selection,
    dhash,
    hamming,
    normalized_to_raw_pixels,
    pair_rejection,
    raw_image_path,
    resolve_recorded_checkpoint,
    validate_examples,
)

PROFILES = {profile.key: profile for profile in PAPER_PROFILES}


class CoordinateMappingTests(unittest.TestCase):
    def test_normalized_corners_and_centre_map_to_raw_pixels(self):
        points = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
        out = normalized_to_raw_pixels(points, (200, 400))  # H=200, W=400
        # Half-pixel convention: -1 is the left edge of pixel 0, i.e. x = -0.5.
        np.testing.assert_allclose(out[0], [-0.5, -0.5])
        np.testing.assert_allclose(out[1], [399.5, 199.5])
        np.testing.assert_allclose(out[2], [199.5, 99.5])

    def test_mapping_is_per_axis(self):
        out = normalized_to_raw_pixels(np.array([[0.5, -0.5]]), (100, 300))
        np.testing.assert_allclose(out[0], [300 * 0.75 - 0.5, 100 * 0.25 - 0.5])


class RawPathTests(unittest.TestCase):
    def test_pre_masked_wildlife_path_maps_to_images_tree(self):
        row = {"path": "masked_images/NyalaData/a/b.jpg"}
        self.assertEqual(raw_image_path(PROFILES["nyala"], row), "images/NyalaData/a/b.jpg")

    def test_salamander_uses_original_path(self):
        row = {"path": "masked_images/query/images/x.jpg", "original_path": "query/images/x.jpg"}
        self.assertEqual(raw_image_path(PROFILES["salamander"], row), "query/images/x.jpg")

    def test_czechlynx_path_is_already_raw(self):
        row = {"path": "CzechLynx/foe/lynx_1/0001.jpg"}
        self.assertEqual(raw_image_path(PROFILES["lynx_closed"], row), "CzechLynx/foe/lynx_1/0001.jpg")


class EligibilityTests(unittest.TestCase):
    def test_same_encounter_or_day_is_rejected_only_when_both_sides_record_it(self):
        self.assertEqual(pair_rejection({"encounter": "7"}, {"encounter": "7"}, None, 0), "same_encounter")
        self.assertEqual(pair_rejection({"date": "2020-01-01"}, {"date": "2020-01-01"}, None, 0), "same_date")
        self.assertIsNone(pair_rejection({"date": "2020-01-01"}, {"date": float("nan")}, None, 0))
        self.assertIsNone(pair_rejection({"encounter": "7"}, {"encounter": "8"}, 20, 6))

    def test_near_duplicates_are_rejected_by_dhash(self):
        gradient = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1))
        a = Image.fromarray(np.stack([gradient] * 3, axis=-1))
        b = Image.fromarray(np.stack([gradient[:, ::-1]] * 3, axis=-1))
        self.assertEqual(hamming(dhash(a), dhash(a)), 0)
        self.assertGreater(hamming(dhash(a), dhash(b)), 30)
        self.assertEqual(pair_rejection({}, {}, 2, 6), "near_duplicate")


class TopOneTests(unittest.TestCase):
    def test_correct_top1_uses_descending_score_and_lowest_index_ties(self):
        # query 0: tie at 0.9 between gallery 2 (wrong) and 1 (right) -> index 1 wins
        # query 1: best is gallery 0 (wrong); query 2 has only unscored/-inf entries
        rows = np.array([0, 0, 0, 1, 1, 2])
        cols = np.array([2, 1, 0, 0, 2, 1])
        values = np.array([0.9, 0.9, 0.1, 0.8, 0.3, -np.inf])
        pairs = correct_top1_pairs(rows, cols, values, ["a", "b", "b"], ["x", "a", "a"])
        self.assertEqual(pairs, [(0, 1, 0.9)])


class CropTests(unittest.TestCase):
    def test_crop_trims_only_the_long_side_and_follows_the_keypoints(self):
        # 800x300 landscape, 4:3 crop -> 400x300, centred on points near the right edge
        box = crop_box_around((800, 300), np.array([[700.0, 100.0], [760.0, 200.0]]), 4 / 3)
        self.assertEqual(box, (400, 0, 800, 300))
        # portrait 300x800 -> 300x225, centred vertically on the points
        box = crop_box_around((300, 800), np.array([[10.0, 390.0], [290.0, 410.0]]), 4 / 3)
        self.assertEqual(box, (0, 288, 300, 513))
        # no points -> centred
        self.assertEqual(crop_box_around((400, 300), np.empty((0, 2)), 4 / 3), (0, 0, 400, 300))


class PaperStyleTests(unittest.TestCase):
    def test_padding_box_trims_only_dark_edge_bands(self):
        image = np.full((50, 80, 3), 120, dtype=np.uint8)
        image[:, :10] = 0  # left padding
        image[40:] = 3  # bottom padding
        image[20:25, 30:35] = 0  # dark content inside stays
        self.assertEqual(padding_box(image), (10, 0, 80, 40))
        self.assertEqual(padding_box(np.zeros((5, 5, 3), dtype=np.uint8)), (0, 0, 5, 5))

    def test_spread_selection_skips_crowded_strong_matches(self):
        p0 = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        p1 = p0 + 500
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        np.testing.assert_array_equal(spread_selection(p0, p1, conf, 3, 10.0), [0, 2, 3])
        # spacing relaxes until enough matches exist
        np.testing.assert_array_equal(spread_selection(p0, p1, conf, 4, 10.0), [0, 1, 2, 3])

    def test_points_under_tags_are_rejected_with_margin(self):
        points = np.array([[5.0, 5.0], [15.0, 5.0], [30.0, 30.0]])
        np.testing.assert_array_equal(points_outside_boxes(points, [(0, 0, 10, 10)]), [False, True, True])
        np.testing.assert_array_equal(points_outside_boxes(points, [(10, 10, 0, 0)], margin=6), [False, False, True])

    def test_dim_background_keeps_the_animal_and_darkens_the_rest(self):
        rgb = np.zeros((60, 60, 3), dtype=np.uint8)
        rgb[..., 0] = 200
        mask = np.zeros((60, 60), dtype=bool)
        mask[20:40, 20:40] = True
        out = np.asarray(dim_background(Image.fromarray(rgb), mask))
        np.testing.assert_array_equal(out[30, 30], rgb[30, 30])
        self.assertLess(int(out[2, 2].max()), 200)


class CheckpointResolutionTests(unittest.TestCase):
    def test_hash_finds_renamed_copy_and_prefers_epoch_over_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorded = root / "legacy" / "epoch_299" / "model.safetensors"
            for path, content in ((recorded, b"overwritten"),
                                  (root / "legacy-mined" / "latest" / "model.safetensors", b"used"),
                                  (root / "legacy-mined" / "epoch_299" / "model.safetensors", b"used")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            hasher = lambda path: path.read_bytes().decode()
            self.assertEqual(resolve_recorded_checkpoint(recorded, "overwritten", root, hasher), recorded)
            self.assertEqual(resolve_recorded_checkpoint(recorded, "used", root, hasher),
                             root / "legacy-mined" / "epoch_299" / "model.safetensors")
            with self.assertRaises(FileNotFoundError):
                resolve_recorded_checkpoint(recorded, "absent", root, hasher)


class ExampleListTests(unittest.TestCase):
    def test_runs_cover_every_paper_dataset(self):
        self.assertEqual(set(RUNS), {key for key, _ in PAPER_ORDER})
        self.assertTrue({key for key, _ in PAPER_ORDER} <= set(PROFILES))

    def test_validate_examples_requires_one_pair_per_dataset_in_order(self):
        good = [Example(key, f"{key}/q.jpg", f"{key}/g.jpg", "") for key, _ in PAPER_ORDER]
        validate_examples(good)
        with self.assertRaises(ValueError):
            validate_examples(good[::-1])
        with self.assertRaises(ValueError):
            validate_examples(good[:-1])

    def test_pinned_examples_are_valid_once_filled(self):
        if not EXAMPLES:
            self.skipTest("EXAMPLES is filled after the candidate sheets are reviewed")
        validate_examples(EXAMPLES)


if __name__ == "__main__":
    unittest.main()
