"""Tests for wildmatch.data.prepare.jaguar on synthetic RGBA images (CPU only)."""

import pytest
import csv
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
from wildmatch.data.prepare import jaguar as P  # noqa: E402


def _rgba(seed: int, size=(48, 64), background=90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((size[0], size[1], 4), background, np.uint8)
    image[..., 3] = 0
    image[8:40, 10:54, :3] = rng.integers(0, 256, (32, 44, 3), dtype=np.uint8)
    image[8:40, 10:54, 3] = 255
    return image


def _write_dataset(root: Path, specs) -> None:
    """specs: list of (filename, identity, rgba array)."""
    (root / P.IMAGE_DIR).mkdir(parents=True)
    with open(root / P.TRAIN_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "ground_truth"])
        for filename, identity, rgba in specs:
            Image.fromarray(rgba, "RGBA").save(root / P.IMAGE_DIR / filename)
            writer.writerow([filename, identity])
    with open(root / "test.csv", "w", newline="", encoding="utf-8") as handle:
        handle.write("row_id,query_image,gallery_image\n0,test_0001.png,test_0002.png\n")


class ImageHelperTests(unittest.TestCase):
    def test_masked_rgb_blackens_transparent_background(self):
        rgba = _rgba(0)
        masked = P.masked_rgb(rgba)
        self.assertTrue((masked[rgba[..., 3] == 0] == 0).all())
        np.testing.assert_array_equal(masked[rgba[..., 3] == 255], rgba[..., :3][rgba[..., 3] == 255])

    def test_hash_ignores_the_hidden_background(self):
        a, b = _rgba(1, background=10), _rgba(1, background=240)
        np.testing.assert_array_equal(P.dhash(a), P.dhash(b))

    def test_hash_hex_roundtrip_and_distance(self):
        bits = [P.dhash(_rgba(seed)) for seed in (2, 2, 3)]
        hexes = [P.bits_to_hex(b) for b in bits]
        np.testing.assert_array_equal(P.hex_to_bits(hexes[0]), bits[0])
        distances = P.distance_matrix(hexes)
        self.assertEqual(distances[0, 1], 0)
        self.assertEqual(distances[0, 2], int((bits[0] != bits[2]).sum()))

    def test_mask_rle_roundtrip(self):
        from pycocotools import mask as mask_utils

        rgba = _rgba(4)
        payload = json.loads(P.encode_mask(P.alpha_mask(rgba)))
        payload["counts"] = payload["counts"].encode("ascii")
        with warnings.catch_warnings():  # pycocotools 2.x vs NumPy 2; not our code
            warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pycocotools\.")
            decoded = mask_utils.decode(payload).astype(bool)
        np.testing.assert_array_equal(decoded, rgba[..., 3] > 0)

    def test_rgb_image_without_alpha_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.png"
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
            with self.assertRaises(P.PreparationError):
                P.load_rgba(path)


def _unit(vectors):
    v = np.asarray(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class BurstGroupTests(unittest.TestCase):
    def _groups(self, distances, vectors, numbers, identities, **kwargs):
        options = dict(threshold=32, adjacent_gap=1, adjacent_cos=0.85, any_cos=0.95)
        options.update(kwargs)
        e = _unit(vectors)
        return P.burst_groups(np.asarray(distances), e @ e.T, numbers, identities, **options)

    def test_adjacent_similar_frames_join_and_far_dissimilar_ones_do_not(self):
        distances = np.full((4, 4), 200); np.fill_diagonal(distances, 0)
        groups = self._groups(distances, [[1, 0, 0], [0.9, 0.1, 0], [1, 0, 0.05], [0, 1, 0]],
                              [1, 2, 10, 11], ["a"] * 4, any_cos=0.999)
        self.assertEqual(groups[0], groups[1])        # adjacent and similar
        self.assertNotEqual(groups[0], groups[2])     # similar but far apart, below any_cos
        self.assertNotEqual(groups[2], groups[3])     # adjacent but dissimilar

    def test_hash_twins_join_wherever_they_sit(self):
        groups = self._groups([[0, 20], [20, 0]], [[1, 0], [0, 1]], [1, 50], ["a", "a"])
        self.assertEqual(groups[0], groups[1])

    def test_any_cos_joins_far_apart_near_identical_frames(self):
        groups = self._groups([[0, 200], [200, 0]], [[1, 0], [1, 0.001]], [1, 50], ["a", "a"])
        self.assertEqual(groups[0], groups[1])

    def test_groups_join_transitively(self):
        distances = np.full((3, 3), 200); np.fill_diagonal(distances, 0)
        groups = self._groups(distances, [[1, 0], [0.95, 0.31], [0.81, 0.59]], [1, 2, 3], ["a"] * 3,
                              adjacent_cos=0.94, any_cos=0.999)
        self.assertEqual(len(set(groups)), 1)

    def test_different_jaguars_are_never_joined(self):
        groups = self._groups([[0, 100], [100, 0]], [[1, 0], [1, 0]], [1, 2], ["a", "b"])
        self.assertNotEqual(groups[0], groups[1])

    def test_near_identical_photos_under_two_labels_fail_closed(self):
        with self.assertRaises(P.PreparationError):
            self._groups([[0, 3], [3, 0]], [[1, 0], [0, 1]], [1, 2], ["a", "b"])

    def test_file_number_parsing_fails_closed(self):
        self.assertEqual(P.file_number("train_0688"), 688)
        with self.assertRaises(P.PreparationError):
            P.file_number("train")


class AssignSplitTests(unittest.TestCase):
    def _split(self, identities, groups, **kwargs):
        options = dict(query_ratio=0.25, min_query=2, seed=0)
        options.update(kwargs)
        keys = [f"f{i:04d}" for i in range(len(identities))]
        return P.assign_split(identities, groups, keys, **options)

    def test_split_keeps_groups_whole_and_every_identity_on_both_sides(self):
        identities = ["a"] * 20 + ["b"] * 12
        groups = [i // 2 for i in range(32)]  # bursts of two
        split = self._split(identities, groups)
        for group in set(groups):
            self.assertEqual(len({split[i] for i in range(32) if groups[i] == group}), 1)
        for identity in ("a", "b"):
            sides = {split[i] for i in range(32) if identities[i] == identity}
            self.assertEqual(sides, {"database", "query"})
        self.assertEqual(sum(1 for i in range(20) if split[i] == "query"), 6)  # round(5) -> 6 by pairs

    def test_split_is_deterministic_and_seeded(self):
        identities = ["a"] * 40
        groups = list(range(40))
        self.assertEqual(self._split(identities, groups), self._split(identities, groups))
        self.assertNotEqual(self._split(identities, groups), self._split(identities, groups, seed=1))

    def test_large_group_is_skipped_when_smaller_ones_fit(self):
        identities = ["a"] * 36
        groups = [0] * 17 + [1] * 5 + [2] * 4 + list(range(3, 13))
        for seed in range(10):
            split = self._split(identities, groups, seed=seed)
            self.assertTrue(all(split[i] == "database" for i in range(17)), seed)
            self.assertLessEqual(split.count("query"), 11)

    def test_identity_with_a_single_group_fails_closed(self):
        with self.assertRaises(P.PreparationError):
            self._split(["a"] * 4, [0, 0, 0, 0])

    def test_leakage_counts_cross_side_same_identity_pairs(self):
        distances = np.array([[0, 5, 5], [5, 0, 30], [5, 30, 0]])
        leak = P.split_leakage(distances, ["a", "a", "b"], ["database", "query", "query"], bands=(8, 32))
        self.assertEqual(leak, {"cross_side_same_identity_pairs_le_8": 1, "cross_side_same_identity_pairs_le_32": 1})


class PrepareTests(unittest.TestCase):
    def test_prepare_writes_masked_images_and_base_table(self):
        specs = [(f"train_{3 * i + k + 1:04d}.png", identity, _rgba(10 * i + k))
                 for i, identity in enumerate(("Abril", "Bento")) for k in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, specs)
            manifest = P.prepare(root, workers=1)
            rows = P.read_base(root)
            self.assertEqual(len(rows), len(specs))
            with open(root / P.BASE_NAME, newline="", encoding="utf-8") as handle:
                self.assertEqual(next(csv.reader(handle)), P.BASE_COLUMNS)
            for row in rows:
                masked = root / row["path"]
                self.assertEqual(P.sha256_file(masked), row["masked_sha256"])
                with Image.open(masked) as image:
                    self.assertEqual(image.mode, "RGB")
                    pixels = np.asarray(image)
                self.assertTrue((pixels[:8] == 0).all())  # transparent rows are black
            self.assertEqual(manifest["counts"], {"images": 6, "identities": 2})
            self.assertEqual(manifest["excluded"]["kaggle_test_rows"], 1)
            self.assertEqual(manifest["outputs"]["base_sha256"], P.sha256_file(root / P.BASE_NAME))

    def test_missing_image_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_dataset(root, [("train_0001.png", "a", _rgba(0))])
            with open(root / P.TRAIN_CSV, "a", encoding="utf-8") as handle:
                handle.write("train_0002.png,a\n")
            with self.assertRaises(P.PreparationError):
                P.prepare(root, workers=1)
            self.assertFalse((root / P.BASE_NAME).exists())


class SplitDatasetTests(unittest.TestCase):
    def _base(self, root: Path, n_per_identity=12):
        rows, vectors = [], []
        rng = np.random.default_rng(0)
        for k, identity in enumerate(("a", "b")):
            for i in range(n_per_identity):
                image_id = f"train_{k * 100 + i:04d}"
                rows.append({c: "" for c in P.BASE_COLUMNS} | {
                    "image_id": image_id, "identity": identity, "path": f"masked_images/{image_id}.png",
                    "dhash": P.bits_to_hex(rng.integers(0, 2, 256).astype(bool))})
                # pairs of consecutive files form one burst (nearly identical vectors)
                vectors.append(rng.normal(size=8) if i % 2 == 0 else vectors[-1] + rng.normal(scale=0.01, size=8))
        with open(root / P.BASE_NAME, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=P.BASE_COLUMNS)
            writer.writeheader(); writer.writerows(rows)
        np.savez(root / P.EMBEDDINGS_NAME, image_id=np.array([r["image_id"] for r in rows]), emb=_unit(vectors),
                 spec=np.array(P.EMBEDDING_SPEC))
        return rows

    def test_split_keeps_bursts_together_and_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._base(root)
            manifest = P.split_dataset(root, threshold=0, adjacent_gap=1, adjacent_cos=0.9, any_cos=0.999,
                                       query_ratio=0.25, min_query=2, seed=0)
            with open(root / P.METADATA_NAME, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), P.METADATA_COLUMNS)
            by_group = {}
            for row in rows:
                by_group.setdefault(row["dup_group"], set()).add(row[P.SPLIT_COL])
                self.assertEqual(row["split_train_test"], {"database": "train", "query": "test"}[row[P.SPLIT_COL]])
            self.assertTrue(all(len(sides) == 1 for sides in by_group.values()))
            self.assertEqual(manifest["leakage"]["cross_side_adjacent_similar_pairs"], 0)
            self.assertEqual(manifest["counts"]["groups_on_both_sides"], 0)
            self.assertEqual(manifest["outputs"]["metadata_sha256"], P.sha256_file(root / P.METADATA_NAME))
            self.assertEqual(manifest["source"]["base_sha256"], P.sha256_file(root / P.BASE_NAME))

    def test_embeddings_must_match_the_base_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self._base(root)
            with self.assertRaises(P.PreparationError):
                P.load_embeddings(root / P.EMBEDDINGS_NAME, [r["image_id"] for r in rows][::-1])
            np.savez(root / P.EMBEDDINGS_NAME, image_id=np.array([r["image_id"] for r in rows]),
                     emb=np.ones((len(rows), 8), np.float32))
            with self.assertRaises(P.PreparationError):
                P.load_embeddings(root / P.EMBEDDINGS_NAME, [r["image_id"] for r in rows])


@pytest.mark.data
class PreparedDatasetTests(unittest.TestCase):
    """Checks the real JaguarReID metadata when it exists on this machine."""

    def setUp(self):
        self.metadata = P.DEFAULT_ROOT / P.METADATA_NAME
        if not self.metadata.is_file():
            self.skipTest("prepared Jaguar metadata not present")
        with open(self.metadata, newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.manifest = json.loads((P.DEFAULT_ROOT / P.MANIFEST_NAME).read_text())

    def test_closed_set_and_groups_never_split(self):
        database = {r["identity"] for r in self.rows if r[P.SPLIT_COL] == "database"}
        query = {r["identity"] for r in self.rows if r[P.SPLIT_COL] == "query"}
        self.assertEqual(database, query)
        sides = {}
        for row in self.rows:
            sides.setdefault(row["dup_group"], set()).add(row[P.SPLIT_COL])
        self.assertTrue(all(len(s) == 1 for s in sides.values()))

    def test_manifest_matches_metadata_and_base(self):
        self.assertEqual(self.manifest["outputs"]["metadata_sha256"], P.sha256_file(self.metadata))
        self.assertEqual(self.manifest["source"]["base_sha256"], P.sha256_file(P.DEFAULT_ROOT / P.BASE_NAME))
        self.assertEqual(self.manifest["counts"]["images"], len(self.rows))

    def test_no_burst_pair_crosses_the_sides(self):
        leakage = self.manifest["leakage"]
        self.assertEqual(leakage["cross_side_adjacent_similar_pairs"], 0)
        self.assertEqual(leakage["cross_side_pairs_at_any_cos"], 0)
        threshold = self.manifest["parameters"]["hash_threshold"]
        for key, count in leakage.items():
            if key.startswith("cross_side_same_identity_pairs_le_") and int(key.rsplit("_", 1)[1]) <= threshold:
                self.assertEqual(count, 0, key)


if __name__ == "__main__":
    unittest.main()
