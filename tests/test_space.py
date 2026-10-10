"""The Hugging Face Space (space/): species table, drawing, ranking, bundled data and Space card (CPU, no downloads)."""

import base64
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SPACE = REPO_ROOT / "space"
sys.path.insert(0, str(SPACE))

from wildmatch_space import drawing, retrieval  # noqa: E402
from wildmatch_space.species import DEFAULT_SPECIES, MATCHERS, SPECIES  # noqa: E402

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv"}
FORBIDDEN_PATTERNS = (re.compile(r"/shared/"), re.compile(r"/home/kargin"), re.compile(r"claude\.ai/code/session"))
# Same embargo rule as the project page (tests/test_project_page.py), base64-encoded so this file does not name the venue.
FORBIDDEN_PHRASES = tuple(
    base64.b64decode("RUNJUnxMTkNTfFNwcmluZ2VyfHVuZGVyIHJldmlld3xTdWJtaXR0ZWQgdG98c3VibWl0dGVkIHRv").decode().split("|")
)


def front_matter(text: str) -> dict:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert match, "no YAML front matter"
    return yaml.safe_load(match.group(1))


class SpeciesTests(unittest.TestCase):
    def test_every_species_has_both_default_set_checkpoints(self):
        from wildmatch import weights

        entries = weights.select(weights.load_manifest()["entries"])
        published = {(e["dataset"], e["matcher"]) for e in entries}
        self.assertIn(DEFAULT_SPECIES, SPECIES)
        for species in SPECIES.values():
            for matcher in MATCHERS.values():
                with self.subTest(species=species.label, matcher=matcher):
                    self.assertIn((species.dataset, matcher), published)

    def test_merge_and_first_prompt_follow_the_registry(self):
        from wildmatch.data.registry import load_dataset

        for species in SPECIES.values():
            prepare = load_dataset(species.dataset).registry.get("prepare") or {}
            with self.subTest(species=species.label):
                self.assertEqual(species.prompts[-1], "Animal")
                if prepare.get("merge"):
                    self.assertEqual(species.merge, prepare["merge"])
                for retry in prepare.get("retry_prompts") or []:
                    self.assertIn(retry, species.prompts)


class DrawingTests(unittest.TestCase):
    def test_processed_to_raw_pixels_uses_pixel_centres(self):
        points = drawing.processed_to_raw_pixels(np.array([[-0.5, -0.5], [255.5, 127.5]]), (128, 256), (256, 512))
        np.testing.assert_allclose(points, [[-0.5, -0.5], [511.5, 255.5]])

    def test_match_figure_draws_at_most_the_requested_lines(self):
        rng = np.random.default_rng(0)
        left = Image.fromarray(rng.integers(0, 255, (200, 300, 3), dtype=np.uint8))
        right = Image.fromarray(rng.integers(0, 255, (240, 260, 3), dtype=np.uint8))
        kl = rng.uniform([10, 10], [290, 190], (60, 2))
        kr = rng.uniform([10, 10], [250, 230], (60, 2))
        figure, drawn = drawing.match_figure(left, right, (None, None), kl, kr, rng.uniform(size=60), lines=12)
        self.assertIsInstance(figure, Image.Image)
        self.assertLessEqual(drawn, 12)
        self.assertGreater(drawn, 0)
        _, none = drawing.match_figure(left, right, (None, None), np.empty((0, 2)), np.empty((0, 2)), np.empty(0))
        self.assertEqual(none, 0)


class RankingTests(unittest.TestCase):
    def test_candidates_break_ties_by_lowest_index(self):
        gallery = retrieval.Gallery("t", Path("."), pd.DataFrame({"identity": list("abcd")}), np.eye(4)[[0, 1, 1, 2]])
        self.assertEqual(retrieval.candidates(np.array([0, 1, 0, 0]), gallery, 3).tolist(), [1, 2, 0])

    def test_order_by_score_keeps_lower_index_first_on_ties(self):
        self.assertEqual(retrieval.order_by_score(np.array([7, 3, 5]), np.array([0.5, 0.5, 0.9])).tolist(), [5, 3, 7])


class BundledDataTests(unittest.TestCase):
    def test_galleries_are_consistent_and_cc_by(self):
        names = retrieval.available()
        self.assertEqual(names, ["czechlynx_unseen", "synthetic"])
        for name in names:
            gallery = retrieval.load(name)
            with self.subTest(gallery=name):
                self.assertEqual(len(gallery.embeddings), len(gallery.metadata))
                np.testing.assert_allclose(np.linalg.norm(gallery.embeddings, axis=1), 1.0, atol=1e-4)
                self.assertEqual(set(gallery.metadata["licence"]), {"CC BY 4.0"})
                self.assertTrue(gallery.metadata["source"].str.startswith("CzechLynx").all())
                for column in ("raw", "masked"):
                    self.assertTrue(all((gallery.root / p).is_file() for p in gallery.metadata[column]))
                self.assertTrue((gallery.root / "ATTRIBUTION.md").is_file())
        self.assertEqual(len(retrieval.load("czechlynx_unseen").metadata), 160)

    def test_only_czechlynx_photos_are_bundled(self):
        examples = pd.read_csv(SPACE / "examples" / "examples.csv")
        self.assertTrue(examples["source"].str.startswith("CzechLynx").all())
        self.assertTrue((SPACE / "examples" / "ATTRIBUTION.md").is_file())
        photos = {p.name for p in (SPACE / "examples").glob("*.jpg")}
        self.assertEqual(photos - set(examples["file"]), {f"synthetic_query_{k}.jpg" for k in range(6)})

    def test_no_cluster_paths_or_venue(self):
        for path in sorted(p for p in SPACE.rglob("*") if p.suffix in TEXT_SUFFIXES):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(SPACE))):
                for pattern in FORBIDDEN_PATTERNS:
                    self.assertIsNone(pattern.search(text))
                for phrase in FORBIDDEN_PHRASES:
                    self.assertNotIn(phrase, text)


class SpaceCardTests(unittest.TestCase):
    def test_front_matter(self):
        meta = front_matter((SPACE / "README.md").read_text(encoding="utf-8"))
        self.assertEqual(meta["sdk"], "gradio")
        self.assertEqual(meta["app_file"], "app.py")
        self.assertEqual(meta["license"], "cc-by-nc-4.0")
        self.assertEqual(str(meta["python_version"]), "3.12")
        self.assertIn("turhancan97/wildmatch-checkpoints", meta["models"])

    def test_bibtex_matches_the_repository_readme(self):
        pattern = re.compile(r"@misc\{kargin2026wildmatch,.*?\n\}", re.S)
        ours = pattern.search((SPACE / "README.md").read_text(encoding="utf-8")).group(0)
        self.assertEqual(ours, pattern.search((REPO_ROOT / "README.md").read_text(encoding="utf-8")).group(0))

    def test_requirements_use_blackwell_torch_and_leave_out_no_deps_packages(self):
        lines = [
            line.split(";")[0].strip()
            for line in (SPACE / "requirements.txt").read_text().splitlines()
            if line and not line.startswith("#") and "sys_platform == 'never'" not in line
        ]
        self.assertIn("torch==2.8.0+cu128", lines)
        self.assertIn("torchvision==0.23.0+cu128", lines)
        self.assertFalse([line for line in lines if line.startswith(("vismatch", "uniception", "nvidia-"))])

    def test_no_deps_commits_match_the_lock(self):
        app = (SPACE / "app.py").read_text(encoding="utf-8")
        vismatch = re.search(r"vismatch\.git@([0-9a-f]{40})", app).group(1)
        self.assertIn(f"vismatch.git?rev={vismatch}", (REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
