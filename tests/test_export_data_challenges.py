import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import export_data_challenges as dc  # noqa: E402


class ExampleListTests(unittest.TestCase):
    def test_examples_reference_known_categories_and_datasets(self):
        categories = {c.key for c in dc.CATEGORIES}
        for example in dc.EXAMPLES:
            self.assertIn(example.category, categories, example)
            self.assertIn(example.dataset, dc.PAGE_DATASETS, example)
            self.assertTrue(example.note, example)

    def test_examples_are_unique_per_photo(self):
        keys = [(e.dataset, e.row) for e in dc.EXAMPLES]
        self.assertEqual(len(keys), len(set(keys)), "a photo appears twice")

    def test_every_category_has_examples(self):
        used = {e.category for e in dc.EXAMPLES}
        self.assertEqual(used, {c.key for c in dc.CATEGORIES})

    def test_page_mount_points_match_categories(self):
        page = (REPO_ROOT / "docs" / "datasets" / "challenges.md").read_text(encoding="utf-8")
        mounts = set(re.findall(r'id="wm-challenge-([a-z_]+)"', page))
        self.assertEqual(mounts, {c.key for c in dc.CATEGORIES})


class CommittedExportTests(unittest.TestCase):
    JSON = dc.WEB_DIR / "challenges.json"

    def setUp(self):
        if not self.JSON.is_file():
            self.skipTest("challenges.json not exported")
        self.data = json.loads(self.JSON.read_text(encoding="utf-8"))

    def test_export_matches_examples(self):
        exported = {(i["dataset"], i["row"]): i for i in self.data["items"]}
        self.assertEqual(set(exported), {(e.dataset, e.row) for e in dc.EXAMPLES})
        for item in self.data["items"]:
            self.assertTrue((dc.WEB_DIR / item["image"]["file"]).is_file(), item["image"]["file"])
            self.assertLessEqual(max(item["image"]["width"], item["image"]["height"]), dc.WEB_LONG_SIDE)
            self.assertIn(item["group"], {"noise", "difficulty"})
            self.assertEqual(len(item["source_sha256"]), 64)

    def test_no_private_paths(self):
        for path in (self.JSON, dc.SUMMARY_OUT):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/shared/", text)
            self.assertNotIn("/home/", text)

    def test_summary_rows_cover_page_datasets(self):
        summary = json.loads(dc.SUMMARY_OUT.read_text(encoding="utf-8"))
        self.assertEqual({r["dataset"] for r in summary["rows"]}, set(dc.PAGE_DATASETS))
        for row in summary["rows"]:
            self.assertGreater(row["images"], 0)
            self.assertLessEqual(row["any_flag_images"], row["images"])


if __name__ == "__main__":
    unittest.main()
