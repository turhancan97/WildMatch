import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is part of the shared environment
    Image = None

REPO_ROOT = Path(__file__).resolve().parents[1]

from scripts import build_demo_cards as cards  # noqa: E402


@unittest.skipIf(Image is None, "Pillow is required")
class ComposeHelpersTests(unittest.TestCase):
    def test_cover_fills_and_crops(self):
        wide = Image.new("RGB", (400, 100), (10, 20, 30))
        tall = Image.new("RGB", (100, 400), (10, 20, 30))
        for image in (wide, tall):
            self.assertEqual(cards.cover(image, (480, 300)).size, (480, 300))

    def test_compose_row_has_card_size_and_rules(self):
        images = [Image.new("RGB", (120, 90), (c, c, c)) for c in (40, 120, 200)]
        card = cards.compose_row(images, rules=[None, cards.BLUE, cards.RED])
        self.assertEqual(card.size, cards.CARD_SIZE)
        width, height = cards.CARD_SIZE
        column = (width - cards.GUTTER * 2) // 3
        # the rule bar sits under the second and third columns only
        self.assertEqual(card.getpixel((column + cards.GUTTER + 5, height - 2)), cards.BLUE)
        self.assertEqual(card.getpixel((2 * (column + cards.GUTTER) + 5, height - 2)), cards.RED)
        self.assertNotEqual(card.getpixel((5, height - 2)), cards.BLUE)
        with self.assertRaises(ValueError):
            cards.compose_row([])

    def test_compose_split_masks_right_half_only(self):
        raw = Image.new("RGB", (200, 200), (200, 150, 100))
        mask = Image.new("L", (200, 200), 0)
        card = cards.compose_split(raw, mask)
        self.assertEqual(card.size, cards.CARD_SIZE)
        self.assertEqual(card.getpixel((20, 20)), (200, 150, 100))  # raw on the left
        self.assertEqual(card.getpixel((cards.CARD_SIZE[0] - 20, 20)), (0, 0, 0))  # masked on the right

    def test_builders_fail_closed_on_missing_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in cards.CARD_NAMES:
                with self.subTest(card=name), self.assertRaises(FileNotFoundError):
                    cards.BUILDERS[name](Path(tmp))


class CommittedCardsTests(unittest.TestCase):
    """The hub page must reference exactly the committed cards, one per demo page."""

    def test_committed_cards_match_hub_page(self):
        hub = (REPO_ROOT / "docs" / "demo" / "index.md").read_text(encoding="utf-8")
        referenced = set(re.findall(r"assets/demo/cards/([a-z-]+)\.jpg", hub))
        self.assertEqual(referenced, set(cards.CARD_NAMES))
        for name in cards.CARD_NAMES:
            path = cards.CARD_DIR / f"{name}.jpg"
            self.assertTrue(path.is_file(), f"missing card {path}")
            if Image is not None:
                with Image.open(path) as image:
                    self.assertEqual(image.size, cards.CARD_SIZE, name)
        links = set(re.findall(r'class="wm-hub-card" href="([a-z-]+)/"', hub))
        self.assertEqual(links, set(cards.CARD_NAMES))
        for name in links:
            self.assertTrue((REPO_ROOT / "docs" / "demo" / f"{name}.md").is_file(), name)

    def test_card_sources_are_committed_exports(self):
        for rel in ("before_after/before_after.json", "rank_change/rank_change.json",
                    "mined_pairs/mined_pairs.json", "synthetic/synthetic_demo.json",
                    "masking/masking_demo.json"):
            path = cards.DEMO_ROOT / rel
            self.assertTrue(path.is_file(), rel)
            json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
