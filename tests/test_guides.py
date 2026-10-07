"""Static checks for README.md and the guides in ``guides/`` (no network, no MkDocs).

The README links into the guides and the guides link to each other by heading anchors, so a
renamed heading or a moved file breaks them silently on GitHub. These tests resolve every
relative link and anchor, and keep cluster paths and the review venue out of these public files.
"""

import base64
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "guides").glob("*.md"))]

FORBIDDEN_PATTERNS = (
    re.compile(r"/shared/"),
    re.compile(r"/home/kargin"),
    re.compile(r"claude\.ai/code/session"),
)
# Same embargo rule as the project page (tests/test_project_page.py).
# Venue and review-status words, base64-encoded so this public file does not name the venue.
FORBIDDEN_PHRASES = tuple(
    base64.b64decode("RUNJUnxMTkNTfFNwcmluZ2VyfHVuZGVyIHJldmlld3xTdWJtaXR0ZWQgdG98c3VibWl0dGVkIHRv").decode().split("|")
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK = re.compile(r"""(?:href|src)="([^"]+)\"""")


def _github_anchors(path: Path) -> set[str]:
    """Heading anchors as GitHub renders them (headings inside code fences excluded)."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            continue
        match = None if in_fence else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = re.sub(r"[^\w\- ]", "", match.group(1).strip().lower()).replace(" ", "-")
        seen = counts.get(slug, 0)
        anchors.add(slug if seen == 0 else f"{slug}-{seen}")
        counts[slug] = seen + 1
    return anchors


class GuidesTest(unittest.TestCase):
    def test_guides_exist(self):
        names = {page.name for page in PAGES}
        for name in ("INSTALLATION.md", "DATASET.md", "CONFIGURATION.md", "EXPERIMENTS.md"):
            with self.subTest(name=name):
                self.assertIn(name, names)
        self.assertFalse((REPO_ROOT / "DATASET.md").exists(), "DATASET.md moved to guides/")

    def test_relative_links_and_anchors_resolve(self):
        for page in PAGES:
            text = page.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                path_part, _, anchor = target.partition("#")
                resolved = (page.parent / path_part).resolve() if path_part else page
                with self.subTest(page=page.name, target=target):
                    self.assertTrue(resolved.exists(), f"{page.name}: broken link {target}")
                    if anchor and resolved.suffix == ".md":
                        self.assertIn(anchor, _github_anchors(resolved), f"{page.name}: missing anchor {target}")

    def test_no_cluster_paths_or_venue(self):
        for page in PAGES:
            text = page.read_text(encoding="utf-8")
            with self.subTest(page=page.name):
                for pattern in FORBIDDEN_PATTERNS:
                    self.assertIsNone(pattern.search(text), f"{page.name} contains {pattern.pattern}")
                for phrase in FORBIDDEN_PHRASES:
                    self.assertNotIn(phrase, text, f"{page.name} names {phrase!r}")


if __name__ == "__main__":
    unittest.main()
