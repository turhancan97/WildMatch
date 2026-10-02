"""Static checks for the MkDocs project page (no mkdocs install required).

The page is built from ``docs/`` with ``mkdocs.yml`` at the repository root. These
tests keep the configuration and the Markdown sources consistent without running
MkDocs, which is not part of the core environment.
"""

import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"

# Paths and identifiers that must never leak into the public page sources.
FORBIDDEN_PATTERNS = (
    re.compile(r"/shared/"),
    re.compile(r"/home/kargin"),
    re.compile(r"claude\.ai/code/session"),
)
# The venue must not be named anywhere on the site until the authors announce it
# (user decision 2026-10-02); the submission status is never described either.
FORBIDDEN_PHRASES = ("ECIR", "LNCS", "Springer", "under review", "Submitted to", "submitted to")


def _nav_paths(items) -> list[str]:
    paths: list[str] = []
    for item in items:
        if isinstance(item, str):
            paths.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str):
                    paths.append(value)
                else:
                    paths.extend(_nav_paths(value))
        elif isinstance(item, list):
            paths.extend(_nav_paths(item))
    return paths


class ProjectPageConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_config_is_strict_and_uses_docs_dir(self):
        self.assertTrue(self.config.get("strict"), "mkdocs.yml must set strict: true")
        self.assertEqual(self.config.get("docs_dir"), "docs")
        self.assertEqual(self.config.get("theme", {}).get("name"), "material")
        self.assertEqual(self.config.get("theme", {}).get("custom_dir"), "overrides")
        self.assertTrue((REPO_ROOT / "overrides" / "main.html").is_file())

    def test_every_nav_entry_exists(self):
        for rel in _nav_paths(self.config["nav"]):
            with self.subTest(page=rel):
                self.assertTrue((DOCS_DIR / rel).is_file(), f"nav entry {rel} is missing")

    def test_every_markdown_page_is_in_nav(self):
        nav = set(_nav_paths(self.config["nav"]))
        pages = {str(p.relative_to(DOCS_DIR)) for p in DOCS_DIR.rglob("*.md")}
        self.assertEqual(pages - nav, set(), "Markdown pages outside the nav")

    def test_local_assets_exist(self):
        for rel in self.config.get("extra_css", []):
            self.assertTrue((DOCS_DIR / rel).is_file(), rel)
        for entry in self.config.get("extra_javascript", []):
            path = entry["path"] if isinstance(entry, dict) else entry
            if path.startswith(("http://", "https://")):
                continue
            self.assertTrue((DOCS_DIR / path).is_file(), path)
        for key in ("favicon", "logo"):
            rel = self.config.get("theme", {}).get(key)
            if rel:
                self.assertTrue((DOCS_DIR / rel).is_file(), rel)

    def test_interactive_mount_points_have_modules(self):
        page = (DOCS_DIR / "assets" / "js" / "page.js").read_text(encoding="utf-8")
        mount_ids = set(re.findall(r'\["(wm-[a-z-]+)", mount', page))
        self.assertTrue(mount_ids, "page.js declares no mount points")
        used = set()
        for md in DOCS_DIR.rglob("*.md"):
            used.update(re.findall(r'id="(wm-[a-z-]+)" class="wm-widget"', md.read_text(encoding="utf-8")))
        self.assertEqual(used - mount_ids, set(), "mount points in pages without a module")
        for imported in re.findall(r'from "\./([a-z-]+\.js)"', page):
            self.assertTrue((DOCS_DIR / "assets" / "js" / imported).is_file(), imported)
        modules = [e for e in self.config.get("extra_javascript", []) if isinstance(e, dict) and e.get("type") == "module"]
        self.assertTrue(any(e["path"] == "assets/js/page.js" for e in modules), "page.js is not registered as a module")

    def test_brand_palette_is_defined(self):
        css = (DOCS_DIR / "assets" / "css" / "extra.css").read_text(encoding="utf-8").lower()
        for colour in ("#3a7eab", "#cf4832", "#d1d3d4"):
            self.assertIn(colour, css, f"brand colour {colour} missing from extra.css")
        palette = self.config["theme"]["palette"]
        for entry in palette:
            self.assertEqual(entry.get("primary"), "custom")
            self.assertEqual(entry.get("accent"), "custom")

    def test_site_output_is_ignored(self):
        ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/site", [line.strip() for line in ignore])


class ProjectPageContentTests(unittest.TestCase):
    def _sources(self):
        yield from DOCS_DIR.rglob("*.md")
        yield from DOCS_DIR.rglob("*.css")
        yield from DOCS_DIR.rglob("*.js")
        yield from (REPO_ROOT / "overrides").rglob("*.html")
        yield CONFIG_PATH

    def test_no_private_paths_or_forbidden_phrases(self):
        for path in self._sources():
            text = path.read_text(encoding="utf-8")
            for pattern in FORBIDDEN_PATTERNS:
                self.assertIsNone(pattern.search(text), f"{path}: matches {pattern.pattern}")
            for phrase in FORBIDDEN_PHRASES:
                self.assertNotIn(phrase, text, f"{path}: contains {phrase!r}")

    def test_relative_links_and_images_resolve(self):
        link = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
        for page in DOCS_DIR.rglob("*.md"):
            text = page.read_text(encoding="utf-8")
            for target in link.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target_path = target.split("#", 1)[0]
                resolved = (page.parent / target_path).resolve()
                with self.subTest(page=page.name, target=target):
                    self.assertTrue(resolved.exists(), f"{page}: broken link {target}")


if __name__ == "__main__":
    unittest.main()
