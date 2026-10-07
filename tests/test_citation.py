"""CITATION.cff stays valid-looking and in step with the package (GitHub's "Cite this repository")."""

import tomllib
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class CitationTest(unittest.TestCase):
    def test_citation_matches_package_and_preprint(self):
        citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(citation["cff-version"], "1.2.0")
        self.assertEqual(str(citation["version"]), project["version"])
        self.assertEqual(citation["license"], "Apache-2.0")
        preferred = citation["preferred-citation"]
        self.assertEqual(preferred["doi"], "10.48550/arXiv.2610.07384")
        names = [(a["given-names"], a["family-names"]) for a in citation["authors"]]
        self.assertEqual(names, [(a["given-names"], a["family-names"]) for a in preferred["authors"]])
        self.assertEqual(len(names), 6)


if __name__ == "__main__":
    unittest.main()
