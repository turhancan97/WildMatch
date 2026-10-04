"""Every name the backbone fine-tuning runner uses at module level resolves.

Guards the bug fixed on 2026-10-04: `file_identity` was called when writing the completed-run
manifest but never imported (since 2026-08-13), so every fine-tuning run would have ended with a
NameError after training. Static check with ruff's undefined-name rule when ruff is available.
"""

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class FinetuneImportTests(unittest.TestCase):
    def test_file_identity_is_imported(self):
        source = (ROOT / "src/wildmatch/train/finetune_runner.py").read_text(encoding="utf-8")
        self.assertIn("file_identity(", source)
        import ast

        tree = ast.parse(source)
        imported = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        self.assertIn("file_identity", imported)

    @unittest.skipUnless(shutil.which("ruff"), "ruff not installed")
    def test_no_undefined_names_in_the_package(self):
        result = subprocess.run(
            ["ruff", "check", "--select", "F821", "--output-format", "concise", "src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
