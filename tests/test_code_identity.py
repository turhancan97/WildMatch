"""Run manifests record the code they ran (commit, uncommitted changes) and checkpoint hashes."""

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wildmatch.reporting import artifacts
from wildmatch.reporting.artifacts import RunContext, _code_identity, file_identity


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(root: Path) -> Path:
    package = root / "src" / "wildmatch"
    package.mkdir(parents=True)
    (package / "module.py").write_text("x = 1\n", encoding="utf-8")
    (root / "notes.md").write_text("notes\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    return package


class CodeIdentityTests(unittest.TestCase):
    def test_clean_modified_and_untracked_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = _repo(root)
            clean = _code_identity(package)
            self.assertEqual(len(clean["commit"]), 40)
            self.assertFalse(clean["dirty"])
            self.assertIsNone(clean["diff_sha256"])
            # edits outside src/, pyproject.toml and uv.lock do not make a run dirty
            (root / "notes.md").write_text("changed\n", encoding="utf-8")
            self.assertFalse(_code_identity(package)["dirty"])

            (package / "module.py").write_text("x = 2\n", encoding="utf-8")
            modified = _code_identity(package)
            self.assertTrue(modified["dirty"])
            self.assertEqual(modified["changed_paths"], ["src/wildmatch/module.py"])
            self.assertIn("+x = 2", modified["diff"])
            self.assertEqual(modified["diff_sha256"], hashlib.sha256(modified["diff"].encode()).hexdigest())

            (package / "new.py").write_text("y = 3\n", encoding="utf-8")
            untracked = _code_identity(package)
            self.assertEqual(untracked["changed_paths"], ["src/wildmatch/module.py", "src/wildmatch/new.py"])
            self.assertIn("--- untracked: src/wildmatch/new.py\ny = 3", untracked["diff"])
            self.assertNotEqual(untracked["diff_sha256"], modified["diff_sha256"])

    def test_code_outside_git_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = _code_identity(Path(tmp))
            self.assertEqual(identity["commit"], "unknown")
            self.assertIsNone(identity["dirty"])

    def test_manifest_records_code_and_writes_the_diff_when_dirty(self):
        identity = {
            "commit": "c" * 40,
            "dirty": True,
            "checkout": "/x",
            "changed_paths": ["src/a.py"],
            "diff_sha256": "d",
            "diff": "+change\n",
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(artifacts, "code_identity", return_value=identity):
            run = Path(tmp)
            context = RunContext(
                "probe", "r", "t", "h", run, run / "c.yaml", run / "m.json", run / "x.json", run / "t.json", run / "v"
            )
            manifest = context.write_manifest({})
            self.assertEqual(manifest["git_commit"], "c" * 40)
            self.assertEqual(manifest["code"]["changed_paths"], ["src/a.py"])
            self.assertNotIn("diff", json.loads((run / "m.json").read_text())["code"])
            self.assertEqual((run / "code.diff").read_text(), "+change\n")
        identity["dirty"] = False
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(artifacts, "code_identity", return_value=identity):
            run = Path(tmp)
            context = RunContext(
                "probe", "r", "t", "h", run, run / "c.yaml", run / "m.json", run / "x.json", run / "t.json", run / "v"
            )
            context.write_manifest({})
            self.assertFalse((run / "code.diff").exists())

    def test_file_identity_has_the_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint-final.pth"
            checkpoint.write_bytes(b"weights")
            self.assertEqual(file_identity(checkpoint)["sha256"], hashlib.sha256(b"weights").hexdigest())
            self.assertNotIn("sha256", file_identity(Path(tmp) / "missing.pth"))


if __name__ == "__main__":
    unittest.main()
