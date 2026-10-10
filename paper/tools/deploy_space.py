#!/usr/bin/env python3
"""Stage and upload the Hugging Face Space (``space/``) with the ``wildmatch`` package beside ``app.py``.

Plain pip cannot install ``wildmatch`` from git (its git-only dependencies are declared as uv sources), so the
package source of the current commit is copied into the staged Space instead, and ``WILDMATCH_COMMIT`` records
which commit it is. The working tree must be clean under ``src/`` and ``space/`` so the upload matches a commit.

    python paper/tools/deploy_space.py --dry-run                 # stage into a temporary folder, list it
    python paper/tools/deploy_space.py --repo turhancan97/wildmatch   # create (if needed) and upload

Uploading needs a write token (``HF_TOKEN`` or ``hf auth login``). The Space hardware (ZeroGPU) and the
``HF_TOKEN`` secret with access to ``facebook/sam3`` are set in the Space settings.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
MAX_SIZE_MB = 200


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def stage(target: Path, allow_dirty: bool = False) -> List[Path]:
    """Copy ``space/`` and ``src/wildmatch`` into ``target``; returns the staged files."""
    dirty = git("status", "--porcelain", "--", "src", "space")
    if dirty and not allow_dirty:
        raise SystemExit(f"uncommitted changes under src/ or space/; commit first:\n{dirty}")
    shutil.copytree(REPO_ROOT / "space", target, dirs_exist_ok=True, ignore=IGNORE)
    shutil.copytree(REPO_ROOT / "src" / "wildmatch", target / "wildmatch", ignore=IGNORE)
    (target / "WILDMATCH_COMMIT").write_text(git("rev-parse", "HEAD") + ("-dirty" if dirty else "") + "\n")
    files = sorted(p for p in target.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files) / 1e6
    if size > MAX_SIZE_MB:
        raise SystemExit(f"staged Space is {size:.0f} MB (> {MAX_SIZE_MB} MB)")
    return files


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default="turhancan97/wildmatch", help="Space repository id")
    parser.add_argument("--dry-run", action="store_true", help="stage and list only; upload nothing")
    parser.add_argument("--private", action="store_true", help="create the Space private")
    parser.add_argument("--allow-dirty", action="store_true", help="stage uncommitted changes (dry runs only)")
    args = parser.parse_args(argv)
    if args.allow_dirty and not args.dry_run:
        raise SystemExit("--allow-dirty is for dry runs only")
    with tempfile.TemporaryDirectory(prefix="wildmatch-space-") as tmp:
        files = stage(Path(tmp), args.allow_dirty)
        size = sum(p.stat().st_size for p in files) / 1e6
        print(f"{len(files)} files, {size:.1f} MB, commit {(Path(tmp) / 'WILDMATCH_COMMIT').read_text().strip()}")
        for top in sorted({p.relative_to(tmp).parts[0] for p in files}):
            print(f"  {top}")
        if args.dry_run:
            return 0
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
        info = api.upload_folder(
            folder_path=tmp,
            repo_id=args.repo,
            repo_type="space",
            commit_message=f"Deploy WildMatch {(Path(tmp) / 'WILDMATCH_COMMIT').read_text().strip()[:12]}",
            # Remove files a previous deploy uploaded but this one no longer has (the Hub keeps .gitattributes).
            delete_patterns=[
                "*.py",
                "*.txt",
                "*.md",
                "WILDMATCH_COMMIT",
                "wildmatch/**",
                "wildmatch_space/**",
                "gallery/**",
                "examples/**",
            ],
        )
        print(info.commit_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
