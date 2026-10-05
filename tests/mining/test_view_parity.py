"""Parity L2: a view rebuilt by wildmatch.mining equals the view the paper's training read.

Views are symlink trees; the comparison covers every directory, every symlink target and every
regular file except `manifest.json`, whose `output_root` names where the view was built. The full
check over all eight paper datasets (2026-10-05) found no other difference; this test repeats it for
the smallest one, SalamanderID2025.
"""

import os
from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import config_from_registry, prepare

ROOT = Path(__file__).resolve().parents[2]
LIVE_VIEW = Path("/shared/sets/datasets/vision/czechlynx/wildlife_processed/SalamanderID2025/legacy")


def tree(root: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if rel == "manifest.json":
                continue
            if p.is_symlink():
                out[rel] = ("link", os.readlink(p))
            elif p.is_dir():
                out[rel] = ("dir", "")
            else:
                out[rel] = ("file", p.read_bytes().hex())
    return out


@pytest.mark.data
def test_salamander_legacy_view_is_rebuilt_identically(tmp_path):
    if not LIVE_VIEW.is_dir():
        pytest.skip(f"{LIVE_VIEW} not available")
    output = tmp_path / "SalamanderID2025" / "legacy"
    prepare(config_from_registry("salamander", "gmum"), output, "legacy")
    rebuilt, live = tree(output), tree(LIVE_VIEW)
    assert sorted(set(rebuilt) ^ set(live)) == []
    assert [k for k in live if rebuilt[k] != live[k]] == []
