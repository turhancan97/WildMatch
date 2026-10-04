"""Dataset/split profiles reported in the paper, plus benchmark-only datasets.

Built from the dataset registry (``conf/dataset/<key>.yaml``, see
:mod:`wildmatch.data.registry`), resolved against the active path profile
(:mod:`wildmatch.paths`). Each registry entry with a ``registry.paper_key`` becomes a
profile under that key; the order follows ``registry.order``.

``PAPER_PROFILES`` is exactly the paper's datasets; the paper figures and the project
page look profiles up there. ``BENCHMARK_ONLY_PROFILES`` holds datasets that run through
the same pipeline but are not in the paper; dataset-level tools (class balance, image
quality audit) iterate ``ALL_PROFILES``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Optional

from wildmatch.data.registry import load_registry


class PaperProfile(NamedTuple):
    key: str
    label: str
    source: str
    metadata: Path
    identity_col: str
    split_col: str
    database_value: str
    query_value: str
    dataset_name: str
    animal: str
    root: Path
    # RLE mask column applied at load time (dataset.no_background=true); None
    # when the metadata already points at pre-masked image files.
    mask_col: Optional[str]


def _paper_metadata(entry) -> str:
    """The table the paper's runs read: ``registry.paper_inputs.metadata_file`` when the entry
    has moved on to new inputs (WildlifeReID-10k SAM 3 masks), else ``metadata_file``."""
    paper = entry.registry.get("paper_inputs")
    return str(paper.metadata_file) if paper is not None else str(entry.metadata_file)


def _profile(entry) -> PaperProfile:
    root = Path(str(entry.root))
    return PaperProfile(
        str(entry.registry.paper_key), str(entry.registry.label), str(entry.registry.source),
        root / str(_paper_metadata(entry)), str(entry.label_col), str(entry.split_col),
        str(entry.database_split_value), str(entry.query_split_value), str(entry.name), str(entry.animal),
        root, str(entry.mask_col) if bool(entry.no_background) else None,
    )


def _profiles(paper: bool) -> List[PaperProfile]:
    entries = [
        entry for entry in load_registry().values()
        if entry.registry.paper_key is not None and bool(entry.registry.paper) is paper
    ]
    return [_profile(entry) for entry in sorted(entries, key=lambda entry: int(entry.registry.order))]


PAPER_PROFILES = _profiles(paper=True)
BENCHMARK_ONLY_PROFILES = _profiles(paper=False)
ALL_PROFILES = PAPER_PROFILES + BENCHMARK_ONLY_PROFILES
