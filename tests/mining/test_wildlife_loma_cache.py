from pathlib import Path
from types import SimpleNamespace

from scripts.lynx_build_loma_cache import enumerate_frames


def test_loma_cache_enumerates_wildlife_layout(tmp_path: Path):
    for split in ("train", "test"):
        frame = tmp_path / split / "animal-1" / "animal-1" / "frame_000000.jpg"
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b"placeholder")

    args = SimpleNamespace(
        dataset_root=tmp_path,
        splits=["train", "test"],
        all_frames=True,
        frames_per_seq=20,
        index=[],
    )

    assert enumerate_frames(args) == [
        "test/animal-1/animal-1/frame_000000.jpg",
        "train/animal-1/animal-1/frame_000000.jpg",
    ]
