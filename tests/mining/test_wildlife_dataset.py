from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import (
    assign_generated_splits,
    build_records,
    list_collections,
    load_config,
    prepare,
    read_rows,
)


def make_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    masked = root / "masked_images"
    rows = []
    for identity in ("a", "b"):
        for index in range(3):
            relative = Path(identity) / f"frame_{index}.jpg"
            target = masked / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"not-an-image-but-a-valid-file")
            rows.append({"path": f"masked_images/{relative}", "identity": identity, "split": "train"})
    test = masked / "a" / "test.jpg"
    test.write_bytes(b"test")
    rows.append({"path": "masked_images/a/test.jpg", "identity": "a", "split": "test"})
    rows.append({"path": "masked_images/a/missing.jpg", "identity": "unknown", "split": "train"})
    metadata = root / "metadata.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "identity", "split"])
        writer.writeheader()
        writer.writerows(rows)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "fixture",
                "source_root": str(root),
                "metadata_csv": "metadata.csv",
                "collection_rule": "identity",
                "validation_fraction": 0.2,
                "seed": 7,
            }
        )
    )
    return config_path


def test_strict_split_and_symlink_view(tmp_path: Path):
    config = load_config(make_fixture(tmp_path))
    summary = prepare(config, tmp_path / "view", "strict")
    assert summary["frames"] == 7
    assert summary["frames_by_split"] == {"test": 1, "train": 4, "val": 2}
    assert summary["excluded_rows"] == 1
    assert (tmp_path / "view" / "val").is_dir()
    links = list((tmp_path / "view").rglob("frame_*.jpg"))
    assert links and all(path.is_symlink() for path in links)
    assert all(path.resolve().is_relative_to(tmp_path / "source") for path in links)
    assert len(list_collections(tmp_path / "view", "train")) == 2


def test_legacy_maps_official_test_and_has_no_generated_val(tmp_path: Path):
    config = load_config(make_fixture(tmp_path))
    rows = read_rows(config)
    assignments, exclusions = assign_generated_splits(rows, config, "legacy")
    records = build_records(rows, config, assignments, exclusions)
    assert {record.generated_split for record in records} == {"train", "test"}
    assert all(record.official_split == record.generated_split for record in records)


def test_invalid_protocol_is_rejected(tmp_path: Path):
    config = load_config(make_fixture(tmp_path))
    with pytest.raises(ValueError, match="protocol"):
        prepare(config, tmp_path / "view", "other")
