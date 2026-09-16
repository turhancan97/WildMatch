from pathlib import Path

from scripts.czechlynx_dataset import (
    assign_splits,
    build_records,
    masked_relative_path,
    validate_records,
)


def row(source, identity, encounter, number, split, *, open_split=None):
    path = f"CzechLynx/{source}/{identity}/{number:05d}_{identity}.jpg"
    return {
        "source": source,
        "unique_name": identity,
        "encounter": str(encounter),
        "path": path,
        "masked_path": path.replace("CzechLynx/", "CzechLynx_masked/", 1),
        "split-time_closed": split,
        "split-time_open": open_split or split,
    }


def test_masked_path_mapping():
    assert masked_relative_path("CzechLynx/foe/lynx_1/a.jpg") == "CzechLynx_masked/foe/lynx_1/a.jpg"


def test_mixed_encounter_is_test_and_validation_is_atomic():
    rows = [
        row("foe", "lynx_1", 1, 0, "train"),
        row("foe", "lynx_1", 1, 1, "test"),
        row("foe", "lynx_1", 2, 0, "train"),
        row("foe", "lynx_2", 3, 0, "train"),
        row("foe", "lynx_2", 4, 0, "train"),
        row("foe", "lynx_3", 5, 0, "test"),
    ]
    assignments, mixed = assign_splits(rows, validation_fraction=0.5, seed=0)
    assert ("foe", "1") in mixed
    assert assignments[("foe", "1")] == "test"
    assert assignments[("foe", "5")] == "test"
    assert sum(value == "val" for value in assignments.values()) == 2

    records = build_records(rows, assignments)
    assert len(records) == len(rows)
    assert len({record.relative_path for record in records}) == len(rows)
    train_encounters = {record.encounter_key for record in records if record.split == "train"}
    val_encounters = {record.encounter_key for record in records if record.split == "val"}
    assert not train_encounters & val_encounters


def test_validate_records_checks_symlinks(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "view"
    target = source_root / "CzechLynx_masked/foe/lynx_1/00000_lynx_1.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"jpg")
    rows = [row("foe", "lynx_1", 1, 0, "train")]
    records = build_records(rows, {("foe", "1"): "train"})
    link = output_root / records[0].relative_path
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    summary = validate_records(records, source_root, output_root)
    assert summary["frames"] == 1
    assert summary["frames_by_split"] == {"train": 1}


def test_time_open_split_column_is_supported():
    rows = [
        row("foe", "lynx_1", 1, 0, "test", open_split="train"),
        row("foe", "lynx_2", 2, 0, "train", open_split="test"),
    ]
    assignments, mixed = assign_splits(rows, split_column="split-time_open", validation_fraction=0.0)
    assert not mixed
    assert assignments[("foe", "1")] == "train"
    assert assignments[("foe", "2")] == "test"
    records = build_records(rows, assignments, split_column="split-time_open")
    assert {record.metadata_split for record in records} == {"train", "test"}


def test_invalid_split_column_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="split_column"):
        assign_splits([], split_column="split-unknown")
