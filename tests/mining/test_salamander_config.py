"""SalamanderID2025: a non-WildlifeReID-10k dataset run through the wildlife pipeline.

Its metadata keeps the original database/query ``split`` column and adds
``split_train_test`` (database->train, query->test) because the preparer writes the
split value as the view folder name and mining/training expect train/ and test/.
The file-backed checks skip when the shared dataset is not mounted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import config_from_registry, resolve_image


def _config():
    return config_from_registry("salamander", "gmum")


def test_salamander_config_fields():
    config = _config()
    assert config.dataset_id == "SalamanderID2025"
    assert config.root == Path("/shared/sets/datasets/vision/czechlynx/SalamanderID2025")
    assert config.metadata_path == config.root / "split_time_closed_no_background.csv"
    assert config.image_prefix == "masked_images"
    assert config.split_column == "split_train_test"
    assert config.allowed_splits == ("train", "test")
    assert config.collection_rule == "identity"


def test_salamander_metadata_maps_database_query_to_train_test():
    config = _config()
    if not config.metadata_path.is_file():
        pytest.skip("SalamanderID2025 dataset is not mounted")
    with config.metadata_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["split"]: row["split_train_test"] for row in rows} == {"database": "train", "query": "test"}
    assert all(row["path"].startswith("masked_images/") for row in rows)
    for row in rows[:25]:
        assert resolve_image(config, row["path"]).is_file()
