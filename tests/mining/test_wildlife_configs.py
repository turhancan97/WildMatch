from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import load_config

NEW_DATASETS = {
    "AmvrakikosTurtles",
    "ATRW",
    "CowDataset",
    "Giraffes",
    "GiraffeZebraID",
    "HyenaID2022",
    "LeopardID2022",
    "ReunionTurtles",
    "SeaStarReID2023",
    "StripeSpotter",
    "ZakynthosTurtles",
}


@pytest.mark.parametrize("dataset_id", sorted(NEW_DATASETS))
def test_new_wildlife_config_matches_shared_metadata_contract(dataset_id: str):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "wildmatch"
        / "mining"
        / "configs"
        / "wildlife"
        / f"{dataset_id}.json"
    )
    raw = json.loads(config_path.read_text())
    config = load_config(config_path)

    assert config.dataset_id == dataset_id
    assert config.metadata_path == config.root / "metadata_mdsplit_no_background" / f"metadata_{dataset_id}.csv"
    assert config.root == Path("/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k")
    assert config.image_prefix == "masked_images"
    assert config.identity_column == "identity"
    assert config.split_column == "split"
    assert config.collection_rule == "identity"
    assert config.validation_fraction == pytest.approx(0.2)
    assert config.seed == 0
    assert config.excluded_identities == ("", "unknown")
    assert config.allowed_splits == ("train", "test")
    assert raw["metadata_csv"].startswith("metadata_mdsplit_no_background/")


def test_all_new_wildlife_configs_are_present():
    config_dir = Path(__file__).resolve().parents[2] / "src" / "wildmatch" / "mining" / "configs" / "wildlife"
    present = {path.stem for path in config_dir.glob("*.json")}
    assert NEW_DATASETS <= present
