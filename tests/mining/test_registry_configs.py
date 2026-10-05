"""Mining configs come from the dataset registry; they equal the former configs/wildlife/*.json.

The JSON files were removed on 2026-10-05 (the registry is the single source). Their values are
pinned below as a literal, the way `tests/test_paths_registry.py` pins the removed launchers'
profiles, so a registry edit that would change a mining view fails here. Every other field had the
shared default in all of them (masked_images, identity, identity rule, 0.2, seed 0, ("", "unknown"),
train/test). The three turtle configs without a registry entry were dropped.
"""

from pathlib import Path

import pytest

from wildmatch.mining.wildlife_dataset import WildlifeConfig, config_from_registry

# registry key: (dataset_id, source_root, metadata_csv, split_column) of the former JSON (gmum profile)
FORMER_JSON = {
    "atrw": (
        "ATRW",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_ATRW.csv",
        "split",
    ),
    "beluga": (
        "BelugaID",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_no_background/metadata_BelugaID.csv",
        "split",
    ),
    "cowdataset": (
        "CowDataset",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_CowDataset.csv",
        "split",
    ),
    "giraffezebraid": (
        "GiraffeZebraID",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_GiraffeZebraID.csv",
        "split",
    ),
    "giraffes": (
        "Giraffes",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_Giraffes.csv",
        "split",
    ),
    "hyenaid2022": (
        "HyenaID2022",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_HyenaID2022.csv",
        "split",
    ),
    "leopardid2022": (
        "LeopardID2022",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_LeopardID2022.csv",
        "split",
    ),
    "nyala": (
        "NyalaData",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_no_background/metadata_NyalaData.csv",
        "split",
    ),
    "salamander": (
        "SalamanderID2025",
        "/shared/sets/datasets/vision/czechlynx/SalamanderID2025",
        "split_time_closed_no_background.csv",
        "split_train_test",
    ),
    "seastarreid2023": (
        "SeaStarReID2023",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_SeaStarReID2023.csv",
        "split",
    ),
    "stripespotter": (
        "StripeSpotter",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_mdsplit_no_background/metadata_StripeSpotter.csv",
        "split",
    ),
    "whaleshark": (
        "WhaleSharkID",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_no_background/metadata_WhaleSharkID.csv",
        "split",
    ),
    "zindi": (
        "ZindiTurtleRecall",
        "/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k",
        "metadata_no_background/metadata_ZindiTurtleRecall.csv",
        "split",
    ),
}


@pytest.mark.parametrize("key", sorted(FORMER_JSON))
def test_registry_config_equals_the_former_json(key):
    dataset_id, root, metadata, split_column = FORMER_JSON[key]
    expected = WildlifeConfig(
        dataset_id=dataset_id,
        metadata_csv=str(Path(root) / metadata),
        source_root=root,
        split_column=split_column,
    )
    assert config_from_registry(key, "gmum") == expected


def test_current_inputs_are_refused_where_they_differ_from_the_paper_inputs():
    with pytest.raises(ValueError, match="separate from the paper inputs"):
        config_from_registry("leopardid2022", "gmum", inputs="current")
    assert config_from_registry("salamander", "gmum", inputs="current") == config_from_registry("salamander", "gmum")


def test_czechlynx_entries_are_not_wildlife_configs():
    with pytest.raises(ValueError, match="CzechLynx"):
        config_from_registry("czechlynx_closed", "gmum")
