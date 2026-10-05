from pathlib import Path

from wildmatch.mining.czechlynx_mine import is_exact_query_frame


def test_train_mining_excludes_only_exact_query_frame():
    query_path = Path("train/lynx_1/foe/encounter/frame_0001.jpg")

    assert is_exact_query_frame(query_path, query_path, "train")
    assert not is_exact_query_frame(Path("train/lynx_1/foe/encounter/frame_0002.jpg"), query_path, "train")
    assert not is_exact_query_frame(query_path, query_path, "val")


def test_test_mining_does_not_apply_train_self_frame_rule():
    query_path = Path("test/lynx_1/foe/encounter/frame_0001.jpg")
    assert not is_exact_query_frame(query_path, query_path, "test")


def test_mining_workflow_submits_and_aggregates_test_split():
    spawner = Path(__file__).resolve().parents[2].joinpath("slurm/mining/spawn_czechlynx_mining.sh").read_text()
    aggregator = Path(__file__).resolve().parents[2].joinpath("slurm/mining/czechlynx_aggregate.sh").read_text()
    assert '"${dataset_root}/test"' in spawner
    assert '"${dump_report}" "${dataset_root}"' in spawner
    assert "CZECHLYNX_SPLIT_COLUMN" in spawner
    assert "CZECHLYNX_MINING_BACKEND" in spawner
    assert "outputs/${experiment_slug}/${protocol}/rdd/strong-matches" in spawner
    assert "outputs/${experiment_slug}/${protocol}/loma/strong-matches" in spawner
    assert "afterok:${train_job}:${val_job}:${test_job}" in spawner
    assert "--splits train val test" in aggregator
