from pathlib import Path

from scripts.czechlynx_mine import is_exact_query_frame
from scripts.czechlynx_evaluate import diverse_topk_columns, select_topk_query_pool


def test_train_mining_excludes_only_exact_query_frame():
    query_path = Path("train/lynx_1/foe/encounter/frame_0001.jpg")

    assert is_exact_query_frame(query_path, query_path, "train")
    assert not is_exact_query_frame(
        Path("train/lynx_1/foe/encounter/frame_0002.jpg"), query_path, "train"
    )
    assert not is_exact_query_frame(query_path, query_path, "val")


def test_test_mining_does_not_apply_train_self_frame_rule():
    query_path = Path("test/lynx_1/foe/encounter/frame_0001.jpg")
    assert not is_exact_query_frame(query_path, query_path, "test")


def test_top15_pool_is_exactly_diverse_gallery_frames():
    names = ["a", "a", "b", "b", "c", "c"]
    row = [0.95, 0.10, 0.90, 0.20, 0.85, 0.30]

    selected = diverse_topk_columns(row, list(range(len(row))), names, 4)

    assert selected == [0, 2, 4, 1]
    assert len(selected) == 4
    assert len({names[index] for index in selected}) == 3


def test_top15_uses_strongest_query_frame():
    import torch

    class Collection:
        def __init__(self, name):
            self.name = name

    scores = torch.tensor([[0.1, 0.2, 0.3], [0.8, 0.1, 0.2]])
    query_index, selected = select_topk_query_pool(
        scores, [Collection("a"), Collection("b"), Collection("c")], 2
    )

    assert query_index == 1
    assert len(selected) == 2


def test_mining_workflow_submits_and_aggregates_test_split():
    spawner = Path("slurm_scripts/spawn_czechlynx_mining.sh").read_text()
    aggregator = Path("slurm_scripts/czechlynx_aggregate.sh").read_text()
    assert '"${dataset_root}/test"' in spawner
    assert '"${dump_report}" test)' in spawner
    assert 'afterok:${train_job}:${val_job}:${test_job}' in spawner
    assert "--splits train val test" in aggregator
