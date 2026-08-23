import json
from pathlib import Path

from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader

from contrastive_finetuning.loading import (
    IndexAssignedTripletDataset,
    ShapeBucketBatchSampler,
)


class FakeFeatureCache:
    def __init__(self, shapes):
        self.shapes = shapes

    def image_hw(self, rel):
        return self.shapes[rel]


def test_shape_bucket_sampler_keeps_global_batches_shape_homogeneous(tmp_path: Path):
    entries = []
    shapes = {}
    for index in range(8):
        shape = (480, 512) if index % 2 else (512, 512)
        query = f"train/lynx_{index}/site/video/query.jpg"
        positive = f"train/lynx_{index}/site/video/positive.jpg"
        negative = f"train/lynx_{index + 10}/site/video/negative.jpg"
        entries.append({"query_frame": query, "positives": [positive], "negatives": [negative]})
        shapes[query] = shapes[positive] = shapes[negative] = shape

    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    dataset = IndexAssignedTripletDataset(
        index_path,
        root=tmp_path,
        feature_cache=FakeFeatureCache(shapes),
        return_meta=True,
    )
    sampler = ShapeBucketBatchSampler(
        dataset, per_gpu_batch_size=2, num_processes=2, seed=7
    )
    sampler.set_epoch(0)

    assert len(sampler) == 2
    def signature(index):
        query, positive, negatives, _ = dataset._planned_triplets[index]
        return (shapes[query], shapes[positive], (shapes[negatives[0]],))

    for batch in sampler:
        signatures = {signature(index) for index in batch}
        assert len(signatures) == 1

        # Both local rank slices come from this same global batch and therefore
        # have the same shape signature, which is the DDP invariant we need.
        first = batch[:2]
        second = batch[2:]
        assert len(first) == len(second) == 2
        assert {signature(i) for i in first} == {signature(i) for i in second}


def test_accelerate_splits_one_global_batch_per_rank(tmp_path: Path):
    entries = []
    shapes = {}
    for index in range(4):
        shape = (480, 512)
        query = f"train/lynx_{index}/site/video/query.jpg"
        positive = f"train/lynx_{index}/site/video/positive.jpg"
        negative = f"train/lynx_{index + 10}/site/video/negative.jpg"
        entries.append({"query_frame": query, "positives": [positive], "negatives": [negative]})
        shapes[query] = shapes[positive] = shapes[negative] = shape

    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    dataset = IndexAssignedTripletDataset(
        index_path, root=tmp_path, feature_cache=FakeFeatureCache(shapes)
    )
    sampler = ShapeBucketBatchSampler(dataset, 2, 2, seed=3)
    sampler.set_epoch(0)
    global_batch = next(iter(sampler))

    loader = DataLoader(list(range(4)), batch_sampler=sampler, num_workers=0)
    rank0 = prepare_data_loader(loader, num_processes=2, process_index=0, split_batches=True)
    rank1 = prepare_data_loader(loader, num_processes=2, process_index=1, split_batches=True)
    assert next(iter(rank0)).tolist() == global_batch[:2]
    assert next(iter(rank1)).tolist() == global_batch[2:]


def test_accelerate_splits_balanced_global_batch_to_eight_samples_per_rank():
    # This mirrors the four-GPU RDD contract: batch_size=8 is per GPU, so
    # Accelerate shards complete local batches across four ranks.
    loader = DataLoader(list(range(64)), batch_size=8, shuffle=False)
    rank0 = prepare_data_loader(loader, num_processes=4, process_index=0, split_batches=False)
    rank3 = prepare_data_loader(loader, num_processes=4, process_index=3, split_batches=False)

    assert next(iter(rank0)).tolist() == list(range(8))
    assert next(iter(rank3)).tolist() == list(range(24, 32))
    assert len(rank0) == 2
    assert len(loader) == 8


def test_balanced_global_batch_has_expected_steps_for_zindi_size():
    loader = DataLoader(list(range(8098)), batch_size=8, shuffle=False)
    prepared = prepare_data_loader(loader, num_processes=4, process_index=0, split_batches=False)
    assert len(prepared) == 254
