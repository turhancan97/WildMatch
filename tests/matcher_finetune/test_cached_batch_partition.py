import torch

from contrastive_finetuning.train_common import batch_features, run_lg_partitioned


class RecordingMatcher:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, payload):
        data0 = payload["image0"]
        data1 = payload["image1"]
        self.batch_sizes.append((data0["image_size"].shape[0], tuple(data0["image_size"].tolist()[0])))
        batch, m = data0["keypoints"].shape[:2]
        n = data1["keypoints"].shape[1]
        value = data0["image_size"][:, 1:2].float().expand(batch, m)
        return {
            "matching_scores0": value,
            "valid0": torch.ones((batch, m), dtype=torch.bool),
            "matches0": torch.full((batch, m), -1, dtype=torch.long),
            "assignment_scores": torch.zeros((batch, m + 1, n + 1)),
            "assignment_scores_nogate": torch.zeros((batch, m, n)),
            "matches": [torch.empty((0, 2), dtype=torch.long) for _ in range(batch)],
            "scores": [torch.empty(0) for _ in range(batch)],
            "stop": 1,
        }


def feature(n):
    return {
        "keypoints": torch.zeros((n, 2)),
        "descriptors": torch.zeros((n, 4)),
    }


def test_mixed_cached_pair_sizes_are_partitioned_and_restored():
    query = batch_features(
        [feature(2), feature(3), feature(1)],
        [480, 512, 480],
        [512, 512, 512],
    )
    candidate = batch_features(
        [feature(2), feature(3), feature(1)],
        [512, 512, 480],
        [512, 512, 512],
    )
    matcher = RecordingMatcher()

    stats = {}
    result = run_lg_partitioned(matcher, query, candidate, stats=stats)

    assert stats == {"calls": 1, "groups": 3, "pairs": 3, "partitioned_calls": 1}
    assert len(matcher.batch_sizes) == 3
    assert result["matching_scores0"][:, 0].tolist() == [480.0, 512.0, 480.0]
    assert result["valid0"].shape == (3, 3)


def test_mixed_cached_pair_sizes_use_one_forward_when_requested():
    query = batch_features(
        [feature(2), feature(3), feature(1)],
        [480, 512, 480],
        [512, 512, 512],
    )
    candidate = batch_features(
        [feature(2), feature(3), feature(1)],
        [512, 512, 480],
        [512, 512, 512],
    )
    matcher = RecordingMatcher()
    stats = {}

    run_lg_partitioned(matcher, query, candidate, stats=stats, partition=False)

    assert stats == {"calls": 1, "groups": 3, "pairs": 3, "partitioned_calls": 1}
    assert len(matcher.batch_sizes) == 1
