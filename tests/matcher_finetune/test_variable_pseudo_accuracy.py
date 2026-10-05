from contrastive_finetuning.train_common import group_pseudo_accuracy_entries


def entry(n_pos, n_neg=5, index=0):
    return {
        "query_frame": f"train/lynx_0/collection/frame_{index}.jpg",
        "positives": [f"train/lynx_0/collection/pos_{i}.jpg" for i in range(n_pos)],
        "negatives": [f"train/lynx_{i + 1}/collection/neg.jpg" for i in range(n_neg)],
    }


def test_variable_positive_counts_are_bucketed_without_dropping_entries():
    entries = [entry(5, index=0), entry(1, index=1), entry(3, index=2), entry(5, index=3)]

    buckets = group_pseudo_accuracy_entries(entries)

    assert [(len(bucket[0]["positives"]), len(bucket[0]["negatives"]), len(bucket)) for bucket in buckets] == [
        (1, 5, 1),
        (3, 5, 1),
        (5, 5, 2),
    ]
    assert sum(len(bucket) for bucket in buckets) == len(entries)
