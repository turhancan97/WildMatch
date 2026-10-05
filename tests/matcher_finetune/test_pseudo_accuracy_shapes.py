import torch

from wildmatch.matcher_finetune.loading import (
    PseudoAccuracyDataset,
    collate_pseudo_accuracy_images,
    get_loader,
)


def _fake_loader(path):
    shapes = {
        "query.jpg": (3, 480, 512),
        "positive.jpg": (3, 512, 512),
        "negative.jpg": (3, 714, 714),
    }
    return torch.zeros(shapes[path.name])


def test_live_pseudo_accuracy_loader_keeps_mixed_candidate_shapes_ragged(tmp_path):
    entries = [
        {
            "query_frame": "query.jpg",
            "positives": ["positive.jpg"],
            "negatives": ["negative.jpg"],
        }
    ]
    dataset = PseudoAccuracyDataset(entries, root=tmp_path, loader=_fake_loader)
    loader = get_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        collate_fn=collate_pseudo_accuracy_images,
    )

    batch = next(iter(loader))
    assert len(batch) == 1
    query, candidates, index = batch[0]
    assert query.shape == (3, 480, 512)
    assert [tuple(candidate.shape) for candidate in candidates] == [
        (3, 512, 512),
        (3, 714, 714),
    ]
    assert index == 0
