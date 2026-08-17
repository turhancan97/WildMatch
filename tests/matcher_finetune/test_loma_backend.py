from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from contrastive_finetuning.loma_backend import (
    checkpoint_files,
    freeze_loma_backbone,
    train_pair_score,
)
from contrastive_finetuning.loma_cache import LomaFeatureCache
import contrastive_finetuning.train_loma_matches as loma_train
from contrastive_finetuning.train_loma_matches import resize_long_side


class FakeLoMa(nn.Module):
    def __init__(self):
        super().__init__()
        self.matcher_weight = nn.Parameter(torch.tensor(0.0))
        self._detector = nn.Linear(2, 2)
        self._descriptor = nn.Linear(2, 2)

    def forward(self, keypoints0, keypoints1, descriptors0, descriptors1):
        shape = (keypoints0.shape[0], keypoints0.shape[1] + 1, keypoints1.shape[1] + 1)
        return {"scores": self.matcher_weight * torch.ones(shape, device=keypoints0.device)}


def test_freeze_loma_backbone_and_trainable_score():
    model = FakeLoMa()
    freeze_loma_backbone(model)
    assert all(not parameter.requires_grad for parameter in model._detector.parameters())
    assert all(not parameter.requires_grad for parameter in model._descriptor.parameters())

    score = train_pair_score(
        model,
        torch.zeros(2, 4, 2), torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 2), torch.zeros(2, 4, 8),
    ).sum()
    score.backward()
    assert model.matcher_weight.grad is not None


def test_checkpoint_bundle_resolution(tmp_path: Path):
    bundle = tmp_path / "latest"
    bundle.mkdir()
    save_file({"transformers.0.weight": torch.ones(2, 2)}, str(bundle / "model.safetensors"))
    (bundle / "metadata.json").write_text('{"base_weights": null}')
    model_path, base_path, metadata = checkpoint_files(bundle)
    assert model_path.name == "model.safetensors"
    assert base_path is None
    assert metadata["base_weights"] is None


def test_loma_checkpoint_restores_scheduler_state(tmp_path: Path):
    args = SimpleNamespace(
        loma_variant="loma-b",
        loma_weights=None,
        train_index=tmp_path / "train.json",
        val_index=tmp_path / "val.json",
        resize=512,
        num_keypoints=512,
    )
    model = FakeLoMa()
    freeze_loma_backbone(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    optimizer.zero_grad()
    model.matcher_weight.backward()
    optimizer.step()
    scheduler.step()
    expected_last_epoch = scheduler.last_epoch

    checkpoint = tmp_path / "epoch_002"
    loma_train.save_checkpoint(model, optimizer, scheduler, 2, 17, args, checkpoint)

    restored_model = FakeLoMa()
    freeze_loma_backbone(restored_model)
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored_model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=10)
    epoch, step = loma_train.restore_checkpoint(
        restored_model, restored_optimizer, restored_scheduler, checkpoint
    )

    assert (epoch, step) == (2, 17)
    assert restored_scheduler.last_epoch == expected_last_epoch
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_loma_resize_aligns_both_dimensions_to_dinov2_patch_size():
    images = torch.zeros(1, 3, 240, 960)
    resized = resize_long_side(images, 512)
    assert resized.shape[-1] == 504
    assert resized.shape[-2] % 14 == 0
    assert resized.shape[-1] % 14 == 0


def test_loma_mini_eval_matches_rdd_protocol(monkeypatch, tmp_path: Path):
    model = FakeLoMa()
    extract_batch_sizes = []

    monkeypatch.setattr(
        loma_train,
        "load_image",
        lambda path, resize: torch.zeros(3, 14, 14),
    )

    def fake_extract(model, images, num_keypoints):
        extract_batch_sizes.append(images.shape[0])
        return torch.zeros(images.shape[0], 4, 2), torch.zeros(images.shape[0], 4, 8)

    monkeypatch.setattr(loma_train, "extract_loma_features", fake_extract)
    eval_call = {"count": 0}

    def fake_eval(model, keypoints0, descriptors0, keypoints1, descriptors1):
        eval_call["count"] += 1
        match_count = 2 if eval_call["count"] % 2 else 1
        return (
            torch.zeros(keypoints0.shape[0]),
            torch.full((keypoints0.shape[0],), match_count),
        )

    monkeypatch.setattr(
        loma_train,
        "eval_pair_scores",
        fake_eval,
    )

    entries = [
        {
            "query_frame": f"query/{index}.jpg",
            "positives": [f"positive/{index}.jpg"],
            "negatives": [f"negative/{index}.jpg"],
        }
        for index in range(3)
    ]
    metrics = loma_train.evaluate_mini_index(
        model, entries, tmp_path, torch.device("cpu"), 512, 512, batch_size=2
    )

    assert metrics == {"mean_matches_pos": 2.0, "mean_matches_neg": 1.0}
    assert extract_batch_sizes == [2, 2, 2, 1, 1, 1]


def test_loma_cache_round_trip_and_metadata_validation(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    frame = cache_dir / "train" / "lynx_a" / "site" / "seq" / "frame_0000.npz"
    frame.parent.mkdir(parents=True)
    np_keypoints = torch.arange(8, dtype=torch.float32).reshape(4, 2).numpy()
    np_descriptors = torch.ones(4, 8, dtype=torch.float32).numpy()
    np.savez_compressed(
        frame,
        keypoints=np_keypoints,
        descriptors=np_descriptors,
        scores=np.ones(4, dtype=np.float32),
        image_size=np.asarray([504, 504], dtype=np.int32),
    )
    (cache_dir / "manifest.json").write_text(
        '{"format":"lynx-loma-cache-v1","backend":"loma",'
        '"variant":"loma-b","resize":512,"num_keypoints":4,"patch_size":14}'
    )

    cache = LomaFeatureCache(
        cache_dir, variant="loma-b", resize=512, num_keypoints=4
    )
    keypoints, descriptors = cache.load(
        "train/lynx_a/site/seq/frame_0000.jpg", torch.device("cpu")
    )
    assert keypoints.shape == (1, 4, 2)
    assert descriptors.shape == (1, 4, 8)

    with pytest.raises(ValueError, match="variant"):
        LomaFeatureCache(cache_dir, variant="loma-l", resize=512, num_keypoints=4)


def test_cached_triplet_dataset_returns_batched_feature_tensors(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    manifest = {
        "format": "lynx-loma-cache-v1",
        "backend": "loma",
        "variant": "loma-b",
        "resize": 512,
        "num_keypoints": 4,
        "patch_size": 14,
    }
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    entries = [{
        "query_frame": "train/lynx_a/video/query.jpg",
        "positives": ["train/lynx_a/video/positive.jpg"],
        "negatives": ["train/lynx_b/video/negative.jpg"],
    }]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(entries))
    for relative in (
        entries[0]["query_frame"],
        entries[0]["positives"][0],
        entries[0]["negatives"][0],
    ):
        output = cache_dir / Path(relative).with_suffix(".npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            keypoints=np.zeros((4, 2), dtype=np.float32),
            descriptors=np.zeros((4, 8), dtype=np.float32),
            scores=np.ones(4, dtype=np.float32),
            image_size=np.asarray([504, 504], dtype=np.int32),
        )

    cache = LomaFeatureCache(cache_dir, variant="loma-b", resize=512, num_keypoints=4)
    dataset = loma_train.IndexAssignedTripletDataset(
        index_path, root=tmp_path / "frames", transform=None, feature_cache=cache
    )
    query, positive, negative = dataset[0]
    assert query["keypoints"].shape == (4, 2)
    assert positive["descriptors"].shape == (4, 8)
    assert negative["image_size"].tolist() == [504, 504]


def test_distributed_index_eval_gathers_one_dict_per_rank(monkeypatch, tmp_path: Path):
    model = FakeLoMa()

    monkeypatch.setattr(
        loma_train,
        "load_features_for_paths",
        lambda model, paths, device, resize, num_keypoints: (
            torch.zeros(len(paths), 4, 2), torch.zeros(len(paths), 4, 8)
        ),
    )
    monkeypatch.setattr(
        loma_train,
        "eval_pair_scores",
        lambda model, keypoints0, descriptors0, keypoints1, descriptors1: (
            torch.ones(keypoints0.shape[0]), torch.ones(keypoints0.shape[0])
        ),
    )

    gathered_payloads = []

    class FakeAccelerator:
        process_index = 0
        num_processes = 1
        is_main_process = True

        @staticmethod
        def unwrap_model(model):
            return model

    def fake_gather(payload):
        gathered_payloads.append(payload)
        return payload

    monkeypatch.setattr(loma_train, "gather_object", fake_gather)
    entries = [{
        "query_frame": "test/lynx_a/video/query.jpg",
        "positives": ["train/lynx_a/video/positive.jpg"],
        "negatives": ["train/lynx_b/video/negative.jpg"],
    }]
    metrics = loma_train.evaluate_index(
        model, entries, tmp_path, torch.device("cpu"), 512, 4,
        accelerator=FakeAccelerator(),
    )

    assert isinstance(gathered_payloads[0], list)
    assert isinstance(gathered_payloads[0][0], dict)
    assert metrics["entries"] == 1.0
