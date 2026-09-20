from pathlib import Path
import json
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from contrastive_finetuning.loma_backend import (
    checkpoint_files,
    configure_loma_trainable_component,
    describe_keypoints_with_grad,
    freeze_loma_backbone,
    LoMaDescriptorTrainingModel,
    matcher_scores_with_descriptor_grad,
    set_loma_train_mode,
    train_pair_score,
)
from contrastive_finetuning.loma_cache import LomaFeatureCache
from contrastive_finetuning.loma_keypoint_cache import LomaKeypointCache, module_fingerprint
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


class FakePositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 4, bias=False)

    def forward(self, keypoints):
        return self.projection(keypoints)


class FakePairTransformer(nn.Module):
    def forward(self, desc0, desc1, encoding0, encoding1):
        return desc0 + encoding0, desc1 + encoding1


class FakeAssignment(nn.Module):
    def forward(self, desc0, desc1):
        scores = torch.matmul(desc0, desc1.transpose(-1, -2)) / 4
        scores = torch.nn.functional.pad(scores, (0, 1, 0, 1), value=-4.0)
        return scores, None


class FakeDescriptorLoMa(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=1, mp=False, filter_threshold=0.2)
        self.input_proj = nn.Linear(3, 4, bias=False)
        self.posenc = FakePositionEncoding()
        self.transformers = nn.ModuleList([FakePairTransformer()])
        self.log_assignment = nn.ModuleList([FakeAssignment()])
        self._detector = nn.Linear(2, 2)
        self._descriptor = nn.Conv2d(3, 3, kernel_size=1)


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


def test_descriptor_mode_only_unfreezes_dedode_and_preserves_eval_modes():
    model = FakeDescriptorLoMa()
    configure_loma_trainable_component(model, "descriptor")
    set_loma_train_mode(model, True, "descriptor")

    assert all(parameter.requires_grad for parameter in model._descriptor.parameters())
    assert all(not parameter.requires_grad for parameter in model._detector.parameters())
    assert all(not parameter.requires_grad for parameter in model.input_proj.parameters())
    assert not model._detector.training
    assert model._descriptor.training
    assert not model.input_proj.training


def test_descriptor_and_frozen_matcher_forward_propagates_gradient_to_descriptors():
    model = FakeDescriptorLoMa()
    configure_loma_trainable_component(model, "descriptor")
    desc0 = torch.randn(2, 5, 3, requires_grad=True)
    desc1 = torch.randn(2, 5, 3, requires_grad=True)
    keypoints = torch.rand(2, 5, 2) * 2 - 1

    scores = matcher_scores_with_descriptor_grad(
        model, keypoints, keypoints, desc0, desc1
    )
    scores[:, :-1, :-1].exp().sum().backward()

    assert desc0.grad is not None and desc0.grad.abs().sum() > 0
    assert desc1.grad is not None and desc1.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.input_proj.parameters())


def test_descriptor_features_support_mixed_shapes_and_backpropagate():
    model = FakeDescriptorLoMa()
    configure_loma_trainable_component(model, "descriptor")
    images = [torch.rand(3, 14, 28), torch.rand(3, 28, 14)]
    keypoints = torch.zeros(2, 4, 2)

    descriptors = describe_keypoints_with_grad(model, images, keypoints)
    descriptors.square().mean().backward()

    assert descriptors.shape == (2, 4, 3)
    assert model._descriptor.weight.grad is not None
    assert model._descriptor.weight.grad.abs().sum() > 0


def test_descriptor_training_wrapper_updates_only_descriptor(monkeypatch):
    # The match-count diagnostics are not part of the loss; stub LoMa's
    # optional correspondence filter so this test does not need its package.
    package = ModuleType("loma")
    package.__path__ = []
    loma_module = ModuleType("loma.loma")

    def fake_filter_matches(scores, threshold):
        batch, rows, columns = scores.shape
        counts = torch.zeros(batch, rows - 1, device=scores.device)
        matches = torch.full((batch, rows - 1), -1, device=scores.device, dtype=torch.long)
        return matches, matches.clone(), counts, counts.clone()

    loma_module.filter_matches = fake_filter_matches
    monkeypatch.setitem(sys.modules, "loma", package)
    monkeypatch.setitem(sys.modules, "loma.loma", loma_module)

    model = FakeDescriptorLoMa()
    configure_loma_trainable_component(model, "descriptor")
    wrapper = LoMaDescriptorTrainingModel(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    detector_before = [parameter.detach().clone() for parameter in model._detector.parameters()]
    matcher_before = [parameter.detach().clone() for parameter in model.input_proj.parameters()]
    descriptor_before = [parameter.detach().clone() for parameter in model._descriptor.parameters()]

    query = [torch.rand(3, 14, 14), torch.rand(3, 14, 14)]
    positive = [torch.rand(3, 14, 14), torch.rand(3, 14, 14)]
    negative = [torch.rand(3, 14, 14), torch.rand(3, 14, 14)]
    keypoints = torch.rand(2, 5, 2) * 2 - 1
    positive_score, _, negative_score, _ = wrapper(
        query, positive, negative, keypoints, keypoints, keypoints
    )
    loss = torch.relu(0.5 - positive_score + negative_score).mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in model._detector.parameters())
    assert all(parameter.grad is None for parameter in model.input_proj.parameters())
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model._descriptor.parameters()
    )
    optimizer.step()
    assert all(torch.equal(before, after) for before, after in zip(detector_before, model._detector.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(matcher_before, model.input_proj.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(descriptor_before, model._descriptor.parameters()))


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
        loma_train_component="matcher",
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


def test_descriptor_checkpoint_contains_only_descriptor_weights(tmp_path: Path):
    model = FakeDescriptorLoMa()
    configure_loma_trainable_component(model, "descriptor")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    args = SimpleNamespace(
        loma_variant="loma-b", loma_weights=Path("pretrained.pt"),
        loma_train_component="descriptor", split_protocol="strict",
        train_index=Path("train.json"), val_index=Path("val.json"),
        resize=512, num_keypoints=512,
    )
    output = tmp_path / "descriptor-checkpoint"
    loma_train.save_checkpoint(model, optimizer, scheduler, 0, 1, args, output)

    from safetensors.torch import load_file

    state = load_file(str(output / "model.safetensors"))
    metadata = json.loads((output / "metadata.json").read_text())
    assert state and all(name.startswith("_descriptor.") for name in state)
    assert metadata["train_component"] == "descriptor"
    assert metadata["format"] == "lynx-loma-descriptor-v1"

    restored = FakeDescriptorLoMa()
    configure_loma_trainable_component(restored, "descriptor")
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.parameters() if parameter.requires_grad], lr=1e-3
    )
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=2
    )
    epoch, step = loma_train.restore_checkpoint(
        restored, restored_optimizer, restored_scheduler, output, "descriptor"
    )
    assert (epoch, step) == (0, 1)
    for name, parameter in restored._descriptor.named_parameters():
        assert torch.equal(parameter, dict(model._descriptor.named_parameters())[name])
    with pytest.raises(ValueError, match="cannot resume"):
        loma_train.restore_checkpoint(
            restored, restored_optimizer, restored_scheduler, output, "matcher"
        )


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


def test_loma_keypoint_cache_round_trip_and_detector_validation(tmp_path: Path):
    cache_dir = tmp_path / "keypoints"
    frame = cache_dir / "train" / "lynx_a" / "frame_0000.npz"
    frame.parent.mkdir(parents=True)
    np.savez_compressed(
        frame,
        keypoints=np.ones((4, 2), dtype=np.float32),
        image_size=np.asarray([504, 504], dtype=np.int32),
    )
    detector_hash = "a" * 64
    (cache_dir / "manifest.json").write_text(json.dumps({
        "format": "lynx-loma-keypoints-v1", "backend": "loma",
        "variant": "loma-b", "resize": 512, "num_keypoints": 4,
        "patch_size": 14, "detector_sha256": detector_hash, "complete": True,
    }))

    cache = LomaKeypointCache(
        cache_dir, variant="loma-b", resize=512, num_keypoints=4,
        detector_sha256=detector_hash,
    )
    keypoints, image_size = cache.load("train/lynx_a/frame_0000.jpg", torch.device("cpu"))
    assert keypoints.shape == (1, 4, 2)
    assert image_size.tolist() == [504, 504]
    with pytest.raises(ValueError, match="detector_sha256"):
        LomaKeypointCache(
            cache_dir, variant="loma-b", resize=512, num_keypoints=4,
            detector_sha256="b" * 64,
        )
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    manifest["complete"] = False
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        LomaKeypointCache(
            cache_dir, variant="loma-b", resize=512, num_keypoints=4,
            detector_sha256=detector_hash,
        )


def test_module_fingerprint_is_stable():
    model = nn.Linear(3, 2)
    assert module_fingerprint(model) == module_fingerprint(model)


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
        lambda model, paths, device, resize, num_keypoints, **kwargs: (
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
