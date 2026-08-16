from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from contrastive_finetuning.loma_backend import (
    checkpoint_files,
    freeze_loma_backbone,
    train_pair_score,
)
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


def test_loma_resize_aligns_both_dimensions_to_dinov2_patch_size():
    images = torch.zeros(1, 3, 240, 960)
    resized = resize_long_side(images, 512)
    assert resized.shape[-1] == 504
    assert resized.shape[-2] % 14 == 0
    assert resized.shape[-1] % 14 == 0
