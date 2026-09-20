"""LoMa loading, feature extraction, and scoring helpers.

The LoMa project exposes a stable inference API through ``loma.loma``.  This
module deliberately imports it lazily: the existing RDD/LightGlue workflow
must remain usable in environments where LoMa is not installed.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


VARIANT_CONFIGS = {
    "loma-b": "LoMaB",
    "loma-b128": "LoMaB128",
    "loma-l": "LoMaL",
    "loma-g": "LoMaG",
    "loma-r": "LoMaR",
}


def _load_raw_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "weights", "matcher"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise TypeError(f"{path} does not contain a PyTorch state dictionary")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}


def checkpoint_files(path: Path) -> tuple[Path, Path | None, dict[str, Any]]:
    """Return (matcher weights, optional base weights, metadata)."""
    if path.is_file():
        return path, None, {}
    if not path.is_dir():
        raise FileNotFoundError(path)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    matcher = next(
        (path / name for name in ("model.safetensors", "matcher.safetensors", "weights.pth") if (path / name).exists()),
        None,
    )
    if matcher is None:
        raise FileNotFoundError(f"No LoMa model file found in {path}")
    base = metadata.get("base_weights")
    base_path = Path(base).expanduser() if base else None
    if base_path is not None and not base_path.is_absolute():
        base_path = path / base_path
    if base_path is not None and not base_path.exists():
        base_path = None
    return matcher, base_path, metadata


def _config_without_downloads(config: Any) -> Any:
    names = {field.name for field in fields(config)}
    updates: dict[str, Any] = {}
    if "weights_url" in names:
        updates["weights_url"] = None
    if "compile" in names:
        updates["compile"] = False
    return replace(config, **updates) if updates else config


def build_loma(
    variant: str = "loma-b",
    weights: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Build a LoMa model and optionally overlay a local checkpoint bundle."""
    try:
        from loma.loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, LoMaR
    except ImportError as exc:
        raise ImportError(
            "LoMa is not installed. Install requirements-loma.txt in the loma environment."
        ) from exc

    variant = variant.lower()
    try:
        config_cls = {name.lower(): value for name, value in {
            "loma-b": LoMaB,
            "loma-b128": LoMaB128,
            "loma-l": LoMaL,
            "loma-g": LoMaG,
            "loma-r": LoMaR,
        }.items()}[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported LoMa variant {variant!r}; expected {sorted(VARIANT_CONFIGS)}") from exc

    matcher_path: Path | None = None
    base_path: Path | None = None
    metadata: dict[str, Any] = {}
    if weights is not None:
        matcher_path, base_path, metadata = checkpoint_files(Path(weights).expanduser())

    config = config_cls()
    if weights is not None:
        # A local checkpoint replaces the matcher weights. The detector and
        # descriptor are still initialized by LoMa's own pretrained loaders.
        config = _config_without_downloads(config)
    else:
        # Keep LoMaB's official matcher URL when no local checkpoint was
        # supplied, but disable compilation for portable first runs.
        names = {field.name for field in fields(config)}
        config = replace(config, compile=False) if "compile" in names else config

    model = LoMa(config)
    model.to(device)

    def overlay(path: Path) -> None:
        state = _load_raw_state(path)
        model_state = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_state}
        if not matched:
            raise RuntimeError(f"{path} contains no keys compatible with {variant}")
        model.load_state_dict(matched, strict=False)

    if base_path is not None:
        overlay(base_path)
    if matcher_path is not None:
        overlay(matcher_path)

    freeze_loma_backbone(model)
    return model


def configure_loma_trainable_component(model: nn.Module, component: str) -> None:
    """Enable gradients for only the requested LoMa component."""
    if component not in {"matcher", "descriptor"}:
        raise ValueError(f"unsupported LoMa training component {component!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if component == "descriptor":
        module = getattr(model, "_descriptor", None)
        if module is None:
            raise AttributeError("LoMa model has no _descriptor module")
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    else:
        for name, parameter in model.named_parameters():
            if not name.startswith(("_detector.", "_descriptor.")):
                parameter.requires_grad_(True)


def freeze_loma_backbone(model: nn.Module) -> None:
    """Compatibility helper: freeze DaD and DeDoDe for matcher-only runs."""
    configure_loma_trainable_component(model, "matcher")
    model._detector.eval()
    model._descriptor.eval()


def set_loma_train_mode(
    model: nn.Module, training: bool, component: str = "matcher"
) -> None:
    model.train(training)
    model._detector.eval()
    if component == "descriptor":
        model._descriptor.train(training)
        # The matcher is fixed in descriptor mode. Keeping it in eval mode
        # prevents any stateful behavior while retaining gradients to inputs.
        for name, module in model.named_children():
            if name not in {"_detector", "_descriptor"}:
                module.eval()
    else:
        model._descriptor.eval()


@torch.inference_mode()
def extract_loma_features(model: nn.Module, images: torch.Tensor, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
    keypoints, descriptors, _, _ = model.detect_and_describe(images, num_keypoints=num_keypoints)
    return keypoints, descriptors


def _describe_image_group_with_grad(
    descriptor: nn.Module, images: torch.Tensor, keypoints: torch.Tensor
) -> torch.Tensor:
    """Run DeDoDe directly, bypassing its inference-mode public wrapper."""
    dense = descriptor(images)
    return F.grid_sample(
        dense.float(), keypoints[:, None].to(dense.device),
        mode="bilinear", align_corners=False,
    )[:, :, 0].mT


def describe_keypoints_with_grad(
    model: nn.Module,
    images: list[torch.Tensor],
    keypoints: torch.Tensor,
) -> torch.Tensor:
    """Describe fixed normalized keypoints, grouping images by spatial shape."""
    if len(images) != keypoints.shape[0]:
        raise ValueError("image and keypoint batch sizes differ")
    groups: dict[tuple[int, int], list[int]] = {}
    for index, image in enumerate(images):
        groups.setdefault(tuple(image.shape[-2:]), []).append(index)
    output: list[torch.Tensor | None] = [None] * len(images)
    device = keypoints.device
    for indices in groups.values():
        image_batch = torch.stack([images[index] for index in indices]).to(
            device, non_blocking=True
        )
        kp_batch = keypoints[indices]
        described = _describe_image_group_with_grad(
            model._descriptor, image_batch, kp_batch
        )
        for local_index, original_index in enumerate(indices):
            output[original_index] = described[local_index:local_index + 1]
    if any(item is None for item in output):
        raise RuntimeError("failed to describe one or more images")
    return torch.cat([item for item in output if item is not None], dim=0)


@torch.no_grad()
def detect_loma_keypoints(
    model: nn.Module, images: list[torch.Tensor], num_keypoints: int
) -> torch.Tensor:
    """Detect fixed DaD keypoints for a possibly mixed-shape image list."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, image in enumerate(images):
        groups.setdefault(tuple(image.shape[-2:]), []).append(index)
    output: list[torch.Tensor | None] = [None] * len(images)
    device = next(model.parameters()).device
    for indices in groups.values():
        image_batch = torch.stack([images[index] for index in indices]).to(
            device, non_blocking=True
        )
        # DaD.detect() moves input to loma.device.device (usually cuda:0),
        # which is wrong for DDP ranks assigned another local GPU. Its forward
        # method does not perform that implicit move, so call it with the
        # already rank-local image batch. The outer no_grad context keeps the
        # frozen detector out of autograd.
        detected = model._detector(
            image_batch, num_keypoints=num_keypoints
        )["keypoints"].clone()
        for local_index, original_index in enumerate(indices):
            output[original_index] = detected[local_index:local_index + 1]
    if any(item is None for item in output):
        raise RuntimeError("failed to detect keypoints for one or more images")
    return torch.cat([item for item in output if item is not None], dim=0)


def matcher_scores_with_descriptor_grad(
    model: nn.Module,
    keypoints0: torch.Tensor,
    keypoints1: torch.Tensor,
    descriptors0: torch.Tensor,
    descriptors1: torch.Tensor,
) -> torch.Tensor:
    """LoMa matcher forward that preserves gradients to descriptor inputs.

    LoMa's upstream ``forward`` detaches descriptors because it is designed
    for fixed descriptors. This mirrors that forward while intentionally
    omitting those two detach calls for descriptor fine-tuning.
    """
    try:
        from loma.device import amp_dtype
    except ImportError:
        amp_dtype = torch.float16

    device = descriptors0.device
    with torch.autocast(
        enabled=bool(getattr(model.cfg, "mp", False)),
        dtype=amp_dtype,
        device_type=device.type,
    ):
        keypoints0 = keypoints0.to(device)
        keypoints1 = keypoints1.to(device)
        descriptors0 = descriptors0.to(device).contiguous()
        descriptors1 = descriptors1.to(device).contiguous()
        desc0 = model.input_proj(descriptors0)
        desc1 = model.input_proj(descriptors1)
        encoding0 = model.posenc(keypoints0)
        encoding1 = model.posenc(keypoints1)
        for index in range(model.cfg.n_layers):
            desc0, desc1 = model.transformers[index](
                desc0, desc1, encoding0, encoding1
            )
        scores, _ = model.log_assignment[index](desc0, desc1)
    return scores


def train_pair_score(model: nn.Module, keypoints0: torch.Tensor, descriptors0: torch.Tensor,
                     keypoints1: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
    """Differentiable confidence used by the lynx positive/negative loss."""
    scores = model(keypoints0, keypoints1, descriptors0, descriptors1)["scores"]
    # In training mode LoMa returns a log assignment matrix with a dustbin.
    pair_scores = scores[:, :-1, :-1].exp()
    row_best = pair_scores.max(dim=-1).values.mean(dim=-1)
    col_best = pair_scores.max(dim=-2).values.mean(dim=-1)
    return 0.5 * (row_best + col_best)


def train_pair_score_with_matches(
    model: nn.Module,
    keypoints0: torch.Tensor,
    descriptors0: torch.Tensor,
    keypoints1: torch.Tensor,
    descriptors1: torch.Tensor,
    *,
    preserve_descriptor_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the differentiable score and detached mutual-match counts.

    The score is the same objective as :func:`train_pair_score`.  The counts
    are only diagnostics, matching the ``matches/mean_*`` metrics logged by
    the RDD/LightGlue trainer; they do not participate in backpropagation.
    """
    from loma.loma import filter_matches

    if preserve_descriptor_grad:
        scores = matcher_scores_with_descriptor_grad(
            model, keypoints0, keypoints1, descriptors0, descriptors1
        )
    else:
        scores = model(keypoints0, keypoints1, descriptors0, descriptors1)["scores"]
    pair_scores = scores[:, :-1, :-1].exp()
    row_best = pair_scores.max(dim=-1).values.mean(dim=-1)
    col_best = pair_scores.max(dim=-2).values.mean(dim=-1)
    pair_score = 0.5 * (row_best + col_best)
    base_model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        _, _, matching_scores0, _ = filter_matches(scores.detach(), base_model.cfg.filter_threshold)
        match_counts = (matching_scores0 > 0).sum(dim=-1).float()
    return pair_score, match_counts


class LoMaDescriptorTrainingModel(nn.Module):
    """DDP-visible wrapper that trains DeDoDe through LoMa's frozen matcher."""

    def __init__(self, loma_model: nn.Module) -> None:
        super().__init__()
        self.loma = loma_model

    def forward(
        self,
        query_images: list[torch.Tensor],
        positive_images: list[torch.Tensor],
        negative_images: list[torch.Tensor],
        query_keypoints: torch.Tensor,
        positive_keypoints: torch.Tensor,
        negative_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_descriptors = describe_keypoints_with_grad(
            self.loma, query_images, query_keypoints
        )
        positive_descriptors = describe_keypoints_with_grad(
            self.loma, positive_images, positive_keypoints
        )
        negative_descriptors = describe_keypoints_with_grad(
            self.loma, negative_images, negative_keypoints
        )
        positive_score, positive_matches = train_pair_score_with_matches(
            self.loma,
            query_keypoints,
            query_descriptors,
            positive_keypoints,
            positive_descriptors,
            preserve_descriptor_grad=True,
        )
        negative_score, negative_matches = train_pair_score_with_matches(
            self.loma,
            query_keypoints,
            query_descriptors,
            negative_keypoints,
            negative_descriptors,
            preserve_descriptor_grad=True,
        )
        return positive_score, positive_matches, negative_score, negative_matches


@torch.inference_mode()
def eval_pair_scores(model: nn.Module, keypoints0: torch.Tensor, descriptors0: torch.Tensor,
                     keypoints1: torch.Tensor, descriptors1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized mutual-match confidence and match counts."""
    from loma.loma import filter_matches

    scores = model(keypoints0, keypoints1, descriptors0, descriptors1)["scores"]
    base_model = model.module if hasattr(model, "module") else model
    _, _, matching_scores0, _ = filter_matches(scores, base_model.cfg.filter_threshold)
    confidence = matching_scores0.sum(dim=-1)
    denominator = min(keypoints0.shape[1], keypoints1.shape[1])
    confidence = confidence / max(1, denominator)
    return confidence, (matching_scores0 > 0).sum(dim=-1)


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
