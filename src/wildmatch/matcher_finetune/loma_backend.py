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


def freeze_loma_backbone(model: nn.Module) -> None:
    """Freeze and keep the detector/descriptor in eval mode."""
    for attr in ("_detector", "_descriptor"):
        module = getattr(model, attr, None)
        if module is None:
            raise AttributeError(f"LoMa model has no {attr} module")
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def set_loma_train_mode(model: nn.Module, training: bool) -> None:
    model.train(training)
    freeze_loma_backbone(model)


@torch.inference_mode()
def extract_loma_features(model: nn.Module, images: torch.Tensor, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
    keypoints, descriptors, _, _ = model.detect_and_describe(images, num_keypoints=num_keypoints)
    return keypoints, descriptors


def train_pair_score(model: nn.Module, keypoints0: torch.Tensor, descriptors0: torch.Tensor,
                     keypoints1: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
    """Differentiable confidence used by the lynx positive/negative loss."""
    scores = model(keypoints0, keypoints1, descriptors0, descriptors1)["scores"]
    # In training mode LoMa returns a log assignment matrix with a dustbin.
    pair_scores = scores[:, :-1, :-1].exp()
    row_best = pair_scores.max(dim=-1).values.mean(dim=-1)
    col_best = pair_scores.max(dim=-2).values.mean(dim=-1)
    return 0.5 * (row_best + col_best)


@torch.inference_mode()
def eval_pair_scores(model: nn.Module, keypoints0: torch.Tensor, descriptors0: torch.Tensor,
                     keypoints1: torch.Tensor, descriptors1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized mutual-match confidence and match counts."""
    from loma.loma import filter_matches

    scores = model(keypoints0, keypoints1, descriptors0, descriptors1)["scores"]
    _, _, matching_scores0, _ = filter_matches(scores, model.cfg.filter_threshold)
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
