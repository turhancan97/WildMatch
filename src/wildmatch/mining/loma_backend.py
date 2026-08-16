"""LoMa feature-cache and matching backend for the lynx benchmark."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image


LOMA_PATCH_SIZE = 14


def _load_state(path: Path) -> dict[str, torch.Tensor]:
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
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state dictionary")
    return {k.removeprefix("module."): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def _files(path: Path) -> tuple[Path, Path | None]:
    if path.is_file():
        return path, None
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    model_path = next(
        (path / name for name in ("model.safetensors", "matcher.safetensors", "weights.pth") if (path / name).exists()),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(f"No LoMa model file found in {path}")
    base = metadata.get("base_weights")
    base_path = Path(base).expanduser() if base else None
    if base_path is not None and not base_path.is_absolute():
        base_path = path / base_path
    return model_path, base_path if base_path and base_path.exists() else None


def build_loma(device: torch.device, weights: Path, variant: str = "loma-b"):
    try:
        from loma.loma import LoMa, LoMaB, LoMaB128, LoMaG, LoMaL, LoMaR
    except ImportError as exc:
        raise ImportError("LoMa is not installed in this environment") from exc

    config_cls = {
        "loma-b": LoMaB,
        "loma-b128": LoMaB128,
        "loma-l": LoMaL,
        "loma-g": LoMaG,
        "loma-r": LoMaR,
    }.get(variant.lower())
    if config_cls is None:
        raise ValueError(f"Unsupported LoMa variant: {variant}")
    model_path, base_path = _files(weights)
    config = config_cls()
    names = {field.name for field in fields(config)}
    updates = {name: value for name, value in (("weights_url", None), ("compile", False)) if name in names}
    model = LoMa(replace(config, **updates) if updates else config).to(device).eval()

    def overlay(path: Path) -> None:
        state = _load_state(path)
        model_state = model.state_dict()
        matched = {key: value for key, value in state.items() if key in model_state}
        if not matched:
            raise RuntimeError(f"{path} contains no keys compatible with {variant}")
        model.load_state_dict(matched, strict=False)

    if base_path is not None:
        overlay(base_path)
    overlay(model_path)
    model.eval()
    return model


def resize_image(image: torch.Tensor, resize_max: int) -> torch.Tensor:
    """Resize LoMa inputs for DeDoDe's DINOv2 ViT-L/14 descriptor."""
    if resize_max <= 0:
        return image
    _, _, height, width = image.shape
    scale = resize_max / max(height, width)
    new_height = max(LOMA_PATCH_SIZE, int(height * scale) // LOMA_PATCH_SIZE * LOMA_PATCH_SIZE)
    new_width = max(LOMA_PATCH_SIZE, int(width * scale) // LOMA_PATCH_SIZE * LOMA_PATCH_SIZE)
    return torch.nn.functional.interpolate(image.float(), (new_height, new_width), mode="bilinear", align_corners=False)


@torch.inference_mode()
def extract_frame(model, path: Path, device: torch.device, num_keypoints: int, resize_max: int):
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = resize_image(image, resize_max).to(device)
    keypoints, descriptors, _, _ = model.detect_and_describe(image, num_keypoints=num_keypoints)
    return {
        "keypoints": keypoints[0].cpu().numpy(),
        "descriptors": descriptors[0].cpu().numpy(),
        "scores": np.ones(keypoints.shape[1], dtype=np.float32),
        "image_size": np.asarray(image.shape[-2:], dtype=np.int32),
    }


def _feature_tensors(features, device):
    keypoints = torch.from_numpy(np.stack([feature.keypoints for feature in features])).to(device)
    descriptors = torch.from_numpy(np.stack([feature.descriptors for feature in features])).to(device)
    return keypoints, descriptors


@torch.inference_mode()
def score_batch_loma(model, query_features, gallery_features, device):
    from loma.loma import filter_matches

    if not query_features or not gallery_features:
        return torch.empty(len(query_features), len(gallery_features), device=device)
    if len({feature.keypoints.shape[0] for feature in query_features + gallery_features}) != 1:
        raise ValueError("LoMa feature cache contains variable keypoint counts; rebuild it with fixed --num_keypoints")
    g_k, g_d = _feature_tensors(gallery_features, device)
    rows = []
    for query in query_features:
        q_k, q_d = _feature_tensors([query], device)
        q_k = q_k.expand(len(gallery_features), -1, -1)
        q_d = q_d.expand(len(gallery_features), -1, -1)
        scores = model(q_k, g_k, q_d, g_d)["scores"]
        _, _, matching_scores0, _ = filter_matches(scores, model.cfg.filter_threshold)
        norm = max(1, min(query.keypoints.shape[0], min(feature.keypoints.shape[0] for feature in gallery_features)))
        rows.append(matching_scores0.sum(dim=-1) / norm)
    return torch.stack(rows)


@torch.inference_mode()
def score_all_loma(model, query_features, gallery_features, device, batch_size: int):
    # LoMa uses a fixed keypoint count, so gallery batches are safe to stack.
    chunks = []
    for start in range(0, len(gallery_features), batch_size):
        chunks.append(score_batch_loma(model, query_features, gallery_features[start:start + batch_size], device).cpu())
    return torch.cat(chunks, dim=-1)


@torch.inference_mode()
def score_pairs_loma(model, query_features, gallery_features, pairs, device, batch_size, n_cols):
    output = torch.zeros(len(query_features), n_cols)
    for query_index, columns in pairs.items():
        for start in range(0, len(columns), batch_size):
            selected = columns[start:start + batch_size]
            row = score_batch_loma(model, [query_features[query_index]], [gallery_features[index] for index in selected], device)
            output[query_index, torch.tensor(selected)] = row[0].cpu()
    return output
