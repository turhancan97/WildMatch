"""Fixed DaD keypoint cache used when LoMa descriptors are fine-tuned."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MANIFEST_NAME = "manifest.json"
CACHE_FORMAT = "lynx-loma-keypoints-v1"


def module_fingerprint(module: torch.nn.Module) -> str:
    """Stable fingerprint of a model component's parameters and buffers."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class LomaKeypointCache:
    """Cache normalized keypoints and resized image dimensions, never descriptors."""

    def __init__(
        self,
        root: str | Path,
        *,
        variant: str,
        resize: int,
        num_keypoints: int,
        detector_sha256: str,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_NAME} in keypoint cache {self.root}")
        self.manifest = json.loads(manifest_path.read_text())
        expected = {
            "format": CACHE_FORMAT,
            "backend": "loma",
            "variant": variant,
            "resize": resize,
            "num_keypoints": num_keypoints,
            "patch_size": 14,
            "detector_sha256": detector_sha256,
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if self.manifest.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (wanted {wanted!r})"
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"incompatible LoMa keypoint cache at {self.root}: {details}")
        if self.manifest.get("complete") is not True:
            raise ValueError(f"LoMa keypoint cache is incomplete: {self.root}")

    def path_for(self, relative_path: str) -> Path:
        return self.root / Path(relative_path).with_suffix(".npz")

    def has(self, relative_path: str) -> bool:
        return self.path_for(relative_path).is_file()

    def load(self, relative_path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.path_for(relative_path)
        try:
            with np.load(path) as data:
                keypoints = np.asarray(data["keypoints"])
                image_size = np.asarray(data["image_size"])
        except FileNotFoundError:
            raise KeyError(
                f"frame {relative_path!r} is missing from LoMa keypoint cache {self.root}"
            ) from None
        if keypoints.shape != (self.manifest["num_keypoints"], 2):
            raise ValueError(f"{path} has invalid keypoint shape {keypoints.shape}")
        if image_size.shape != (2,):
            raise ValueError(f"{path} has invalid image_size shape {image_size.shape}")
        return (
            torch.from_numpy(keypoints.astype(np.float32, copy=False)).unsqueeze(0).to(device),
            torch.from_numpy(image_size.astype(np.int64, copy=False)).to(device),
        )
