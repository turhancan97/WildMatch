"""Reader for the benchmark-compatible LoMa per-frame feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MANIFEST_NAME = "manifest.json"


def weights_fingerprint(path: str | Path) -> str:
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            base = metadata.get("base_weights")
            if base:
                base_path = Path(base).expanduser()
                if not base_path.is_absolute():
                    base_path = path / base_path
                if base_path.exists():
                    path = base_path
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LomaFeatureCache:
    """Load fixed-keypoint LoMa features through the training dataset API."""

    def __init__(
        self,
        root: str | Path,
        *,
        variant: str,
        resize: int,
        num_keypoints: int,
        weights: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {self.root}; build the full LoMa cache first"
            )
        self.manifest = json.loads(manifest_path.read_text())
        expected = {
            "backend": "loma",
            "variant": variant,
            "resize": resize,
            "num_keypoints": num_keypoints,
            "patch_size": 14,
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
            raise ValueError(f"incompatible LoMa cache at {self.root}: {details}")
        if weights is not None and self.manifest.get("weights_sha256"):
            wanted_hash = weights_fingerprint(weights)
            if self.manifest["weights_sha256"] != wanted_hash:
                raise ValueError(
                    f"LoMa cache at {self.root} was built from different weights "
                    f"(cache={self.manifest['weights_sha256']}, wanted={wanted_hash})"
                )

    def path_for(self, rel: str) -> Path:
        return self.root / Path(rel).with_suffix(".npz")

    def has(self, rel: str) -> bool:
        return self.path_for(rel).exists()

    def _load(self, rel: str) -> dict[str, torch.Tensor]:
        path = self.path_for(rel)
        try:
            with np.load(path) as data:
                keypoints = np.asarray(data["keypoints"])
                descriptors = np.asarray(data["descriptors"])
                scores = np.asarray(data["scores"])
                image_size = np.asarray(data["image_size"])
        except FileNotFoundError:
            raise KeyError(
                f"frame {rel!r} is missing from the LoMa cache at {self.root}"
            ) from None

        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"{path} has invalid keypoint shape {keypoints.shape}")
        if descriptors.ndim != 2 or descriptors.shape[0] != keypoints.shape[0]:
            raise ValueError(f"{path} has incompatible descriptor shape {descriptors.shape}")
        if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
            raise ValueError(f"{path} has incompatible score shape {scores.shape}")
        if image_size.shape != (2,):
            raise ValueError(f"{path} has invalid image_size shape {image_size.shape}")
        if keypoints.shape[0] != self.manifest["num_keypoints"]:
            raise ValueError(
                f"{path} contains {keypoints.shape[0]} keypoints; the cache must use fixed "
                f"num_keypoints={self.manifest['num_keypoints']}"
            )

        return {
            "keypoints": torch.from_numpy(keypoints.astype(np.float32, copy=False)),
            "descriptors": torch.from_numpy(descriptors.astype(np.float32, copy=False)),
            "image_size": torch.from_numpy(image_size.astype(np.int64, copy=False)),
        }

    def load_padded(self, rel: str) -> dict[str, torch.Tensor]:
        """Dataset-compatible single-frame loader.

        The benchmark format already contains a fixed keypoint count, so no
        padding or masks are needed for LoMa.
        """
        return self._load(rel)

    def load_padded_stack(self, rels: list[str]) -> dict[str, torch.Tensor]:
        items = [self._load(rel) for rel in rels]
        if not items:
            raise ValueError("cannot load an empty LoMa feature stack")
        return {
            key: torch.stack([item[key] for item in items])
            for key in items[0]
        }

    def load(self, rel: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._load(rel)
        return item["keypoints"].unsqueeze(0).to(device), item["descriptors"].unsqueeze(0).to(device)
