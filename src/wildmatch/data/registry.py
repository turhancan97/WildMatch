"""The dataset registry: one Hydra config per dataset/split in ``conf/dataset/<key>.yaml``.

Each file holds the probe's ``dataset`` fields (name, animal, root, metadata file, label and
split columns, mask handling) plus a ``registry`` block (paper key, label, source, whether
it is reported in the paper, default fine-tuned checkpoints, licence, download route).
Hydra runs select an entry with ``dataset=<key>``; other code calls :func:`load_dataset`,
which resolves the entry against a path profile from :mod:`wildmatch.paths`.
"""

from __future__ import annotations

from importlib import resources
from typing import Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from wildmatch.paths import active_profile, load_paths


def _dataset_dir():
    return resources.files("wildmatch") / "conf" / "dataset"


def dataset_keys() -> List[str]:
    return sorted(entry.name[:-5] for entry in _dataset_dir().iterdir() if entry.name.endswith(".yaml"))


def load_dataset(key: str, profile: Optional[str] = None) -> DictConfig:
    """The registry entry ``key`` with every interpolation resolved."""
    source = _dataset_dir() / f"{key}.yaml"
    if not source.is_file():
        raise KeyError(f"unknown dataset {key!r}; registry keys: {', '.join(dataset_keys())}")
    merged = OmegaConf.create(
        {
            "paths": load_paths(profile),
            "dataset": OmegaConf.create(source.read_text(encoding="utf-8")),
        }
    )
    OmegaConf.resolve(merged)
    return merged.dataset


def load_registry(profile: Optional[str] = None) -> Dict[str, DictConfig]:
    """Every registry entry, resolved against one path profile."""
    name = active_profile(profile)
    return {key: load_dataset(key, name) for key in dataset_keys()}
