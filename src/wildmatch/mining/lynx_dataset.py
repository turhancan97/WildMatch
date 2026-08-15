"""Dataset and feature-cache utilities that do not import RDD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class FrameFeat:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray


@dataclass
class SequenceEntry:
    split: str
    lynx_id: str
    site: str
    sequence_id: str
    frame_paths: List[Path]

    @property
    def name(self) -> str:
        return f"{self.lynx_id}/{self.site}/{self.sequence_id}"


def list_sequences(root: Path, split: str) -> List[SequenceEntry]:
    entries: List[SequenceEntry] = []
    split_dir = root / split
    for lynx_dir in sorted(split_dir.iterdir()):
        if not lynx_dir.is_dir():
            continue
        for site_dir in sorted(lynx_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            for seq_dir in sorted(site_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("frame_*.jpg"))
                if frames:
                    entries.append(SequenceEntry(split, lynx_dir.name, site_dir.name, seq_dir.name, frames))
    return entries


def sample_frames(frame_paths: List[Path], max_frames: int) -> List[Path]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, num=max_frames, dtype=int)
    return [frame_paths[index] for index in indices]


def load_cached_feat(path: Path) -> FrameFeat:
    data = np.load(path)
    return FrameFeat(
        keypoints=data["keypoints"],
        descriptors=data["descriptors"],
        scores=data["scores"],
        image_size=data["image_size"],
    )

