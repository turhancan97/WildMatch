"""Calibration policy compatible with official WildFusion all-pairs fitting."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

CALIBRATION_MODES = ("same_set", "disjoint")


def _feature_paths(dataset: Any) -> Optional[Tuple[str, ...]]:
    frame = getattr(dataset, "df", getattr(dataset, "metadata", None))
    if frame is None:
        return None
    for column in ("path", "filepath", "file", "image_path"):
        if column in frame.columns:
            return tuple(frame[column].astype(str).tolist())
    return None


def _is_same_image_set(dataset_a: Any, dataset_b: Any) -> bool:
    if dataset_a is dataset_b:
        return True
    paths_a = _feature_paths(dataset_a)
    paths_b = _feature_paths(dataset_b)
    return paths_a is not None and paths_a == paths_b


def resolve_calibration_mode(value: Any) -> str:
    """Validate ``benchmark.calibration.mode``."""
    mode = str(value)
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"benchmark.calibration.mode must be one of {CALIBRATION_MODES}, got {mode!r}")
    return mode


def disjoint_calibration_frames(
    metadata: pd.DataFrame, label_col: str, size: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a calibration sample into two image sets that share identities but no image.

    Identities are visited in a seeded random order, and each visit takes two unused images of one
    identity, one for each side, so every side-A image has a positive on side B and no pair is an
    image with itself. Rounds repeat until ``size`` images are taken or no identity has two images
    left; remaining slots take single images (negatives only), alternating sides. Identities with
    many images are thus not over-represented. Fails when no identity has two images, because
    calibration then has no positive pairs.
    """
    if size < 2:
        raise ValueError("disjoint calibration needs dataset.calibration_size >= 2")
    rng = np.random.default_rng(int(seed))
    labels = metadata[label_col].astype(str)
    order = list(dict.fromkeys(labels.tolist()))
    order = [order[i] for i in rng.permutation(len(order))]
    remaining = {label: list(rng.permutation(np.flatnonzero(labels.to_numpy() == label))) for label in order}
    if not any(len(rows) >= 2 for rows in remaining.values()):
        raise ValueError("disjoint calibration needs at least one identity with two images")
    side_a: list = []
    side_b: list = []
    progress = True
    while progress and len(side_a) + len(side_b) + 2 <= size:
        progress = False
        for label in order:
            if len(side_a) + len(side_b) + 2 > size:
                break
            rows = remaining[label]
            if len(rows) >= 2:
                side_a.append(rows.pop())
                side_b.append(rows.pop())
                progress = True
    for label in order:
        rows = remaining[label]
        while rows and len(side_a) + len(side_b) < size:
            (side_a if len(side_a) <= len(side_b) else side_b).append(rows.pop())
    return metadata.iloc[side_a], metadata.iloc[side_b]


def fit_pipeline_calibration(
    pipeline: Any,
    dataset_a: Any,
    dataset_b: Any,
    *,
    exclude_self_pairs: bool = True,
) -> Dict[str, Any]:
    """Fit one pipeline using all pairs, optionally excluding same-image diagonals."""
    features_a = pipeline.get_feature_dataset(dataset_a)
    features_b = pipeline.get_feature_dataset(dataset_b)
    scores = np.asarray(pipeline.matcher(features_a, features_b))
    labels_a = np.asarray(getattr(features_a, "labels_string"))
    labels_b = np.asarray(getattr(features_b, "labels_string"))
    hits = labels_a[:, None] == labels_b[None, :]
    use = np.ones(scores.shape, dtype=bool)
    same_set = _is_same_image_set(dataset_a, dataset_b)
    excluded = 0
    if exclude_self_pairs and same_set and scores.shape[0] == scores.shape[1]:
        diagonal = np.eye(scores.shape[0], dtype=bool)
        use &= ~diagonal
        excluded = int(diagonal.sum())
    calibration_scores = scores[use]
    calibration_hits = hits[use]
    if calibration_scores.size == 0:
        raise ValueError("WildFusion calibration has no usable pairs after self-pair exclusion")
    pipeline.calibration.fit(calibration_scores, calibration_hits)
    pipeline.calibration_done = True
    return {
        "source": (
            ("same_set_excluding_self_pairs" if exclude_self_pairs else "official_all_pairs")
            if same_set
            else "disjoint_sets"
        ),
        "total_pairs": int(scores.size),
        "used_pairs": int(calibration_scores.size),
        "excluded_self_pairs": excluded,
    }


def fit_wildfusion_calibration(
    wildfusion: Any,
    dataset_a: Any,
    dataset_b: Any,
    *,
    exclude_self_pairs: bool = True,
    official_same_set: bool = False,
) -> Dict[str, Any]:
    """Fit all calibrated WildFusion pipelines and return provenance."""
    if official_same_set:
        exclude_self_pairs = False
    same_set = _is_same_image_set(dataset_a, dataset_b)
    if same_set and exclude_self_pairs and not official_same_set:
        warnings.warn(
            "WildFusion calibration uses a same-set fallback; self-pairs are excluded. "
            "Configure a disjoint calibration split for paper-grade evaluation.",
            RuntimeWarning,
            stacklevel=2,
        )
    diagnostics = []
    for pipeline in list(getattr(wildfusion, "calibrated_pipelines", [])):
        diagnostics.append(
            fit_pipeline_calibration(
                pipeline,
                dataset_a,
                dataset_b,
                exclude_self_pairs=exclude_self_pairs,
            )
        )
    if not diagnostics:
        raise ValueError("WildFusion has no calibrated pipelines")
    return {
        "source": "official_same_set_all_pairs" if official_same_set else diagnostics[0]["source"],
        "pipelines": diagnostics,
        "total_pairs": int(sum(item["total_pairs"] for item in diagnostics)),
        "used_pairs": int(sum(item["used_pairs"] for item in diagnostics)),
        "excluded_self_pairs": int(sum(item["excluded_self_pairs"] for item in diagnostics)),
    }
