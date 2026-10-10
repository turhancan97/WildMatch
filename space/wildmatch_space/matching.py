"""Matchers for the Space: default Vismatch weights or the published fine-tuned checkpoints (GPU).

Fine-tuned files come from the Hub repository in ``conf/weights.yaml`` (default set: the paper's LoMa and
the shared-recipe RDD-LightGlue retrains), fetched with ``wildmatch.weights.download``, which checks every
SHA-256 and keeps each CzechLynx ``czechlynx_protocol.json`` beside its weights. Settings follow
``conf/probe.yaml`` (``resize_max`` 512, 512 keypoints, LoMa-B, the matcher's default threshold); fine-tuned
checkpoints load the matcher only, as in the paper's matcher-only rows.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

RESIZE_MAX = 512
TOP_K = 512
LOMA_ARCH = "LoMa-B"
MAX_LIVE_MATCHERS = 3

_matchers: "OrderedDict[Tuple[str, Optional[str]], object]" = OrderedDict()


def checkpoint_root() -> Path:
    return Path(os.environ.get("WILDMATCH_SPACE_CHECKPOINTS", Path.home() / ".cache" / "wildmatch-checkpoints"))


def checkpoint_path(dataset: str, matcher: str) -> Path:
    """The fine-tuned model file for ``dataset``/``matcher``, downloaded and verified on first use."""
    from wildmatch import weights

    manifest = weights.load_manifest()
    entries = weights.select(manifest["entries"], datasets=[dataset], matchers=[matcher])
    if len(entries) != 1:
        raise weights.WeightsError(f"expected one default-set checkpoint for {dataset}/{matcher}, found {len(entries)}")
    root = checkpoint_root()
    repo = os.environ.get("WILDMATCH_HUB_REPO") or manifest["repo_id"]
    weights.download(entries, root, repo, manifest.get("revision") or "main", log=lambda _: None)
    model = [item for item in entries[0]["files"] if item["local"].endswith(".safetensors")]
    return root / model[0]["local"]


def on_zero_gpu() -> bool:
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true"}


def matcher(matcher_name: str, dataset: Optional[str], keep: Optional[bool] = None):
    """A ``VismatchMatcherBackend``; ``dataset=None`` gives the default (not fine-tuned) weights.

    Kept matchers are reused. On ZeroGPU only matchers built at start-up (``keep=True``, outside a GPU call)
    are kept, because CUDA memory allocated inside a GPU call does not outlive it; the others are rebuilt
    per request. Elsewhere the ``MAX_LIVE_MATCHERS`` most recently used ones are kept.
    """
    import torch

    from wildmatch.matchers.vismatch import VismatchMatcherBackend
    from wildmatch.matchers.vismatch_profiles import default_matcher_threshold

    key = (matcher_name, dataset)
    if key in _matchers:
        _matchers.move_to_end(key)
        return _matchers[key]
    if keep is None:
        keep = not on_zero_gpu()
    custom = dataset is not None
    backend = VismatchMatcherBackend(
        matcher_name,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        TOP_K,
        default_matcher_threshold(matcher_name),
        checkpoint_source="custom" if custom else "default",
        checkpoint_path=str(checkpoint_path(dataset, matcher_name)) if custom else None,
        checkpoint_components="matcher_only" if custom else "auto",
        loma_arch=LOMA_ARCH,
        resize_max=RESIZE_MAX,
    )
    if keep:
        _matchers[key] = backend
        while not on_zero_gpu() and len(_matchers) > MAX_LIVE_MATCHERS:
            _matchers.popitem(last=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return backend


@dataclass
class PairMatch:
    score: float
    match_count: int
    kpts_left: np.ndarray  # pixel coordinates on the raw left photo
    kpts_right: np.ndarray
    confidences: np.ndarray


def features(backend, masked: Image.Image):
    from wildmatch.matchers.vismatch_preprocessing import to_rgb_float_tensor

    return backend.extract_frame(to_rgb_float_tensor(masked.convert("RGB")))


def match(backend, left_feat, right_feat, left_size: Tuple[int, int], right_size: Tuple[int, int]) -> PairMatch:
    """Match two extracted frames; keypoints are returned on the raw photos (``size`` = (W, H))."""
    from wildmatch.matchers.vismatch import _loma_points_to_processed_pixel

    from .drawing import processed_to_raw_pixels

    result = backend.match_features(left_feat, right_feat)
    empty = np.empty((0, 2), dtype=np.float32)
    k0 = result.matched_kpts0 if result.matched_kpts0 is not None else empty
    k1 = result.matched_kpts1 if result.matched_kpts1 is not None else empty
    if backend.matcher_name == "loma":  # normalized [-1, 1] -> matcher-input pixels, as _match_frames does
        k0, k1 = _loma_points_to_processed_pixel(k0, left_feat), _loma_points_to_processed_pixel(k1, right_feat)
    confidence = result.confidences if result.confidences is not None else np.asarray([], dtype=np.float32)
    raw0 = processed_to_raw_pixels(k0, left_feat.image_size, (left_size[1], left_size[0]))
    raw1 = processed_to_raw_pixels(k1, right_feat.image_size, (right_size[1], right_size[0]))
    return PairMatch(float(result.score), int(result.match_count), raw0, raw1, np.asarray(confidence, dtype=np.float64))
