#!/usr/bin/env python
"""Qualitative figure: one successful fine-tuned LoMa match per paper dataset.

Two steps:

``candidates``
    For every paper dataset, take the fine-tuned LoMa (Vismatch, k=50) probe run
    behind the paper tables, find the queries whose top-1 gallery image is the
    correct individual (shared descending-score, lowest-index tie rule), drop
    pairs from the same encounter/day where the metadata records it and visual
    near-duplicates (dHash), and rank the rest from hardest to easiest by
    MegaDescriptor-L cosine similarity of the two model inputs. The LoMa matcher
    is re-run on the hardest pairs to count correspondences; pairs with at least
    ``--min-matches`` are written to ``<out>/match_candidates/<dataset>.csv``
    with a contact sheet of the top ``--sheet`` pairs.

``render``
    Draws the pairs pinned in ``EXAMPLES`` (one per dataset, confirmed by eye
    from the contact sheets) as a 4 x 2 figure on the original raw photos: the
    matcher runs on the exact probe input (pre-masked file, or the CzechLynx
    RLE mask applied through ``BenchmarkDatasetView``), and its normalized LoMa
    keypoints are mapped to raw-photo pixels. Only the ``--top-matches``
    strongest correspondences are drawn; the label reports the total.
    Writes ``match_examples.{pdf,png}`` and a ``match_examples.json`` sidecar
    (runs, checkpoint SHA-256, pairs, match counts, scores).

Both steps need the matcher, so run them on an A100/H100 (or CPU for a few
pairs). Add a pair to ``EXAMPLES`` only after inspecting it in the sheets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
for entry in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from reid.reporting.paper_datasets import PAPER_PROFILES, PaperProfile  # noqa: E402

# Paper order (alphabetical by panel label, user decision 2026-09-30) and labels.
PAPER_ORDER: Tuple[Tuple[str, str], ...] = (
    ("lynx_closed", "Czech Lynx"),
    ("hyena", "Hyena"),
    ("leopard", "Leopard"),
    ("nyala", "Nyala"),
    ("salamander", "Salamander"),
    ("sea_star", "Sea Star"),
    ("turtle", "Turtle (Zindi)"),
    ("whale_shark", "Whale Shark"),
)
# Fine-tuned LoMa (custom, matcher_only, k=50) runs behind the paper tables.
RUN_ROOT = ROOT_DIR / "experiments" / "probe"
RUNS: Dict[str, str] = {
    "nyala": "WildlifeReID-10k/NyalaData/split/megadescriptor-l/vismatch/loma/20260821T154814Z_f4b5fd8f",
    "hyena": "WildlifeReID-10k/HyenaID2022/split/megadescriptor-l/vismatch/loma/20260905T170746Z_cbd935ba",
    "leopard": "WildlifeReID-10k/LeopardID2022/split/megadescriptor-l/vismatch/loma/20260905T165838Z_4d56bf7d",
    "sea_star": "WildlifeReID-10k/SeaStarReID2023/split/megadescriptor-l/vismatch/loma/20260906T230044Z_c1426a21",
    "whale_shark": "WildlifeReID-10k/WhaleSharkID/split/megadescriptor-l/vismatch/loma/20260821T213729Z_5d10e41d",
    "turtle": "WildlifeReID-10k/ZindiTurtleRecall/split/megadescriptor-l/vismatch/loma/20260823T181530Z_1ee24291",
    "salamander": "SalamanderID2025/SalamanderID2025/split/megadescriptor-l/vismatch/loma/20260930T083009Z_57276f29",
    "lynx_closed": "CzechLynx_v2/CzechLynx/split-time_closed/megadescriptor-l/vismatch/loma/20260920T122915Z_0015f14a",
}
# Dates for WildlifeReID-10k datasets live in the full metadata file only.
WILDLIFE_FULL_METADATA = Path("/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata.csv")


class Example(NamedTuple):
    dataset: str
    query_path: str  # metadata `path` of the query (model-input path)
    gallery_path: str  # metadata `path` of its top-1 gallery image
    note: str
    # Background dimming strength for this panel; None uses --dim. Raised only where
    # the probe mask follows the animal cleanly, since strong dimming exposes outlines.
    dim: Optional[float] = None


# Confirmed by eye from the candidate contact sheets. Keep paper order.
EXAMPLES: List[Example] = [
    Example("lynx_closed", "CzechLynx/foe_bohemia/lynx_041/22605_lynx_041.jpg",
            "CzechLynx/foe_bohemia/lynx_041/22326_lynx_041.jpg",
            "daylight sheet row 12 (17-09-2022 vs 12-01-2020); cosine 0.38, 396 matches, probe score 0.734", dim=0.35),
    Example("hyena", "masked_images/HyenaID2022/hyena.coco/images/train2022/000000002398_2422.jpg",
            "masked_images/HyenaID2022/hyena.coco/images/train2022/000000001579_1599.jpg",
            "sheet row 3; cosine 0.28, 350 matches, probe score 0.587", dim=0.0),
    Example("leopard", "masked_images/LeopardID2022/leopard.coco/images/train2022/000000003152_3164.jpg",
            "masked_images/LeopardID2022/leopard.coco/images/train2022/000000002995_3006.jpg",
            "sheet row 3; cosine 0.32, 389 matches, probe score 0.670", dim=0.4),
    Example("nyala", "masked_images/NyalaData/wildlife_reidentification-main/Nyala_Data_Zero/train/100/903_rightphoto_5f81d5ae0075167b.jpg",
            "masked_images/NyalaData/wildlife_reidentification-main/Nyala_Data_Zero/train/100/530_rightphoto_64ee1486b979a67d.jpg",
            "sheet row 5; cosine 0.32, 383 matches, probe score 0.661", dim=0.4),
    Example("salamander", "masked_images/query/images/616e46c4c91594a7_795.jpg",
            "masked_images/database/images/25372f49baf1e44c_305.jpg",
            "sheet row 2; cosine 0.52, 375 matches, probe score 0.646", dim=0.4),
    Example("sea_star", "masked_images/SeaStarReID2023/sea-star-re-id/Asru76/IMG_8688_3a7f724d27a8c2f7.jpg",
            "masked_images/SeaStarReID2023/sea-star-re-id/Asru76/IMG_8548_eb1339ecfd3810df.jpg",
            "sheet row 5; cosine 0.52, 420 matches, probe score 0.771", dim=0.35),
    Example("turtle", "masked_images/ZindiTurtleRecall/images/ID_BACZNN7V_ID_BACZNN7V.JPG",
            "masked_images/ZindiTurtleRecall/images/ID_HL5N7GRB_ID_HL5N7GRB.JPG",
            "sheet row 3; cosine 0.23, 373 matches, probe score 0.646", dim=0.35),
    Example("whale_shark", "masked_images/WhaleSharkID/whaleshark.coco/images/train2020/000000007575_7574.jpg",
            "masked_images/WhaleSharkID/whaleshark.coco/images/train2020/000000005151_5150.jpg",
            "sheet row 5; cosine 0.35, 368 matches, probe score 0.655"),
]


# ── pure helpers (unit-tested, no model needed) ──────────────────────────────
def normalized_to_raw_pixels(points: np.ndarray, raw_hw: Sequence[int]) -> np.ndarray:
    """LoMa normalized [-1, 1] keypoints -> pixel coordinates on the raw photo.

    LoMa coordinates are relative to the full frame, and the matcher input is the
    raw frame resampled without cropping, so the raw photo is addressed directly
    with the Vismatch/LoMa half-pixel convention: x = W (x_n + 1) / 2 - 0.5.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    height, width = int(raw_hw[0]), int(raw_hw[1])
    out = np.empty_like(points)
    out[:, 0] = width * (points[:, 0] + 1.0) / 2.0 - 0.5
    out[:, 1] = height * (points[:, 1] + 1.0) / 2.0 - 0.5
    return out


def raw_image_path(profile: PaperProfile, row: Mapping[str, Any]) -> str:
    """Metadata row -> path (relative to the profile root) of the unmasked photo."""
    if "original_path" in row and isinstance(row["original_path"], str) and row["original_path"]:
        return row["original_path"]  # SalamanderID2025 records its SAM3 source explicitly
    path = str(row["path"])
    if profile.mask_col is None and path.startswith("masked_images/"):
        return "images/" + path[len("masked_images/"):]  # pre-masked WildlifeReID-10k tree
    return path  # CzechLynx: the path is the raw photo; the mask is applied at load time


def dhash(image: Image.Image, size: int = 8) -> int:
    """Difference hash of an image (64 bits for size=8)."""
    gray = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int(sum(1 << index for index, bit in enumerate(bits) if bit))


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip() != ""


def pair_rejection(query: Mapping[str, Any], gallery: Mapping[str, Any],
                   hash_distance: Optional[int], min_hash_distance: int) -> Optional[str]:
    """Why a (query, gallery) pair may not be shown, or None if it may.

    Same encounter or same day (when both sides record it) would show two frames
    of one sighting; a small dHash distance flags visual near-duplicates for the
    datasets without such metadata.
    """
    for key, reason in (("encounter", "same_encounter"), ("date", "same_date")):
        if _present(query.get(key)) and _present(gallery.get(key)) and str(query[key]) == str(gallery[key]):
            return reason
    if hash_distance is not None and hash_distance < min_hash_distance:
        return "near_duplicate"
    return None


def correct_top1_pairs(rows: np.ndarray, cols: np.ndarray, values: np.ndarray,
                       query_labels: Sequence[Any], database_labels: Sequence[Any]) -> List[Tuple[int, int, float]]:
    """(query, gallery, score) for queries whose top-1 is the correct identity.

    Uses the shared ranking rule on a sparse score matrix: descending score,
    lowest database index on ties; unscored positions never rank.
    """
    rows, cols, values = (np.asarray(a) for a in (rows, cols, values))
    finite = np.isfinite(values)
    rows, cols, values = rows[finite], cols[finite], values[finite]
    order = np.lexsort((cols, -values, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    first = np.r_[True, rows[1:] != rows[:-1]] if len(rows) else np.zeros(0, dtype=bool)
    query_labels, database_labels = np.asarray(query_labels), np.asarray(database_labels)
    out = []
    for q, g, s in zip(rows[first], cols[first], values[first]):
        if query_labels[q] == database_labels[g]:
            out.append((int(q), int(g), float(s)))
    return out


def resolve_recorded_checkpoint(recorded: Path, sha256: str, search_root: Path,
                                hasher: Callable[[Path], str]) -> Path:
    """The file whose SHA-256 equals the one the probe manifest recorded.

    Checkpoint directories were renamed and one file was overwritten after some
    runs (e.g. ``legacy/`` -> ``legacy-loma-mined/``), so the recorded path alone
    is not trusted. Prefer the recorded path, then ``epoch_*`` copies over
    ``latest``; fail closed when no file under ``search_root`` has the hash.
    """
    if recorded.is_file() and hasher(recorded) == sha256:
        return recorded
    candidates = sorted(search_root.rglob("*.safetensors"),
                        key=lambda path: ("/latest/" in str(path), str(path)))
    for candidate in candidates:
        if candidate.is_file() and hasher(candidate) == sha256:
            return candidate
    raise FileNotFoundError(f"no checkpoint with SHA-256 {sha256} under {search_root} (recorded {recorded})")


def photo_quality(photo: Image.Image, foreground: np.ndarray) -> Dict[str, float]:
    """Foreground share, colour saturation and sharpness of the animal in a raw photo.

    Saturation is the mean HSV saturation over the foreground, so greyscale frames
    score near zero; ``warm`` is the share of foreground pixels with an orange-to-brown
    hue (about 5-50 degrees) and saturation above 0.15, which separates daylight coats
    from pink/magenta infrared frames that saturation alone does not; sharpness is the
    variance of a 3x3 Laplacian of the foreground's grey level. Used only to shortlist
    presentable pairs.
    """
    mask = np.asarray(foreground, dtype=bool)
    if mask.shape != (photo.height, photo.width) or not mask.any():
        return {"foreground": float(mask.mean()) if mask.size else 0.0, "saturation": 0.0, "warm": 0.0,
                "sharpness": 0.0}
    hsv = np.asarray(photo.convert("HSV"), dtype=np.float32)
    grey = np.asarray(photo.convert("L"), dtype=np.float32)
    lap = (-4 * grey[1:-1, 1:-1] + grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:])
    inner = mask[1:-1, 1:-1]
    return {
        "foreground": float(mask.mean()),
        "saturation": float(hsv[..., 1][mask].mean() / 255.0),
        "warm": float(((hsv[..., 0] >= 4) & (hsv[..., 0] <= 36) & (hsv[..., 1] > 38))[mask].mean()),
        "sharpness": float(lap[inner].var()) if inner.any() else 0.0,
    }


def validate_examples(examples: Sequence[Example]) -> None:
    keys = [key for key, _ in PAPER_ORDER]
    if [example.dataset for example in examples] != keys:
        raise ValueError(f"EXAMPLES must hold exactly one pair per dataset in paper order: {keys}")
    for example in examples:
        if not example.query_path or not example.gallery_path or example.query_path == example.gallery_path:
            raise ValueError(f"{example.dataset}: query and gallery paths must be distinct and non-empty")


# ── data access ───────────────────────────────────────────────────────────────
class DatasetContext:
    """Metadata, split rows and the probe run for one paper dataset."""

    def __init__(self, key: str) -> None:
        profiles = {profile.key: profile for profile in PAPER_PROFILES}
        self.key = key
        self.profile = profiles[key]
        self.run_dir = RUN_ROOT / RUNS[key]
        self.manifest = json.loads((self.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self._check_run()
        self.frame = pd.read_csv(self.profile.metadata, low_memory=False)
        split = self.frame[self.profile.split_col].astype(str)
        self.query_rows = np.flatnonzero((split == self.profile.query_value).to_numpy())
        self.database_rows = np.flatnonzero((split == self.profile.database_value).to_numpy())
        self.labels = self.frame[self.profile.identity_col].astype(str).to_numpy()
        self._attach_dates()
        self.view = None
        if self.profile.mask_col:
            from audit_image_quality import _FrameAdapter
            from reid.data.dataset_view import BenchmarkDatasetView
            self.view = BenchmarkDatasetView(_FrameAdapter(self.frame, self.profile.identity_col),
                                             label_col=self.profile.identity_col, no_background=True,
                                             mask_col=self.profile.mask_col)

    def _check_run(self) -> None:
        checkpoint = self.manifest.get("vismatch_checkpoint") or {}
        mode = checkpoint.get("resolved_component_mode") or checkpoint.get("component_mode")
        k = (self.manifest.get("timings") or {}).get("vismatch_candidate_k")
        if self.manifest.get("status") != "completed" or checkpoint.get("source") != "custom" \
                or mode != "matcher_only" or int(k or -1) != 50:
            raise ValueError(f"{self.run_dir} is not a completed fine-tuned LoMa k=50 run")

    def _attach_dates(self) -> None:
        if "date" in self.frame.columns or self.profile.dataset_name != "WildlifeReID-10k":
            return
        if not WILDLIFE_FULL_METADATA.is_file():
            return
        full = pd.read_csv(WILDLIFE_FULL_METADATA, usecols=["path", "date"], low_memory=False)
        dates = dict(zip(full["path"].astype(str), full["date"]))
        self.frame["date"] = [dates.get(raw_image_path(self.profile, row)) for row in self.frame.to_dict("records")]

    @property
    def checkpoint_path(self) -> str:
        """The exact checkpoint file the probe loaded, located by its recorded SHA-256."""
        if getattr(self, "_checkpoint", None) is None:
            from reid.utils.fingerprints import sha256_file
            components = self.manifest["vismatch_checkpoint"]["components"]
            if len(components) != 1:
                raise ValueError(f"{self.key}: expected one LoMa checkpoint component, got {len(components)}")
            recorded = Path(components[0]["path"])
            self.checkpoint_sha256 = str(components[0]["sha256"])
            self._checkpoint = resolve_recorded_checkpoint(recorded, self.checkpoint_sha256,
                                                           recorded.parents[2], sha256_file)
        return str(self._checkpoint)

    def probe_score(self, row_q: int, row_g: int) -> float:
        rows, cols, values = self.scores()
        q = int(np.flatnonzero(self.query_rows == row_q)[0])
        g = int(np.flatnonzero(self.database_rows == row_g)[0])
        hit = values[(rows == q) & (cols == g)]
        return float(hit[0]) if len(hit) else float("-inf")

    def scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.run_dir / "scores.npz") as data:
            shape = tuple(int(v) for v in data["shape"])
            if shape != (len(self.query_rows), len(self.database_rows)):
                raise ValueError(f"{self.key}: scores.npz shape {shape} does not match the metadata split")
            return data["rows"].copy(), data["cols"].copy(), data["values"].copy()

    def row_of_path(self, path: str) -> int:
        matches = np.flatnonzero(self.frame["path"].astype(str).to_numpy() == path)
        if len(matches) != 1:
            raise ValueError(f"{self.key}: metadata path {path!r} matched {len(matches)} rows")
        return int(matches[0])

    def model_input(self, row: int) -> Image.Image:
        from audit_image_quality import load_model_input
        image, _ = load_model_input(self.profile, self.frame, self.view, row)
        return Image.fromarray(image)

    def foreground(self, row: int) -> np.ndarray:
        """The probe's foreground mask (RLE when shipped, else the pre-masked threshold)."""
        from audit_image_quality import load_model_input
        return load_model_input(self.profile, self.frame, self.view, row)[1]

    def raw_photo(self, row: int) -> Image.Image:
        with Image.open(self.profile.root / raw_image_path(self.profile, self.frame.iloc[row])) as handle:
            return handle.convert("RGB")


# ── models ────────────────────────────────────────────────────────────────────
class Matcher:
    """The fine-tuned LoMa backend configured exactly like the probe run."""

    def __init__(self, context: DatasetContext, device: str) -> None:
        from omegaconf import OmegaConf
        from reid.methods.vismatch import VismatchMatcherBackend, _choose_vismatch_device
        from reid.methods.vismatch_profiles import default_matcher_threshold

        config = OmegaConf.load(context.run_dir / "config.snapshot.yaml").benchmark.methods.vismatch
        threshold = config.get("matcher_threshold")
        self.backend = VismatchMatcherBackend(
            "loma", _choose_vismatch_device(device), int(config.top_k),
            float(threshold) if threshold is not None else default_matcher_threshold("loma"),
            checkpoint_source="custom", checkpoint_path=context.checkpoint_path,
            checkpoint_components="matcher_only", loma_arch=str(config.loma_arch),
            resize_max=int(config.resize_max),
        )

        self.context = context
        snapshot = OmegaConf.load(context.run_dir / "config.snapshot.yaml")
        self.dataset_root = Path(str(snapshot.dataset.root))
        self.cache_dir = Path(str(config.cache_dir)) if config.get("cache_dir") else None
        self.resize_max, self.top_k = int(config.resize_max), int(config.top_k)
        self.cfg_tag = (context.manifest.get("metrics") or {}).get("vismatch_cache_fingerprint")
        self.cache_hits = 0

    def _features(self, row: int, split_name: str):
        """The probe's own cached features when present, else a fresh extraction.

        The probe extracted in batches of 8, which moves a few LoMa keypoints
        relative to single-image extraction, so the cache gives the probe's
        exact keypoints and descriptors. Matching still runs under LoMa's
        bfloat16 autocast, so recomputed scores agree with ``scores.npz`` only
        to about 1e-2 (measured 2026-09-30 on V100 and H100), not bit-exactly;
        rank 1 is taken from ``scores.npz``, never from the recompute.
        """
        from reid.methods.vismatch import _cache_key, _cache_path, _load_cached_feat, FEATURE_SCHEMA_VERSION
        from reid.methods.vismatch_preprocessing import to_rgb_float_tensor
        from reid.utils.fingerprints import sha256_file

        image_path = str(self.context.frame.iloc[row]["path"])
        if self.cache_dir is not None and self.cfg_tag:
            resolved = Path(image_path) if Path(image_path).is_absolute() else self.dataset_root / image_path
            key = _cache_key(image_path=image_path, image_content_hash=sha256_file(resolved), split_name=split_name,
                             resize_max=self.resize_max, top_k=self.top_k, cfg_tag=self.cfg_tag)
            path = _cache_path(self.cache_dir, key)
            if path.is_file():
                cached = _load_cached_feat(path)
                if cached.schema_version == FEATURE_SCHEMA_VERSION:
                    self.cache_hits += 1
                    return cached
        return self.backend.extract_frame(to_rgb_float_tensor(self.context.model_input(row)))

    def match(self, row_q: int, row_g: int):
        before = self.cache_hits
        left, right = self._features(row_q, "query"), self._features(row_g, "database")
        result = self.backend.match_features(left, right)
        return left, right, result, "probe_cache" if self.cache_hits - before == 2 else "extracted"


class Embedder:
    """MegaDescriptor-L embeddings with the probe's square-resize transform."""

    def __init__(self, device: str) -> None:
        import torch
        from models.model import get_model
        from reid.engine.probe_runner import build_transforms

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                                   ("cpu" if device == "auto" else device))
        backbone, _, mean, std, img_size, *_ = get_model("megadescriptor-l")
        self.model = backbone.to(self.device).eval()
        self.transform = build_transforms(mean, std, img_size)[1]

    def __call__(self, images: Iterable[Image.Image]) -> np.ndarray:
        batch = self.torch.stack([self.transform(image) for image in images]).to(self.device)
        with self.torch.no_grad():
            features = self.model(batch)
        return self.torch.nn.functional.normalize(features, dim=-1).cpu().numpy()


# ── drawing ───────────────────────────────────────────────────────────────────
PAPER_LINE_COLOR = "#00d9ff"  # cyan: contrasts with yellow spots, grass, water and pink IR frames
PADDING_THRESHOLD = 12  # max(RGB) at or below this counts as black source padding


def padding_box(array: np.ndarray, threshold: int = PADDING_THRESHOLD, fraction: float = 0.98) -> Tuple[int, int, int, int]:
    """(left, top, right, bottom) without the near-black bands stored in the source file.

    Only whole rows/columns reached from an edge are trimmed, and only when at least
    ``fraction`` of their pixels are dark, so dark night photos keep their content.
    """
    dark = np.asarray(array).max(axis=2) <= threshold
    rows, cols = dark.mean(axis=1) >= fraction, dark.mean(axis=0) >= fraction

    def span(flags: np.ndarray) -> Tuple[int, int]:
        start, stop = 0, len(flags)
        while start < stop and flags[start]:
            start += 1
        while stop > start and flags[stop - 1]:
            stop -= 1
        return (start, stop) if stop > start else (0, len(flags))

    left, right = span(cols)
    top, bottom = span(rows)
    return left, top, right, bottom


def spread_selection(p0: np.ndarray, p1: np.ndarray, confidences: np.ndarray, count: int,
                     min_distance: float) -> np.ndarray:
    """Up to ``count`` strong matches whose endpoints are at least ``min_distance`` apart.

    Greedy in the shared descending-confidence order (``stable_rank_1d``); a match is
    kept only when both its endpoints are far enough from every kept one. The spacing
    halves until ``count`` matches are found or it reaches zero (plain top-``count``).
    """
    from reid.evaluation.ranking import stable_rank_1d

    order = stable_rank_1d(np.asarray(confidences, dtype=np.float64))
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    distance = float(min_distance)
    while True:
        kept: List[int] = []
        for index in order:
            if len(kept) == count:
                break
            if kept:
                d0 = np.hypot(*(p0[kept] - p0[index]).T).min()
                d1 = np.hypot(*(p1[kept] - p1[index]).T).min()
                if min(d0, d1) < distance:
                    continue
            kept.append(int(index))
        if len(kept) >= min(count, len(order)) or distance <= 1e-9:
            return np.asarray(kept, dtype=np.int64)
        distance = distance / 2 if distance > 0.5 else 0.0


def dim_background(image: Image.Image, foreground: np.ndarray, brightness: float = 0.55,
                   saturation: float = 0.45, feather: float = 1.0) -> Image.Image:
    """Darken and desaturate the background, leaving the animal untouched.

    The mask is closed, hole-filled and feathered first, so thresholded masks of dark
    animals do not leave dimmed holes on the body.
    """
    from scipy import ndimage

    mask = np.asarray(foreground, dtype=bool)
    radius = max(2, round(0.004 * max(mask.shape)))
    mask = ndimage.binary_closing(mask, iterations=radius)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=radius)
    alpha = ndimage.gaussian_filter(mask.astype(np.float32), sigma=radius * feather)[..., None]
    rgb = np.asarray(image, dtype=np.float32)
    gray = rgb.mean(axis=2, keepdims=True)
    background = (gray + saturation * (rgb - gray)) * brightness
    return Image.fromarray(np.clip(alpha * rgb + (1 - alpha) * background, 0, 255).astype(np.uint8))


def crop_box_around(size: Sequence[int], points: np.ndarray, aspect: float) -> Tuple[int, int, int, int]:
    """Largest ``aspect`` (w/h) crop of an image of ``size`` (w, h), centred on the points.

    The crop is as large as the image allows, so it only trims the long side; its
    centre follows the keypoints' bounding-box centre and is clamped to the image.
    """
    width, height = int(size[0]), int(size[1])
    crop_w, crop_h = (width, round(width / aspect)) if width / height <= aspect else (round(height * aspect), height)
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cx, cy = ((points.min(0) + points.max(0)) / 2.0) if len(points) else (width / 2.0, height / 2.0)
    left = int(round(min(max(cx - crop_w / 2.0, 0), width - crop_w)))
    top = int(round(min(max(cy - crop_h / 2.0, 0), height - crop_h)))
    return left, top, left + crop_w, top + crop_h


def draw_pair(axis, query: Image.Image, gallery: Image.Image, kq: np.ndarray, kg: np.ndarray,
              confidences: np.ndarray, top_matches: int, height: int = 520, gutter: int = 14,
              aspect: Optional[float] = None) -> int:
    """Query | gallery at a common height with the strongest correspondences.

    With ``aspect`` each photo is cropped to that width/height ratio around its
    matched keypoints, so every pair has the same shape; correspondences with an
    endpoint outside a crop are dropped before the strongest are chosen.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt
    from reid.evaluation.ranking import stable_rank_1d

    kq, kg = np.asarray(kq, dtype=np.float64), np.asarray(kg, dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    if aspect is not None:
        boxes = [crop_box_around(image.size, points, aspect) for image, points in ((query, kq), (gallery, kg))]
        inside = np.ones(len(confidences), dtype=bool)
        for (l, t, r, b), points in zip(boxes, (kq, kg)):
            inside &= (points[:, 0] >= l) & (points[:, 0] < r) & (points[:, 1] >= t) & (points[:, 1] < b)
        query, gallery = query.crop(boxes[0]), gallery.crop(boxes[1])
        kq = kq[inside] - np.array(boxes[0][:2], dtype=np.float64)
        kg = kg[inside] - np.array(boxes[1][:2], dtype=np.float64)
        confidences = confidences[inside]
    scale_q, scale_g = height / query.height, height / gallery.height
    q_img = query.resize((round(query.width * scale_q), height), Image.Resampling.LANCZOS)
    g_img = gallery.resize((round(gallery.width * scale_g), height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (q_img.width + gutter + g_img.width, height), (255, 255, 255))
    canvas.paste(q_img, (0, 0))
    canvas.paste(g_img, (q_img.width + gutter, 0))
    axis.imshow(canvas)
    keep = stable_rank_1d(confidences)[:top_matches]
    p0 = kq[keep] * scale_q
    p1 = kg[keep] * scale_g + np.array([q_img.width + gutter, 0.0])
    colors = plt.get_cmap("viridis")(np.linspace(0.95, 0.25, len(keep))) if len(keep) else []
    axis.add_collection(LineCollection(np.stack([p0, p1], axis=1), colors=colors, linewidths=0.45, alpha=0.9))
    axis.scatter(np.r_[p0[:, 0], p1[:, 0]], np.r_[p0[:, 1], p1[:, 1]], s=1.2,
                 c=np.r_[colors, colors] if len(keep) else None, linewidths=0)
    axis.set_xlim(0, canvas.width)
    axis.set_ylim(height, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return len(keep)


def _prepare_photo(image: Image.Image, foreground: Optional[np.ndarray], points: np.ndarray,
                   aspect: float, height: int, dim: float):
    """Trim source padding, dim the background, crop to ``aspect`` around the points, resize.

    Returns the photo, the points in its pixel frame and a mask of the points kept.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if dim > 0 and foreground is not None:
        # A light, widely feathered dim: strong dimming exposes mask outlines.
        image = dim_background(image, foreground, brightness=1 - dim, saturation=1 - dim, feather=6.0)
    l, t, r, b = padding_box(np.asarray(image))
    inside = (points[:, 0] >= l) & (points[:, 0] < r) & (points[:, 1] >= t) & (points[:, 1] < b)
    image, points = image.crop((l, t, r, b)), points - np.array([l, t], dtype=np.float64)
    l, t, r, b = crop_box_around(image.size, points[inside], aspect)
    inside &= (points[:, 0] >= l) & (points[:, 0] < r) & (points[:, 1] >= t) & (points[:, 1] < b)
    image, points = image.crop((l, t, r, b)), points - np.array([l, t], dtype=np.float64)
    scale = height / image.height
    image = image.resize((round(image.width * scale), height), Image.Resampling.LANCZOS)
    return image, points * scale, inside


def _corner_tag(axis, x: float, y: float, text: str, ha: str, va: str, size: float):
    return axis.text(x, y, text, ha=ha, va=va, fontsize=size, color="white", zorder=4,
              bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.15", facecolor="black", alpha=0.55,
                        linewidth=0))


def points_outside_boxes(points: np.ndarray, boxes: Sequence[Tuple[float, float, float, float]],
                         margin: float = 0.0) -> np.ndarray:
    """True for points outside every (x0, y0, x1, y1) box grown by ``margin``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    outside = np.ones(len(points), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        outside &= ~((points[:, 0] >= x0 - margin) & (points[:, 0] <= x1 + margin)
                     & (points[:, 1] >= y0 - margin) & (points[:, 1] <= y1 + margin))
    return outside


def draw_paper_pair(axis, query: Image.Image, gallery: Image.Image, foregrounds, kq: np.ndarray,
                    kg: np.ndarray, confidences: np.ndarray, count: int, aspect: float,
                    match_count: int, dim: float, height: int = 420, gutter: int = 6,
                    spacing: float = 0.12, font_size: float = 6.0, side_tags: bool = True) -> int:
    """Paper panel: padded-trimmed, background-dimmed, fixed-aspect photos with spread lines.

    Corner tags are drawn first; matches with an endpoint under a tag are dropped
    before the ``count`` spread matches are chosen, so no dot hides behind a label.
    ``side_tags`` adds the "Query"/"Top-1 match" tags (the paper shows them once).
    """
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection

    q_img, q_pts, q_in = _prepare_photo(query, foregrounds[0], kq, aspect, height, dim)
    g_img, g_pts, g_in = _prepare_photo(gallery, foregrounds[1], kg, aspect, height, dim)
    inside = q_in & g_in
    q_pts, g_pts = q_pts[inside], g_pts[inside] + np.array([q_img.width + gutter, 0.0])
    confidences = np.asarray(confidences)[inside]
    canvas = Image.new("RGB", (q_img.width + gutter + g_img.width, height), (255, 255, 255))
    canvas.paste(q_img, (0, 0))
    canvas.paste(g_img, (q_img.width + gutter, 0))
    axis.imshow(canvas, interpolation="none")
    axis.set_xlim(0, canvas.width)
    axis.set_ylim(height, 0)
    axis.set_axis_off()

    pad = 0.035 * height
    tags = [_corner_tag(axis, canvas.width - pad, height - pad, f"{match_count} matches", "right", "bottom", font_size)]
    if side_tags:
        tags.append(_corner_tag(axis, pad, pad, "Query", "left", "top", font_size))
        tags.append(_corner_tag(axis, q_img.width + gutter + pad, pad, "Top-1 match", "left", "top", font_size))
    renderer = axis.figure.canvas.get_renderer()
    inverse = axis.transData.inverted()
    boxes = []
    for tag in tags:
        tag.draw(renderer)  # lays out the bbox patch so its extent is final
        (x0, y0), (x1, y1) = inverse.transform(tag.get_bbox_patch().get_window_extent(renderer).get_points())
        boxes.append((x0, y0, x1, y1))
    clear = points_outside_boxes(q_pts, boxes, 0.02 * height) & points_outside_boxes(g_pts, boxes, 0.02 * height)
    candidates = np.flatnonzero(clear)
    keep = candidates[spread_selection(q_pts[candidates], g_pts[candidates], confidences[candidates],
                                       count, spacing * height)]

    segments = np.stack([q_pts[keep], g_pts[keep]], axis=1)
    lines = LineCollection(segments, colors=PAPER_LINE_COLOR, linewidths=0.45, alpha=0.95, zorder=2)
    lines.set_path_effects([patheffects.Stroke(linewidth=1.35, foreground="black", alpha=0.7),
                            patheffects.Normal()])
    axis.add_collection(lines)
    ends = np.r_[q_pts[keep], g_pts[keep]]
    axis.scatter(ends[:, 0], ends[:, 1], s=2.2, facecolors=PAPER_LINE_COLOR, edgecolors="white",
                 linewidths=0.3, zorder=3)
    return len(keep)


def _style(paper: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if paper:
        # Times to match the CVPR body text; STIX (bundled with matplotlib) is the
        # Times-compatible fallback, embedded as TrueType.
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                             "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    else:
        plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
                             "pdf.fonttype": 42, "ps.fonttype": 42})


def _matched_pixels(context: DatasetContext, row_q: int, row_g: int, left, right, result):
    raw_q, raw_g = context.raw_photo(row_q), context.raw_photo(row_g)
    for raw, feature in ((raw_q, left), (raw_g, right)):
        if tuple(feature.original_image_size) != (raw.height, raw.width):
            raise ValueError(f"{context.key}: raw photo size {raw.size} differs from the model input "
                             f"{tuple(feature.original_image_size)}; keypoints cannot be mapped")
    kq = normalized_to_raw_pixels(result.matched_kpts0, (raw_q.height, raw_q.width))
    kg = normalized_to_raw_pixels(result.matched_kpts1, (raw_g.height, raw_g.width))
    return raw_q, raw_g, kq, kg


# ── commands ──────────────────────────────────────────────────────────────────
def command_candidates(args: argparse.Namespace) -> None:
    _style()
    import matplotlib.pyplot as plt

    out_dir = args.output_dir / "match_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(args.device)
    rng = np.random.default_rng(args.seed)
    for key, label in PAPER_ORDER:
        if args.dataset and key not in args.dataset:
            continue
        context = DatasetContext(key)
        pairs = correct_top1_pairs(*context.scores(), context.labels[context.query_rows],
                                   context.labels[context.database_rows])
        rows = [(int(context.query_rows[q]), int(context.database_rows[g]), s) for q, g, s in pairs]
        records = context.frame.to_dict("records")
        eligible = [(q, g, s) for q, g, s in rows if pair_rejection(records[q], records[g], None, 0) is None]
        if len(eligible) > args.pool:
            eligible = [eligible[i] for i in sorted(rng.choice(len(eligible), args.pool, replace=False))]
        scored, quality = [], {}
        for q, g, s in eligible:
            q_raw, g_raw = context.raw_photo(q), context.raw_photo(g)
            distance = hamming(dhash(q_raw), dhash(g_raw))
            if pair_rejection(records[q], records[g], distance, args.min_hash_distance) is not None:
                continue
            if args.min_foreground > 0 or args.min_saturation > 0 or args.min_warm > 0:
                pair_quality = [photo_quality(raw, context.foreground(row)) for raw, row in ((q_raw, q), (g_raw, g))]
                if min(item["foreground"] for item in pair_quality) < args.min_foreground \
                        or min(item["saturation"] for item in pair_quality) < args.min_saturation \
                        or min(item["warm"] for item in pair_quality) < args.min_warm:
                    continue
                quality[(q, g)] = pair_quality
            q_img, g_img = context.model_input(q), context.model_input(g)
            cosine = float(np.dot(*embedder([q_img, g_img])))
            scored.append((cosine, q, g, s, distance))
        scored.sort()
        matcher = Matcher(context, args.device)
        kept = []
        for cosine, q, g, s, distance in scored[:args.rematch]:
            left, right, result, source = matcher.match(q, g)
            if result.match_count < args.min_matches:
                continue
            kept.append({
                "dataset": key, "query_row": q, "gallery_row": g,
                "query_path": records[q]["path"], "gallery_path": records[g]["path"],
                "probe_score": s, "recomputed_score": result.score,
                "features": source, "match_count": result.match_count,
                "cosine": cosine, "dhash_distance": distance,
                "query_date": records[q].get("date"), "gallery_date": records[g].get("date"),
                **{f"{side}_{name}": value for side, item in zip(("query", "gallery"), quality.get((q, g), ()))
                   for name, value in item.items()},
                "_features": (left, right, result),
            })
        table = pd.DataFrame([{k: v for k, v in item.items() if k != "_features"} for item in kept])
        stem = f"{key}_{args.tag}" if args.tag else key
        table.to_csv(out_dir / f"{stem}.csv", index=False)
        print(f"[match-candidates] {label}: {len(pairs)} correct top-1, {len(eligible)} sampled/eligible, "
              f"{len(scored)} passed filters, {len(kept)} with >= {args.min_matches} matches -> {out_dir / f'{stem}.csv'}")
        sheet = kept[:args.sheet]
        if not sheet:
            continue
        figure, axes = plt.subplots(len(sheet), 1, figsize=(6.0, 2.2 * len(sheet)), squeeze=False)
        for axis, item in zip(axes[:, 0], sheet):
            left, right, result = item["_features"]
            raw_q, raw_g, kq, kg = _matched_pixels(context, item["query_row"], item["gallery_row"], left, right, result)
            draw_pair(axis, raw_q, raw_g, kq, kg, result.confidences, args.top_matches)
            axis.set_title(f"{label}: cos={item['cosine']:.2f} matches={item['match_count']} "
                           f"q={item['query_path']} | g={item['gallery_path']}", fontsize=5)
        figure.tight_layout()
        figure.savefig(out_dir / f"{stem}_sheet.png", dpi=200)
        plt.close(figure)


def web_export_pair(out_dir: Path, key: str, label: str, raw_q: Image.Image, raw_g: Image.Image,
                    kq: np.ndarray, kg: np.ndarray, confidences: np.ndarray, match_count: int, score: float,
                    query_path: str, gallery_path: str, long_side: int) -> Dict[str, Any]:
    """Write web-sized raw photos and every correspondence for the project-page match viewer.

    Keypoints are scaled from raw-photo pixels to the exported photo's pixels, so the
    viewer draws them without knowing the raw size. Paths written are dataset-relative.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entry: Dict[str, Any] = {"dataset": key, "label": label, "match_count": int(match_count), "score": float(score),
                             "query_path": query_path, "gallery_path": gallery_path, "images": {}, "points": {}}
    for side, raw, points in (("query", raw_q, kq), ("gallery", raw_g, kg)):
        scale = min(1.0, long_side / max(raw.width, raw.height))
        size = (max(1, round(raw.width * scale)), max(1, round(raw.height * scale)))
        photo = raw.convert("RGB").resize(size, Image.LANCZOS) if scale < 1.0 else raw.convert("RGB")
        name = f"{key}_{side}.jpg"
        photo.save(out_dir / name, format="JPEG", quality=86, optimize=True, progressive=True)
        entry["images"][side] = {"file": name, "width": size[0], "height": size[1]}
        scaled = np.asarray(points, dtype=np.float64).reshape(-1, 2) * (size[0] / raw.width, size[1] / raw.height)
        entry["points"][side] = [[round(float(x), 1), round(float(y), 1)] for x, y in scaled]
    order = np.argsort(-np.asarray(confidences, dtype=np.float64), kind="stable")
    entry["confidence"] = [round(float(c), 4) for c in np.asarray(confidences, dtype=np.float64)]
    entry["order_by_confidence"] = [int(i) for i in order]
    return entry


def command_render(args: argparse.Namespace) -> None:
    validate_examples(EXAMPLES)
    _style(paper=True)
    import matplotlib.pyplot as plt
    from reid.utils.fingerprints import sha256_file

    # Rows of args.columns cells; a short last row is centred. Every cell is two
    # args.aspect photos plus a gutter, so all cells share one size.
    # No side margin (spans \\linewidth); the gap between panels is about three
    # times the gutter inside a pair, so each pair reads as one unit.
    columns, margin, gap = args.columns, 0.0, 0.07
    title = args.title_size / 72 + 0.06  # title band: text height plus padding, inches
    cell_w = (args.width - 2 * margin - (columns - 1) * gap) / columns
    cell_h = cell_w / (2 * args.aspect + 0.02)
    rows = -(-len(EXAMPLES) // columns)
    fig_h = rows * (cell_h + title) + 0.01
    figure = plt.figure(figsize=(args.width, fig_h))
    axes = []
    for index in range(len(EXAMPLES)):
        row, col = divmod(index, columns)
        in_row = min(columns, len(EXAMPLES) - row * columns)
        offset = (args.width - in_row * cell_w - (in_row - 1) * gap) / 2
        x = offset + col * (cell_w + gap)
        y = fig_h - (row + 1) * (cell_h + title) + 0.0
        axes.append(figure.add_axes((x / args.width, y / fig_h, cell_w / args.width, cell_h / fig_h)))
    sidecar = []
    web_entries = []
    for index, (axis, example, (key, label)) in enumerate(zip(axes, EXAMPLES, PAPER_ORDER)):
        context = DatasetContext(key)
        row_q, row_g = context.row_of_path(example.query_path), context.row_of_path(example.gallery_path)
        if context.labels[row_q] != context.labels[row_g]:
            raise ValueError(f"{key}: the pinned pair is not the same individual")
        left, right, result, source = Matcher(context, args.device).match(row_q, row_g)
        probe_score = context.probe_score(row_q, row_g)
        if abs(result.score - probe_score) > 1e-4:
            print(f"[match-examples] WARNING {key}: recomputed score {result.score:.6f} "
                  f"differs from the probe's {probe_score:.6f} ({source} features)")
        raw_q, raw_g, kq, kg = _matched_pixels(context, row_q, row_g, left, right, result)
        foregrounds = (context.foreground(row_q), context.foreground(row_g))
        drawn = draw_paper_pair(axis, raw_q, raw_g, foregrounds, kq, kg, result.confidences,
                                args.paper_matches, args.aspect, int(result.match_count),
                                dim=args.dim if example.dim is None else example.dim,
                                height=args.photo_height, font_size=args.tag_size, side_tags=index == 0)
        axis.set_title(f"({chr(ord('a') + index)}) {label}", fontsize=args.title_size, pad=1.5)
        sidecar.append({
            "dataset": key, "label": label, "run": RUNS[key], "checkpoint": context.checkpoint_path,
            "checkpoint_sha256": sha256_file(Path(context.checkpoint_path)),
            "features": source, "probe_score": probe_score,
            "query_path": example.query_path, "gallery_path": example.gallery_path,
            "match_count": int(result.match_count), "drawn_matches": int(drawn),
            "score": float(result.score), "note": example.note,
            "background_dim": float(args.dim if example.dim is None else example.dim), "aspect": args.aspect,
        })
        if args.web_export is not None:
            web_entries.append(web_export_pair(args.web_export, key, label, raw_q, raw_g, kq, kg,
                                               result.confidences, int(result.match_count), float(result.score),
                                               example.query_path, example.gallery_path, args.web_long_side))
    if args.web_export is not None:
        args.web_export.mkdir(parents=True, exist_ok=True)
        payload = {"generated_by": "scripts/plot_match_examples.py render --web-export", "method": "LoMa + WildMatch",
                   "candidate_k": 50, "coordinates": "pixels on the exported photos (origin top-left)",
                   "examples": web_entries}
        (args.web_export / "match_examples.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"[match-examples] wrote web assets to {args.web_export}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt, dpi in (("pdf", 300), ("png", 300)):
        figure.savefig(args.output_dir / f"match_examples.{fmt}", dpi=dpi, facecolor="white",
                       metadata={"Creator": "scripts/plot_match_examples.py"})
    plt.close(figure)
    (args.output_dir / "match_examples.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"[match-examples] wrote {args.output_dir / 'match_examples.pdf'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-matches", type=int, default=40, help="Strongest correspondences drawn per pair")
    sub = parser.add_subparsers(dest="command", required=True)
    cand = sub.add_parser("candidates", help="Shortlist hard, correct, non-duplicate pairs per dataset")
    cand.add_argument("--dataset", action="append", choices=[key for key, _ in PAPER_ORDER])
    cand.add_argument("--pool", type=int, default=300, help="Eligible correct pairs sampled per dataset")
    cand.add_argument("--rematch", type=int, default=20, help="Hardest pairs re-matched with LoMa")
    cand.add_argument("--min-matches", type=int, default=40)
    cand.add_argument("--min-hash-distance", type=int, default=6, help="dHash bits; below = near-duplicate")
    cand.add_argument("--sheet", type=int, default=6, help="Pairs drawn per contact sheet")
    cand.add_argument("--seed", type=int, default=0)
    cand.add_argument("--min-foreground", type=float, default=0.0,
                      help="Minimum mask share of each raw photo (a large, visible animal)")
    cand.add_argument("--min-saturation", type=float, default=0.0,
                      help="Minimum mean HSV saturation on the animal (rejects greyscale/infrared frames)")
    cand.add_argument("--min-warm", type=float, default=0.0,
                      help="Minimum share of warm (orange-brown) animal pixels; rejects pink infrared frames")
    cand.add_argument("--tag", default="", help="Suffix for the CSV and sheet names, so earlier sheets are kept")
    render = sub.add_parser("render", help="Draw the pinned EXAMPLES figure")
    render.add_argument("--paper-matches", type=int, default=10,
                        help="Spatially spread high-confidence correspondences drawn per pair")
    render.add_argument("--dim", type=float, default=0.2,
                        help="Background dimming strength in [0, 1) using the probe mask, widely feathered; "
                             "0 disables it. Strong values (about 0.45) expose provider/SAM3 mask outlines")
    render.add_argument("--photo-height", type=int, default=260,
                        help="Embedded pixel height per photo (260 is about 300 dpi at three panels per row)")
    render.add_argument("--columns", type=int, default=3, help="Panels per row; a short last row is centred")
    render.add_argument("--tag-size", type=float, default=6.5, help="Corner tag font size in points")
    render.add_argument("--title-size", type=float, default=9.0, help="Panel title font size in points")
    render.add_argument("--aspect", type=float, default=4 / 3,
                        help="Width/height of each photo crop, shared by all panels")
    render.add_argument("--width", type=float, default=6.875, help="Figure width in inches (CVPR full text width)")
    render.add_argument("--web-export", type=Path, default=None,
                        help="Also write web-sized raw photos and all correspondences for the project-page "
                             "match viewer into this directory (docs/assets/match for the page)")
    render.add_argument("--web-long-side", type=int, default=1000, help="Long side of the exported web photos")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    {"candidates": command_candidates, "render": command_render}[args.command](args)


if __name__ == "__main__":
    main()
