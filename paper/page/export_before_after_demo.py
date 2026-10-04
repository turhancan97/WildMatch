#!/usr/bin/env python3
"""Export the project-page "before vs after fine-tuning" demo: default LoMa against
LoMa + WildMatch on the same photo pair.

For each paper dataset the match-figure pair (query and its correct top-1 gallery photo,
picked by the user) is matched twice on the background-removed inputs the pipeline uses,
once with the default LoMa matcher and once with the dataset's fine-tuned matcher, and the
correspondences are drawn on the original photos with background, as in the paper's
qualitative figure. Both results are exported for every pair so the page can toggle
between them: image score, match count, every correspondence in exported-photo pixels,
confidences and confidence order.

Matching mirrors the probe configuration (LoMa-B, 512 px long side, up to 512 keypoints,
default mutual-match threshold, matcher-only fine-tuned checkpoint, verified by SHA-256
against the match-figure sidecar). Features are extracted per image here, which moves a
few LoMa keypoints relative to the probe's batched extraction, so scores can differ from
the probe's by about 1e-2; the fine-tuned score of each pair is printed next to the
probe's recorded score for reference. Needs a GPU and the ex-reid environment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # run as a script: make `paper` importable
    sys.path.insert(0, str(REPO_ROOT))
from paper.page.export_synthetic_demo import decode_rle, normalized_to_pixels  # noqa: E402
from wildmatch.reporting.paper_datasets import PAPER_PROFILES  # noqa: E402

SIDECAR = REPO_ROOT / "reports" / "figures" / "match_examples.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "before_after"
WEB_LONG_SIDE = 1000
PROFILE_BY_KEY = {p.key: p for p in PAPER_PROFILES}
ATTRIBUTION = (
    "Photographs from the paper's datasets: CzechLynx (Picek et al.), WildlifeReID-10k (Adam et al.) "
    "and SalamanderID2025 (AnimalCLEF 2025, shown with the dataset team's permission)."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_row(profile, path: str) -> Dict[str, str]:
    csv.field_size_limit(1 << 30)
    with Path(profile.metadata).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("path") == path:
                return row
    raise FileNotFoundError(f"{profile.key}: {path} not in {profile.metadata}")


def load_images(profile, path: str) -> tuple:
    """(raw photo with background, background-removed model input) for one metadata path."""
    row = metadata_row(profile, path)
    if profile.mask_col:  # CzechLynx: raw photo + RLE mask
        raw = Image.open(profile.root / path).convert("RGB")
        mask = decode_rle(row[profile.mask_col])
        model_input = Image.fromarray(np.asarray(raw) * mask[..., None].astype(np.uint8), "RGB")
    elif row.get("original_path"):  # SalamanderID2025
        raw = Image.open(profile.root / row["original_path"]).convert("RGB")
        model_input = Image.open(profile.root / path).convert("RGB")
    else:  # WildlifeReID-10k pre-masked tree
        raw_rel = "images/" + path[len("masked_images/") :] if path.startswith("masked_images/") else path
        raw = Image.open(profile.root / raw_rel).convert("RGB")
        model_input = Image.open(profile.root / path).convert("RGB")
    if raw.size != model_input.size:
        raise ValueError(f"{profile.key}: raw {raw.size} and masked {model_input.size} sizes differ for {path}")
    return raw, model_input


def web_copy(image: Image.Image, out_path: Path, long_side: int = WEB_LONG_SIDE) -> Dict[str, int]:
    scale = min(1.0, long_side / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    (image.resize(size, Image.LANCZOS) if scale < 1.0 else image).save(
        out_path, format="JPEG", quality=88, optimize=True, progressive=True
    )
    return {"file": out_path.name, "width": size[0], "height": size[1]}


def match_record(result, size_q: Sequence[int], size_g: Sequence[int]) -> Dict[str, Any]:
    kq = normalized_to_pixels(result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2)), size_q)
    kg = normalized_to_pixels(result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2)), size_g)
    conf = np.asarray(result.confidences if result.confidences is not None else [], dtype=np.float64).reshape(-1)
    return {
        "score": float(result.score),
        "match_count": int(result.match_count),
        "points": {
            "query": [[round(float(x), 1), round(float(y), 1)] for x, y in kq],
            "gallery": [[round(float(x), 1), round(float(y), 1)] for x, y in kg],
        },
        "confidence": [round(float(c), 4) for c in conf],
        "order_by_confidence": [int(i) for i in np.argsort(-conf, kind="stable")],
    }


def build_payload(pairs: List[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_before_after_demo.py",
        "attribution": ATTRIBUTION,
        "inputs": "matches computed on background-removed inputs (as in the paper); drawn on the original photos",
        "matchers": {
            "default": "LoMa (default checkpoint)",
            "finetuned": "LoMa + WildMatch (matching module fine-tuned on this dataset)",
        },
        "settings": settings,
        "pairs": pairs,
    }
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"demo payload contains a private path fragment {needle!r}")
    return payload


def _backend(device: str, source: str, checkpoint: Optional[Path], settings: Dict[str, Any]):
    from wildmatch.matchers.vismatch import VismatchMatcherBackend, _choose_vismatch_device

    kwargs = dict(checkpoint_source=source, loma_arch=settings["loma_arch"], resize_max=settings["resize_max"])
    if checkpoint is not None:
        kwargs.update(checkpoint_path=str(checkpoint), checkpoint_components="matcher_only")
    return VismatchMatcherBackend(
        "loma", _choose_vismatch_device(device), settings["top_k"], settings["threshold"], **kwargs
    )


def export(out_dir: Path, device: str, sidecar: Path = SIDECAR, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    from wildmatch.matchers.vismatch_preprocessing import to_rgb_float_tensor
    from wildmatch.matchers.vismatch_profiles import default_matcher_threshold

    settings = {
        "loma_arch": "LoMa-B",
        "resize_max": 512,
        "top_k": 512,
        "threshold": default_matcher_threshold("loma"),
        "checkpoint_components": "matcher_only",
    }
    examples = json.loads(sidecar.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    default_backend = _backend(device, "default", None, settings)
    pairs: List[Dict[str, Any]] = []
    for example in examples:
        if only and example["dataset"] not in only:
            continue
        profile = PROFILE_BY_KEY[example["dataset"]]
        checkpoint = Path(example["checkpoint"])
        if sha256_file(checkpoint) != example["checkpoint_sha256"]:
            raise ValueError(f"{example['dataset']}: checkpoint {checkpoint} does not match the sidecar SHA-256")
        tuned_backend = _backend(device, "custom", checkpoint, settings)
        raw_q, in_q = load_images(profile, example["query_path"])
        raw_g, in_g = load_images(profile, example["gallery_path"])
        web_q = web_copy(raw_q, out_dir / f"{example['dataset']}_query.jpg")
        web_g = web_copy(raw_g, out_dir / f"{example['dataset']}_gallery.jpg")
        size_q, size_g = (web_q["width"], web_q["height"]), (web_g["width"], web_g["height"])
        results: Dict[str, Any] = {}
        for name, backend in (("default", default_backend), ("finetuned", tuned_backend)):
            fq, fg = backend.extract_frame(to_rgb_float_tensor(in_q)), backend.extract_frame(to_rgb_float_tensor(in_g))
            results[name] = match_record(backend.match_features(fq, fg), size_q, size_g)
        pairs.append(
            {
                "dataset": example["dataset"],
                "label": example["label"],
                "query": {"image": web_q, "source": example["query_path"]},
                "gallery": {"image": web_g, "source": example["gallery_path"]},
                "checkpoint_sha256": example["checkpoint_sha256"],
                "probe_score": example.get("probe_score"),
                "results": results,
            }
        )
        print(
            f"[before-after] {example['label']:14s} default {results['default']['score']:.3f} ({results['default']['match_count']} m)"
            f"  fine-tuned {results['finetuned']['score']:.3f} ({results['finetuned']['match_count']} m)"
            f"  probe recorded {example.get('probe_score', float('nan')):.3f}",
            flush=True,
        )
    payload = build_payload(pairs, settings)
    (out_dir / "before_after.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[before-after] wrote {out_dir / 'before_after.json'} ({len(pairs)} pairs)")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sidecar", type=Path, default=SIDECAR, help="match_examples.json with per-dataset checkpoints"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--only", nargs="*", default=None, help="dataset keys to export (default: all pairs)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.sidecar.is_file():
        print(f"error: sidecar not found: {args.sidecar}", file=sys.stderr)
        return 2
    export(args.out, args.device, args.sidecar, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
