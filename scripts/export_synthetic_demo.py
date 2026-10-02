#!/usr/bin/env python3
"""Export the project-page keypoint-match demo on CzechLynx synthetic renders.

The demo shows one synthetic query render against six synthetic gallery renders,
ranked by the fine-tuned LoMa matcher (LoMa + WildMatch), with every mutual-nearest
correspondence and its confidence, so visitors can explore real matcher output on
openly licensed images (CzechLynx synthetic subset, Picek et al., Zenodo record
17592004, CC BY 4.0). Synthetic renders are not the paper's test data; the page labels
the demo as such.

Inputs are the paper's teaser crops (``paper/figures/teaser/<tag>_full.png`` and
``prep.json`` in the paper repository): full-body renders with the dataset mask applied
and the background replaced by one flat light colour. For inference that flat colour is
set to black, which reproduces the paper's masked input (pixels outside the mask are
blacked out, nothing is cropped); the displayed photo keeps the light background.
Matching is configured exactly like the paper's CzechLynx fine-tuned LoMa probe run
(``config.snapshot.yaml`` of the run: LoMa-B, 512 px long side, up to 512 keypoints,
default mutual-match threshold, ``matcher_only`` checkpoint), and the checkpoint file's
SHA-256 is recorded. Keypoints are mapped from LoMa's normalized coordinates to the
render's pixels with the Vismatch half-pixel convention.

Requires a GPU with the ex-reid environment and the fine-tuned checkpoint; run it
yourself and commit the output under ``docs/assets/demo/synthetic/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEASER_DIR = REPO_ROOT.parent / "ECIR-Animal-ReID-Paper" / "paper" / "figures" / "teaser"
DEFAULT_RUN_DIR = (REPO_ROOT / "experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/megadescriptor-l"
                   "/vismatch/loma/20260920T122915Z_0015f14a")
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "synthetic"
TEASER_BACKGROUND = (243, 241, 236)  # the flat colour teaser_prep.py put behind the animal
ATTRIBUTION = ("Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), "
               "Zenodo record 17592004, CC BY 4.0.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_of(src: str) -> str:
    """``.../synthetic_lynx_173/09954_synthetic_lynx_173.jpg`` -> ``lynx_173``."""
    match = re.search(r"synthetic_lynx_(\d+)", src)
    if not match:
        raise ValueError(f"cannot read the individual from {src!r}")
    return f"lynx_{match.group(1)}"


def mask_background(render: Image.Image, background: Sequence[int] = TEASER_BACKGROUND, tolerance: int = 6) -> Image.Image:
    """Model input: the teaser's flat background colour becomes black, the animal stays.

    The teaser crops already carry the dataset mask (everything outside the animal is
    one flat colour), so thresholding that colour recovers the masked input the paper
    fed to the matcher without re-downloading the renders and masks.
    """
    array = np.asarray(render.convert("RGB")).astype(np.int16)
    is_background = (np.abs(array - np.asarray(background, dtype=np.int16)) <= tolerance).all(axis=2)
    out = array.astype(np.uint8).copy()
    out[is_background] = 0
    return Image.fromarray(out, "RGB")


def normalized_to_pixels(points: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """LoMa normalized [-1, 1] keypoints -> pixel coordinates on a (width, height) image."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = int(size[0]), int(size[1])
    out = np.empty_like(points)
    out[:, 0] = width * (points[:, 0] + 1.0) / 2.0 - 0.5
    out[:, 1] = height * (points[:, 1] + 1.0) / 2.0 - 0.5
    return out


def candidate_record(tag: str, src: str, query_identity: str, size: Sequence[int], result: Any,
                     kq: np.ndarray, kg: np.ndarray) -> Dict[str, Any]:
    identity = identity_of(src)
    confidences = np.asarray(result.confidences if result.confidences is not None else [], dtype=np.float64).reshape(-1)
    order = np.argsort(-confidences, kind="stable")
    return {
        "tag": tag, "identity": identity, "same_individual": identity == query_identity,
        "image": {"file": f"{tag}.jpg", "width": int(size[0]), "height": int(size[1])},
        "score": float(result.score), "match_count": int(result.match_count),
        "points": {"query": [[round(float(x), 1), round(float(y), 1)] for x, y in kq],
                   "gallery": [[round(float(x), 1), round(float(y), 1)] for x, y in kg]},
        "confidence": [round(float(c), 4) for c in confidences],
        "order_by_confidence": [int(i) for i in order],
    }


def rank_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Descending score; ties keep input order, like the probe's stable ranking."""
    ranked = sorted(candidates, key=lambda c: -c["score"])
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def build_payload(query: Dict[str, Any], candidates: List[Dict[str, Any]], model: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "scripts/export_synthetic_demo.py",
        "synthetic": True,
        "attribution": ATTRIBUTION,
        "method": "LoMa + WildMatch (matching module fine-tuned on CzechLynx, closed split)",
        "coordinates": "pixels on the exported renders (origin top-left)",
        "score_definition": "sum of match confidences over mutual-nearest matches above the threshold, divided by min(n_query, n_gallery)",
        "model": model,
        "query": query,
        "candidates": rank_candidates(candidates),
    }
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"demo payload contains a private path fragment {needle!r}")
    return payload


def _load_backend(run_dir: Path, checkpoint: Optional[Path], device: str):
    from omegaconf import OmegaConf
    from reid.methods.vismatch import VismatchMatcherBackend, _choose_vismatch_device
    from reid.methods.vismatch_profiles import default_matcher_threshold

    snapshot = OmegaConf.load(run_dir / "config.snapshot.yaml")
    config = snapshot.benchmark.methods.vismatch
    if str(config.matcher) != "loma" or str(config.checkpoint_components) != "matcher_only":
        raise ValueError(f"{run_dir}: expected a matcher-only LoMa run")
    path = checkpoint or Path(str(config.checkpoint_path))
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    threshold = config.get("matcher_threshold")
    threshold = float(threshold) if threshold is not None else default_matcher_threshold("loma")
    backend = VismatchMatcherBackend(
        "loma", _choose_vismatch_device(device), int(config.top_k), threshold,
        checkpoint_source="custom", checkpoint_path=str(path), checkpoint_components="matcher_only",
        loma_arch=str(config.loma_arch), resize_max=int(config.resize_max),
    )
    model = {"matcher": "loma", "loma_arch": str(config.loma_arch), "resize_max": int(config.resize_max),
             "top_k": int(config.top_k), "threshold": threshold, "checkpoint_components": "matcher_only",
             "checkpoint_sha256": sha256_file(path), "checkpoint_epoch": path.parent.name,
             "configured_from_run": run_dir.name}
    return backend, model


def export(teaser_dir: Path, run_dir: Path, out_dir: Path, checkpoint: Optional[Path], device: str,
           query_tag: str = "L173_0") -> Dict[str, Any]:
    from reid.methods.vismatch_preprocessing import to_rgb_float_tensor

    prep = json.loads((teaser_dir / "prep.json").read_text(encoding="utf-8"))
    if query_tag not in prep:
        raise ValueError(f"query tag {query_tag!r} not in prep.json")
    backend, model = _load_backend(run_dir, checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    renders: Dict[str, Image.Image] = {}
    features: Dict[str, Any] = {}
    for tag in prep:
        render = Image.open(teaser_dir / f"{tag}_full.png").convert("RGB")
        renders[tag] = render
        render.save(out_dir / f"{tag}.jpg", format="JPEG", quality=90, optimize=True, progressive=True)
        features[tag] = backend.extract_frame(to_rgb_float_tensor(mask_background(render)))
        print(f"[synthetic-demo] {tag}: {len(features[tag].keypoints)} keypoints", flush=True)

    query_identity = identity_of(prep[query_tag]["src"])
    query = {"tag": query_tag, "identity": query_identity, "source": prep[query_tag]["src"],
             "image": {"file": f"{query_tag}.jpg", "width": renders[query_tag].width, "height": renders[query_tag].height}}
    candidates: List[Dict[str, Any]] = []
    for tag, entry in prep.items():
        if tag == query_tag:
            continue
        result = backend.match_features(features[query_tag], features[tag])
        kq = normalized_to_pixels(result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2)), renders[query_tag].size)
        kg = normalized_to_pixels(result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2)), renders[tag].size)
        record = candidate_record(tag, entry["src"], query_identity, renders[tag].size, result, kq, kg)
        record["source"] = entry["src"]
        candidates.append(record)
        print(f"[synthetic-demo] {query_tag} vs {tag} ({record['identity']}): score {record['score']:.4f}, "
              f"{record['match_count']} matches", flush=True)

    payload = build_payload(query, candidates, model)
    (out_dir / "synthetic_demo.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[synthetic-demo] wrote {out_dir / 'synthetic_demo.json'}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teaser-dir", type=Path, default=DEFAULT_TEASER_DIR,
                        help="paper/figures/teaser in the paper repository clone")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="fine-tuned LoMa probe run whose config.snapshot.yaml configures the matcher")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the checkpoint path recorded in the run snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query", default="L173_0", help="teaser tag used as the query render")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not (args.teaser_dir / "prep.json").is_file():
        print(f"error: {args.teaser_dir} has no prep.json", file=sys.stderr)
        return 2
    export(args.teaser_dir, args.run_dir, args.out, args.checkpoint, args.device, args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
