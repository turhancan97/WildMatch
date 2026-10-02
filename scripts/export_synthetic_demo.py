#!/usr/bin/env python3
"""Export the project-page keypoint-match demo on CzechLynx synthetic renders.

The demo shows one synthetic query render against six synthetic gallery renders,
ranked by the fine-tuned LoMa matcher (LoMa + WildMatch), with every mutual-nearest
correspondence and its confidence, so visitors can explore real matcher output on
openly licensed images (CzechLynx synthetic subset, Picek et al., Zenodo record
17592004, CC BY 4.0). Synthetic renders are not the paper's test data; the page labels
the demo as such.

Inputs are the original renders and the dataset's own COCO-RLE masks from the CzechLynx
synthetic metadata (``CzechLynxDataset-Metadata-Synthetic.csv`` under the dataset root).
The model input is the render with every pixel outside the mask set to black, exactly
as the probe's ``BenchmarkDatasetView`` does for CzechLynx (``no_background=true``);
nothing is cropped, so LoMa's normalized keypoints map straight onto the raw render,
which is what the page displays (resized for the web). Matching is configured like the
paper's CzechLynx fine-tuned LoMa probe run (``config.snapshot.yaml`` of the run:
LoMa-B, 512 px long side, up to 512 keypoints, default mutual-match threshold,
``matcher_only`` checkpoint), and the checkpoint file's SHA-256 is recorded.

The seven renders are the paper's teaser selection (query ``L173_0`` in snow and six
forest renders, one of them lynx 173 again). Needs a GPU and the ex-reid environment.
"""

from __future__ import annotations

import argparse
import csv
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
if str(REPO_ROOT) not in sys.path:  # allow running as a plain script
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = Path("/shared/sets/datasets/vision/czechlynx/CzechLynx_v2")
DEFAULT_METADATA = "CzechLynxDataset-Metadata-Synthetic.csv"
DEFAULT_RUN_DIR = (REPO_ROOT / "experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/megadescriptor-l"
                   "/vismatch/loma/20260920T122915Z_0015f14a")
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "synthetic"
WEB_LONG_SIDE = 1000
ATTRIBUTION = ("Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), "
               "Zenodo record 17592004, CC BY 4.0.")

# The paper's teaser renders (results/teaser_prep.py in the paper repository): tag ->
# path relative to the dataset root. L173_0 is the snow query; L173_1 is the same
# individual in the forest.
RENDERS: Dict[str, str] = {
    "L173_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_173/09954_synthetic_lynx_173.jpg",
    "L173_1": "CzechLynx_Synthetic/synthetic/synthetic_lynx_173/06420_synthetic_lynx_173.jpg",
    "L129_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_129/05121_synthetic_lynx_129.jpg",
    "L242_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_242/08305_synthetic_lynx_242.jpg",
    "L88_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_88/04055_synthetic_lynx_88.jpg",
    "L79_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_79/00519_synthetic_lynx_79.jpg",
    "L138_0": "CzechLynx_Synthetic/synthetic/synthetic_lynx_138/09291_synthetic_lynx_138.jpg",
}
DEFAULT_QUERY = "L173_0"


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


def decode_rle(mask_json: str) -> np.ndarray:
    """COCO compressed RLE (as stored in the metadata) -> 2-D uint8 mask."""
    from pycocotools import mask as mask_utils

    data = json.loads(mask_json)
    mask = mask_utils.decode(data).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask.max(axis=-1)
    return mask


def apply_mask(render: Image.Image, mask: np.ndarray) -> Image.Image:
    """Model input: pixels outside the mask become black, as in ``BenchmarkDatasetView``."""
    array = np.asarray(render.convert("RGB"))
    if array.shape[:2] != mask.shape:
        raise ValueError(f"mask {mask.shape} does not match render {array.shape[:2]}")
    return Image.fromarray(array * mask[..., None].astype(np.uint8), "RGB")


def normalized_to_pixels(points: np.ndarray, size: Sequence[int]) -> np.ndarray:
    """LoMa normalized [-1, 1] keypoints -> pixel coordinates on a (width, height) image."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = int(size[0]), int(size[1])
    out = np.empty_like(points)
    out[:, 0] = width * (points[:, 0] + 1.0) / 2.0 - 0.5
    out[:, 1] = height * (points[:, 1] + 1.0) / 2.0 - 0.5
    return out


def web_copy(render: Image.Image, out_path: Path, long_side: int = WEB_LONG_SIDE) -> Dict[str, int]:
    scale = min(1.0, long_side / max(render.size))
    size = (max(1, round(render.width * scale)), max(1, round(render.height * scale)))
    photo = render.convert("RGB").resize(size, Image.LANCZOS) if scale < 1.0 else render.convert("RGB")
    photo.save(out_path, format="JPEG", quality=88, optimize=True, progressive=True)
    return {"file": out_path.name, "width": size[0], "height": size[1]}


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
        "inputs": "renders with the dataset's own masks applied (background set to black), as in the paper",
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


def load_rows(root: Path, metadata: str, paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    csv.field_size_limit(1 << 30)
    wanted = set(paths)
    found: Dict[str, Dict[str, str]] = {}
    with (root / metadata).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("path") in wanted:
                found[row["path"]] = row
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"renders missing from {metadata}: {sorted(missing)}")
    return found


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


def export(root: Path, metadata: str, run_dir: Path, out_dir: Path, checkpoint: Optional[Path], device: str,
           query_tag: str = DEFAULT_QUERY, renders: Dict[str, str] = RENDERS) -> Dict[str, Any]:
    from reid.methods.vismatch_preprocessing import to_rgb_float_tensor

    if query_tag not in renders:
        raise ValueError(f"query tag {query_tag!r} is not one of {sorted(renders)}")
    rows = load_rows(root, metadata, list(renders.values()))
    backend, model = _load_backend(run_dir, checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)

    images: Dict[str, Dict[str, int]] = {}
    features: Dict[str, Any] = {}
    for tag, rel in renders.items():
        render = Image.open(root / rel).convert("RGB")
        mask = decode_rle(rows[rel]["mask"])
        features[tag] = backend.extract_frame(to_rgb_float_tensor(apply_mask(render, mask)))
        images[tag] = web_copy(render, out_dir / f"{tag}.jpg")
        images[tag]["raw_width"], images[tag]["raw_height"] = render.width, render.height
        print(f"[synthetic-demo] {tag}: {render.width}x{render.height}, mask share {mask.mean():.2f}, "
              f"{len(features[tag].keypoints)} keypoints", flush=True)

    query_identity = identity_of(renders[query_tag])
    query = {"tag": query_tag, "identity": query_identity, "source": renders[query_tag],
             "image": {k: images[query_tag][k] for k in ("file", "width", "height")}}
    candidates: List[Dict[str, Any]] = []
    for tag, rel in renders.items():
        if tag == query_tag:
            continue
        result = backend.match_features(features[query_tag], features[tag])
        web_q, web_g = images[query_tag], images[tag]
        kq = normalized_to_pixels(result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2)),
                                  (web_q["width"], web_q["height"]))
        kg = normalized_to_pixels(result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2)),
                                  (web_g["width"], web_g["height"]))
        record = candidate_record(tag, rel, query_identity, (web_g["width"], web_g["height"]), result, kq, kg)
        record["source"] = rel
        candidates.append(record)
        print(f"[synthetic-demo] {query_tag} vs {tag} ({record['identity']}): score {record['score']:.4f}, "
              f"{record['match_count']} matches", flush=True)

    payload = build_payload(query, candidates, model)
    (out_dir / "synthetic_demo.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[synthetic-demo] wrote {out_dir / 'synthetic_demo.json'}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="CzechLynx dataset root")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="synthetic metadata CSV, relative to the root")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                        help="fine-tuned LoMa probe run whose config.snapshot.yaml configures the matcher")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the checkpoint path recorded in the run snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="render tag used as the query")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not (args.root / args.metadata).is_file():
        print(f"error: metadata not found: {args.root / args.metadata}", file=sys.stderr)
        return 2
    export(args.root, args.metadata, args.run_dir, args.out, args.checkpoint, args.device, args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
