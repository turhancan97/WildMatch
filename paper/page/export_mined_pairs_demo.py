#!/usr/bin/env python3
"""Export the project-page mined-pairs browser: for an anchor image, its five mined
positives and five hard negatives with the pretrained matcher's scores.

This makes the paper's "weak supervision" concrete. Pair mining ran once with the
*pretrained* LoMa matcher on the CzechLynx time-closed training split
(``rdd-parallel-benchmark``: 20 images per collection, the five highest-scoring
same-identity images as positives and the five highest-scoring other-identity images as
hard negatives per anchor, ten anchors per collection ranked by their best pair score).
The per-collection mining reports keep every pair score; the combined index lists the
anchors that fine-tuning used. This script joins the two, picks anchors by appearance
(daylight colour photos with a visible animal, one per individual, random seed), and
exports, per anchor, the ten mined partners with their stored mining scores. It also
re-matches every anchor/partner pair with the pretrained matcher on the masked inputs
so the page can draw the correspondences behind each score; the recomputed score is
stored next to the mined one (single-image extraction moves a few keypoints relative to
the mining run's cached features, so they differ slightly).

Photos are displayed raw (with background); matching uses the masked images the mining
run used. Needs a GPU and the ex-reid environment unless ``--no-matches`` is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # run as a script: make `paper` importable
    sys.path.insert(0, str(REPO_ROOT))
from paper.page.export_synthetic_demo import normalized_to_pixels  # noqa: E402

from wildmatch.paths import path as _profile_path  # noqa: E402

_MINING_OUTPUTS = _profile_path("external.mining_outputs")  # rdd-parallel-benchmark outputs; None when unset
BENCH = _MINING_OUTPUTS / "czechlynx-time-closed" / "legacy" / "loma" if _MINING_OUTPUTS else None
VIEW_ROOT = _profile_path("data_root") / "CzechLynx_processed_time_closed"
RAW_ROOT = _profile_path("data_root") / "CzechLynx_v2"
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "mined_pairs"
WEB_LONG_SIDE = 560
ATTRIBUTION = "Photographs from CzechLynx (Picek et al.), time-closed training split."
MINING = {"matcher": "pretrained LoMa-B", "images_per_collection": 20, "positives_per_anchor": 5,
          "hard_negatives_per_anchor": 5, "anchors_per_collection": 10,
          "score": "pretrained matcher image score on background-removed inputs: summed confidence of mutual matches above the threshold, divided by the smaller keypoint count"}


def masked_rel(view_frame: str, view_root: Path = VIEW_ROOT) -> str:
    """View frame (symlink) -> path of the masked image relative to the CzechLynx_v2 root."""
    full = view_root / view_frame if not os.path.isabs(view_frame) else Path(view_frame)
    target = os.readlink(full) if os.path.islink(full) else str(full)
    if "/CzechLynx_v2/" not in target:
        raise ValueError(f"unexpected symlink target for {view_frame}: {target}")
    return target.split("/CzechLynx_v2/", 1)[1]


def raw_rel(masked: str) -> str:
    """``CzechLynx_masked/<site>/<id>/<file>`` -> ``CzechLynx/<site>/<id>/<file>`` (the raw photo)."""
    if not masked.startswith("CzechLynx_masked/"):
        raise ValueError(f"not a masked path: {masked}")
    return "CzechLynx/" + masked[len("CzechLynx_masked/"):]


def identity_of(view_frame: str) -> str:
    parts = Path(view_frame).parts
    return parts[1] if parts[0] in ("train", "val", "test") else parts[0]


def photo_tag(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


def warm_share(raw: Image.Image, masked: Image.Image) -> Tuple[float, float]:
    """(share of warm animal pixels, animal share of the frame) from the masked image."""
    m = np.asarray(masked.convert("RGB")).max(axis=2) > 12
    if not m.any():
        return 0.0, 0.0
    hsv = np.asarray(raw.convert("HSV")).astype(float)
    hue = hsv[..., 0][m] * 360 / 255; sat = hsv[..., 1][m] / 255; val = hsv[..., 2][m] / 255
    warm = ((hue < 50) | (hue > 330)) & (sat > 0.2) & (val > 0.2)
    return float(warm.mean()), float(m.mean())


def load_anchors(bench: Path, view_root: Path) -> List[Dict[str, Any]]:
    """Anchors of the combined training index with their mined pools and scores."""
    combined = {e["query_frame"] for e in json.loads((bench / "strong-matches_train_combined.json").read_text(encoding="utf-8"))}
    anchors: List[Dict[str, Any]] = []
    for report_path in sorted(glob.glob(str(bench / "strong-matches_train_[0-9]*.json"))):
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for frame in report["selected_frames"]:
            rel = os.path.relpath(frame["query_frame"], view_root)
            if rel not in combined:
                continue
            anchors.append({"view": rel, "identity": report["query_identity"], "site": report["query_source"],
                            "selection_score": float(frame["selection_score"]),
                            "positives": [{"view": os.path.relpath(p["frame"], view_root), "identity": p["identity"],
                                           "collection": p["collection"], "mined_score": float(p["score"])} for p in frame["positives"]],
                            "negatives": [{"view": os.path.relpath(n["frame"], view_root), "identity": n["identity"],
                                           "collection": n["collection"], "mined_score": float(n["score"])} for n in frame["negatives"]]})
    return anchors


def select_anchors(anchors: Sequence[Dict[str, Any]], raw_root: Path, n: int, seed: int, min_warm: float = 0.5,
                   min_foreground: float = 0.10) -> List[Dict[str, Any]]:
    """Full pools only, one anchor per individual, daylight colour photos with a visible animal."""
    full = [a for a in anchors if len(a["positives"]) == 5 and len(a["negatives"]) == 5]
    rng = random.Random(seed)
    rng.shuffle(full)
    chosen: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for a in full:
        if a["identity"] in seen_ids:
            continue
        masked = masked_rel(a["view"])
        raw = Image.open(raw_root / raw_rel(masked)).convert("RGB")
        warm, fg = warm_share(raw, Image.open(raw_root / masked))
        if warm < min_warm or fg < min_foreground:
            continue
        a["warm_share"], a["foreground_share"] = round(warm, 3), round(fg, 3)
        chosen.append(a); seen_ids.add(a["identity"])
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} anchors pass the appearance filter")
    return chosen


def build_payload(anchors: List[Dict[str, Any]], settings: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_mined_pairs_demo.py",
        "attribution": ATTRIBUTION,
        "mining": MINING,
        "settings": settings,
        "anchors": anchors,
    }
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"demo payload contains a private path fragment {needle!r}")
    return payload


def export(bench: Path, view_root: Path, raw_root: Path, out_dir: Path, n: int, seed: int, device: str,
           compute_matches: bool = True) -> Dict[str, Any]:
    anchors = select_anchors(load_anchors(bench, view_root), raw_root, n, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = None
    settings: Dict[str, Any] = {"web_long_side": WEB_LONG_SIDE, "anchors": n, "seed": seed, "matches": compute_matches}
    if compute_matches:
        from wildmatch.matchers.vismatch import VismatchMatcherBackend, _choose_vismatch_device
        from wildmatch.matchers.vismatch_profiles import default_matcher_threshold
        settings.update({"loma_arch": "LoMa-B", "resize_max": 512, "top_k": 512, "threshold": default_matcher_threshold("loma"),
                         "checkpoint": "default (pretrained)"})
        backend = VismatchMatcherBackend("loma", _choose_vismatch_device(device), 512, settings["threshold"],
                                         checkpoint_source="default", loma_arch="LoMa-B", resize_max=512)

    images: Dict[str, Dict[str, int]] = {}
    features: Dict[str, Any] = {}

    def photo(view: str) -> Dict[str, Any]:
        masked = masked_rel(view, view_root)
        raw = raw_rel(masked)
        tag = photo_tag(raw)
        if tag not in images:
            image = Image.open(raw_root / raw).convert("RGB")
            scale = min(1.0, WEB_LONG_SIDE / max(image.size))
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            (image.resize(size, Image.LANCZOS) if scale < 1.0 else image).save(out_dir / f"{tag}.jpg", format="JPEG",
                                                                              quality=85, optimize=True, progressive=True)
            images[tag] = {"file": f"{tag}.jpg", "width": size[0], "height": size[1]}
            if backend is not None:
                from wildmatch.matchers.vismatch_preprocessing import to_rgb_float_tensor
                features[tag] = backend.extract_frame(to_rgb_float_tensor(Image.open(raw_root / masked).convert("RGB")))
        return {"tag": tag, "image": images[tag], "source": raw}

    exported: List[Dict[str, Any]] = []
    for anchor in anchors:
        a = photo(anchor["view"])
        record = {"identity": anchor["identity"], "site": anchor["site"], "selection_score": round(anchor["selection_score"], 4),
                  "anchor": a, "positives": [], "negatives": []}
        for pool in ("positives", "negatives"):
            for partner in anchor[pool]:
                p = photo(partner["view"])
                entry = {**p, "identity": partner["identity"], "collection": partner["collection"],
                         "mined_score": round(partner["mined_score"], 4)}
                if backend is not None:
                    result = backend.match_features(features[a["tag"]], features[p["tag"]])
                    kq = normalized_to_pixels(result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2)),
                                              (a["image"]["width"], a["image"]["height"]))
                    kg = normalized_to_pixels(result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2)),
                                              (p["image"]["width"], p["image"]["height"]))
                    conf = np.asarray(result.confidences if result.confidences is not None else [], dtype=np.float64).reshape(-1)
                    entry.update({"recomputed_score": round(float(result.score), 4), "match_count": int(result.match_count),
                                  "points": {"anchor": [[round(float(x), 1), round(float(y), 1)] for x, y in kq],
                                             "partner": [[round(float(x), 1), round(float(y), 1)] for x, y in kg]},
                                  "confidence": [round(float(c), 4) for c in conf],
                                  "order_by_confidence": [int(i) for i in np.argsort(-conf, kind="stable")]})
                record[pool].append(entry)
        pos = [p["mined_score"] for p in record["positives"]]; neg = [p["mined_score"] for p in record["negatives"]]
        print(f"[mined-pairs] {anchor['identity']} ({anchor['site']}): positives {min(pos):.3f}-{max(pos):.3f}, "
              f"hard negatives {min(neg):.3f}-{max(neg):.3f}" +
              (f"; recomputed vs mined max |d| {max(abs(e['recomputed_score'] - e['mined_score']) for e in record['positives'] + record['negatives']):.3f}" if backend else ""),
              flush=True)
        exported.append(record)
    payload = build_payload(exported, settings)
    (out_dir / "mined_pairs.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[mined-pairs] wrote {out_dir / 'mined_pairs.json'} ({len(exported)} anchors, {len(images)} photos)")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, default=BENCH, help="mining output directory (combined index + per-collection reports)")
    parser.add_argument("--view-root", type=Path, default=VIEW_ROOT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-anchors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-matches", action="store_true", help="skip re-matching (CPU only, no correspondences)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.bench is None:
        print("error: no mining output directory; pass --bench or set paths external.mining_outputs", file=sys.stderr)
        return 2
    if not (args.bench / "strong-matches_train_combined.json").is_file():
        print(f"error: combined index not found under {args.bench}", file=sys.stderr)
        return 2
    export(args.bench, args.view_root, args.raw_root, args.out, args.n_anchors, args.seed, args.device, not args.no_matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
