#!/usr/bin/env python3
"""Where do LoMa matches land when the background is left in? Default vs fine-tuned.

Hypothesis (user, 2026-10-02): at inference on photos *with* background, the default
LoMa matcher may spend correspondences on the background, while the matcher fine-tuned
on background-removed images is biased toward the animal. This script matches raw
(unmasked) photo pairs of one WildlifeReID-10k dataset with both matchers, classifies
every correspondence endpoint as animal or background using the provider's pre-masked
image (foreground = max(RGB) > 12, the same approximation as the image-quality audit),
and writes a CSV plus a contact sheet with animal matches in cyan and background matches
in orange. It also reports each matcher's score on the masked inputs the paper uses, for
reference.

The matcher configuration mirrors the paper's probe runs (LoMa-B, 512 px long side, up
to 512 keypoints, default threshold). The fine-tuned checkpoint is given explicitly and
its SHA-256 is recorded. Needs a GPU and the ex-reid environment. Outputs go under the
gitignored ``reports/project_page/analysis/``; this is an analysis, not page content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

from wildmatch.paths import path as _profile_path  # noqa: E402

WILDLIFE_ROOT = _profile_path("data_root") / "WildlifeReID-10k"
FOREGROUND_THRESHOLD = 12
MASK_COLORS = {"animal": "#19c2d6", "background": "#f28c28"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(masked_rel: str) -> str:
    return "images/" + masked_rel[len("masked_images/"):] if masked_rel.startswith("masked_images/") else masked_rel


class Layout:
    """Dataset layout: how to find the raw photo and the animal mask for a metadata row."""

    def __init__(self, root: Path, rows: Sequence[Dict[str, str]], identity_col: str, split_col: str, mask_col: Optional[str]):
        self.root, self.identity_col, self.split_col, self.mask_col = root, identity_col, split_col, mask_col
        self.by_path = {r["path"]: r for r in rows}

    def raw(self, path: str) -> Path:
        return self.root / (path if self.mask_col else raw_path(path))

    def mask(self, path: str) -> np.ndarray:
        if self.mask_col:  # dataset-provided COCO-RLE (CzechLynx): exact animal region
            from pycocotools import mask as mask_utils
            data = json.loads(self.by_path[path][self.mask_col])
            m = mask_utils.decode(data)
            if m.ndim == 3:
                m = m.max(axis=-1)
            return m.astype(bool)
        return animal_mask(self.root, path)  # pre-masked image threshold (WildlifeReID-10k)

    def masked_input(self, path: str) -> Image.Image:
        if self.mask_col:
            image = np.asarray(Image.open(self.raw(path)).convert("RGB"))
            return Image.fromarray(image * self.mask(path)[..., None].astype(np.uint8), "RGB")
        return Image.open(self.root / path).convert("RGB")


def normalized_to_pixels(points: np.ndarray, size_wh: Sequence[int]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(points)
    out[:, 0] = size_wh[0] * (points[:, 0] + 1.0) / 2.0 - 0.5
    out[:, 1] = size_wh[1] * (points[:, 1] + 1.0) / 2.0 - 0.5
    return out


def animal_mask(root: Path, masked_rel: str) -> np.ndarray:
    masked = np.asarray(Image.open(root / masked_rel).convert("RGB"))
    return masked.max(axis=2) > FOREGROUND_THRESHOLD


def on_animal(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return mask[ys, xs]


def load_pairs(root: Path, metadata: Path, split: str, n_same: int, n_diff: int, seed: int,
               pinned: Sequence[Tuple[str, str]] = (), identity_col: str = "identity", split_col: str = "split",
               mask_col: Optional[str] = None) -> Tuple[List[Dict[str, Any]], "Layout"]:
    """Random same-identity and different-identity pairs from one split, plus pinned pairs."""
    csv.field_size_limit(1 << 30)
    all_rows = list(csv.DictReader(metadata.open(newline="", encoding="utf-8")))
    layout = Layout(root, all_rows, identity_col, split_col, mask_col)
    rows = [r for r in all_rows if r[split_col] == split]
    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_id.setdefault(r[identity_col], []).append(r)
    rng = random.Random(seed)
    pairs: List[Dict[str, Any]] = []
    for a, b in pinned:  # pinned pairs may span splits (e.g. a query and its top-1 gallery image)
        same = layout.by_path[a][identity_col] == layout.by_path[b][identity_col]
        pairs.append({"kind": "same" if same else "different", "a": a, "b": b, "pinned": True})
    multi = sorted(k for k, v in by_id.items() if len(v) >= 2)
    rng.shuffle(multi)
    for ident in multi[:n_same]:
        a, b = rng.sample(by_id[ident], 2)
        pairs.append({"kind": "same", "a": a["path"], "b": b["path"], "pinned": False})
    idents = sorted(by_id); rng.shuffle(idents)
    for i in range(n_diff):
        ia, ib = idents[2 * i], idents[2 * i + 1]
        pairs.append({"kind": "different", "a": rng.choice(by_id[ia])["path"], "b": rng.choice(by_id[ib])["path"], "pinned": False})
    return pairs, layout


def build_backends(device: str, checkpoint: Path, loma_arch: str, resize_max: int, top_k: int,
                   joint_checkpoint: Optional[Path] = None):
    """Default LoMa, the matcher-only fine-tuned LoMa and, optionally, the joint
    (descriptor + matcher) checkpoint, which the loader requires to be used as ``full``."""
    from wildmatch.matchers.vismatch import VismatchMatcherBackend, _choose_vismatch_device
    from wildmatch.matchers.vismatch_profiles import default_matcher_threshold

    threshold = default_matcher_threshold("loma")
    dev = _choose_vismatch_device(device)
    backends = {
        "default": VismatchMatcherBackend("loma", dev, top_k, threshold, checkpoint_source="default", loma_arch=loma_arch,
                                          resize_max=resize_max),
        "fine-tuned": VismatchMatcherBackend("loma", dev, top_k, threshold, checkpoint_source="custom",
                                             checkpoint_path=str(checkpoint), checkpoint_components="matcher_only",
                                             loma_arch=loma_arch, resize_max=resize_max),
    }
    if joint_checkpoint is not None:
        backends["joint"] = VismatchMatcherBackend("loma", dev, top_k, threshold, checkpoint_source="custom",
                                                   checkpoint_path=str(joint_checkpoint), checkpoint_components="full",
                                                   loma_arch=loma_arch, resize_max=resize_max)
    return backends, threshold


def analyze(layout: "Layout", pairs: Sequence[Dict[str, Any]], backends, out_dir: Path, draw_top: int) -> List[Dict[str, Any]]:
    from wildmatch.matchers.vismatch_preprocessing import to_rgb_float_tensor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    fig, axes = plt.subplots(len(pairs), len(backends), figsize=(8 * len(backends), 4.2 * len(pairs)), squeeze=False)
    feature_cache: Dict[Tuple[str, str, str], Any] = {}

    def features(name, backend, rel, masked_input: bool):
        key = (name, rel, "masked" if masked_input else "raw")
        if key not in feature_cache:
            image = layout.masked_input(rel) if masked_input else Image.open(layout.raw(rel)).convert("RGB")
            feature_cache[key] = (backend.extract_frame(to_rgb_float_tensor(image)), image.size)
        return feature_cache[key]

    for row, pair in enumerate(pairs):
        mask_a, mask_b = layout.mask(pair["a"]), layout.mask(pair["b"])
        raw_a = Image.open(layout.raw(pair["a"])).convert("RGB")
        raw_b = Image.open(layout.raw(pair["b"])).convert("RGB")
        for col, (name, backend) in enumerate(backends.items()):
            (fa, size_a), (fb, size_b) = features(name, backend, pair["a"], False), features(name, backend, pair["b"], False)
            result = backend.match_features(fa, fb)
            ka = normalized_to_pixels(result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2)), size_a)
            kb = normalized_to_pixels(result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2)), size_b)
            conf = np.asarray(result.confidences if result.confidences is not None else [], dtype=np.float64)
            a_on, b_on = on_animal(ka, mask_a) if len(ka) else np.zeros(0, bool), on_animal(kb, mask_b) if len(kb) else np.zeros(0, bool)
            both = a_on & b_on
            # Keypoint-level view: where did the detector put keypoints at all (shared by both matchers).
            kp_a = normalized_to_pixels(np.asarray(fa.keypoints), size_a); kp_b = normalized_to_pixels(np.asarray(fb.keypoints), size_b)
            kp_a_on, kp_b_on = on_animal(kp_a, mask_a).mean(), on_animal(kp_b, mask_b).mean()
            # Reference: the paper's masked inputs.
            (ma, _), (mb, _) = features(name, backend, pair["a"], True), features(name, backend, pair["b"], True)
            masked_result = backend.match_features(ma, mb)
            rec = {"pair": row, "kind": pair["kind"], "pinned": pair["pinned"], "matcher": name,
                   "a": pair["a"], "b": pair["b"],
                   "raw_score": float(result.score), "raw_matches": int(result.match_count),
                   "matches_both_on_animal": int(both.sum()), "matches_any_background": int((~both).sum()),
                   "frac_matches_on_animal": float(both.mean()) if len(both) else None,
                   "conf_sum_on_animal": float(conf[both].sum()) if len(conf) else 0.0,
                   "conf_sum_background": float(conf[~both].sum()) if len(conf) else 0.0,
                   "keypoints_on_animal_a": float(kp_a_on), "keypoints_on_animal_b": float(kp_b_on),
                   "animal_area_a": float(mask_a.mean()), "animal_area_b": float(mask_b.mean()),
                   "masked_score": float(masked_result.score), "masked_matches": int(masked_result.match_count)}
            records.append(rec)
            print(f"[background] pair {row} ({pair['kind']}) {name:10s}: raw score {rec['raw_score']:.3f}, "
                  f"{rec['raw_matches']} matches, {rec['matches_both_on_animal']} on animal, {rec['matches_any_background']} touching background "
                  f"({(rec['frac_matches_on_animal'] or 0) * 100:.0f}% on animal); masked score {rec['masked_score']:.3f} ({rec['masked_matches']} m); "
                  f"keypoints on animal {kp_a_on * 100:.0f}% / {kp_b_on * 100:.0f}%", flush=True)
            # Draw: strongest `draw_top` matches, cyan on animal, orange touching background.
            ax = axes[row][col]
            gap = 12
            canvas = Image.new("RGB", (raw_a.width + gap + raw_b.width, max(raw_a.height, raw_b.height)), "white")
            canvas.paste(raw_a, (0, 0)); canvas.paste(raw_b, (raw_a.width + gap, 0))
            ax.imshow(canvas); ax.axis("off")
            order = np.argsort(-conf, kind="stable")[:draw_top]
            for i in order:
                color = MASK_COLORS["animal"] if both[i] else MASK_COLORS["background"]
                ax.plot([ka[i, 0], kb[i, 0] + raw_a.width + gap], [ka[i, 1], kb[i, 1]], "-", color=color, lw=0.9, alpha=0.9)
                ax.plot([ka[i, 0], kb[i, 0] + raw_a.width + gap], [ka[i, 1], kb[i, 1]], "o", ms=2.2, color=color)
            ax.set_title(f"{name} · {pair['kind']} pair · score {rec['raw_score']:.3f}, {rec['raw_matches']} matches, "
                         f"{(rec['frac_matches_on_animal'] or 0) * 100:.0f}% on animal (bg matches orange) · masked-input score {rec['masked_score']:.3f}",
                         fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "background_matches.png", dpi=80)
    with (out_dir / "background_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader(); writer.writerows(records)
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=WILDLIFE_ROOT)
    parser.add_argument("--metadata", type=Path, default=WILDLIFE_ROOT / "metadata_mdsplit_no_background/metadata_NyalaData.csv")
    parser.add_argument("--split", default="train", help="metadata split value to draw pairs from (train = database)")
    parser.add_argument("--split-col", default="split"); parser.add_argument("--identity-col", default="identity")
    parser.add_argument("--mask-col", default=None, help="COCO-RLE mask column (CzechLynx: mask); omit for pre-masked datasets")
    parser.add_argument("--checkpoint", type=Path, required=True, help="fine-tuned LoMa matcher checkpoint (matcher only)")
    parser.add_argument("--joint-checkpoint", type=Path, default=None,
                        help="joint descriptor + matcher LoMa checkpoint (loaded with checkpoint_components=full)")
    parser.add_argument("--n-same", type=int, default=4)
    parser.add_argument("--n-diff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin", action="append", default=[], metavar="A_PATH,B_PATH", help="pinned pair of metadata paths")
    parser.add_argument("--draw-top", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loma-arch", default="LoMa-B"); parser.add_argument("--resize-max", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "project_page" / "analysis" / "nyala_background")
    args = parser.parse_args(list(argv) if argv is not None else None)
    pinned = [tuple(p.split(",", 1)) for p in args.pin]
    pairs, layout = load_pairs(args.root, args.metadata, args.split, args.n_same, args.n_diff, args.seed, pinned,
                               args.identity_col, args.split_col, args.mask_col)
    backends, threshold = build_backends(args.device, args.checkpoint, args.loma_arch, args.resize_max, args.top_k,
                                         args.joint_checkpoint)
    records = analyze(layout, pairs, backends, args.out, args.draw_top)
    summary: Dict[str, Any] = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                               "threshold": threshold, "pairs": len(pairs), "mask_source": "dataset RLE" if args.mask_col else "pre-masked image threshold",
                               "joint_checkpoint": str(args.joint_checkpoint) if args.joint_checkpoint else None,
                               "joint_checkpoint_sha256": sha256_file(args.joint_checkpoint) if args.joint_checkpoint else None}
    for name in backends:
        rows = [r for r in records if r["matcher"] == name]
        fr = [r["frac_matches_on_animal"] for r in rows if r["frac_matches_on_animal"] is not None]
        summary[name] = {"mean_frac_matches_on_animal": float(np.mean(fr)) if fr else None,
                         "mean_raw_matches": float(np.mean([r["raw_matches"] for r in rows])),
                         "mean_background_matches": float(np.mean([r["matches_any_background"] for r in rows])),
                         "same_pair_mean_raw_score": float(np.mean([r["raw_score"] for r in rows if r["kind"] == "same"])),
                         "diff_pair_mean_raw_score": float(np.mean([r["raw_score"] for r in rows if r["kind"] == "different"])) if any(r["kind"] == "different" for r in rows) else None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
