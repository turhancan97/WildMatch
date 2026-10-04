#!/usr/bin/env python3
"""Export the project-page background-masking demo (SAM 3) on CzechLynx synthetic renders.

The demo shows, for each render, the raw image and the model input after background
masking with text-prompted SAM 3, with a draggable divider, mask outlines and the
detection readouts (prompt, confidence, instance count, foreground share). Because the
synthetic renders ship with their own masks, the demo also reports the agreement
between the SAM 3 mask and the dataset mask (intersection over union).

Inputs
------
* ``--sam3-dir``: the output directory of ``scripts/segment_with_sam3.py --segment``
  run on the demo renders (``masks.csv`` with full-size COCO-RLE masks, score,
  instance count, threshold and prompt per render).
* ``--root`` / ``--metadata``: the CzechLynx dataset root and synthetic metadata CSV,
  for the raw renders and their dataset masks.
* ``--renders``: CSV listing the renders (``path``, ``identity``, ``role``); by
  default the twenty renders of the synthetic match demo (``INDIVIDUALS`` in
  ``export_synthetic_demo.py``), so both demos show the same animals.

Outputs ``docs/assets/demo/masking/``: ``<tag>.jpg`` (raw render, web size),
``<tag>_sam3.png`` and ``<tag>_dataset.png`` (binary masks at the same size) and
``masking_demo.json``. Images are CC BY 4.0 (CzechLynx synthetic subset, Picek et al.,
Zenodo 17592004); the page attributes them and labels the demo as synthetic. CPU only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_synthetic_demo import ATTRIBUTION, INDIVIDUALS, decode_rle, identity_of, load_rows  # noqa: E402
from wildmatch.reporting.paper_datasets import PAPER_PROFILES  # noqa: E402

from wildmatch.paths import path as _profile_path  # noqa: E402

DEFAULT_ROOT = _profile_path("data_root") / "CzechLynx_v2"
DEFAULT_METADATA = "CzechLynxDataset-Metadata-Synthetic.csv"
DEFAULT_SAM3_DIR = REPO_ROOT / "reports" / "project_page" / "sam3_synthetic"
DEFAULT_OUT = REPO_ROOT / "docs" / "assets" / "demo" / "masking"
WEB_LONG_SIDE = 900
DEFAULT_SAM3_REAL_DIR = REPO_ROOT / "reports" / "project_page" / "sam3_real"
MATCH_EXAMPLES_JSON = REPO_ROOT / "docs" / "assets" / "match" / "match_examples.json"
# Real photographs: the match-figure pairs the user picked by eye (two per paper dataset).
# A reference mask exists only where the dataset itself ships segmentation masks, i.e.
# CzechLynx (RLE in the metadata); the WildlifeReID-10k and SalamanderID2025 masks were
# computed by us, so those photos show the SAM 3 mask alone (user decision 2026-10-02).
# SalamanderID2025's team allowed display on 2026-10-02.
REAL_ATTRIBUTION = ("Photographs from the paper's datasets: CzechLynx (Picek et al.), WildlifeReID-10k "
                    "(Adam et al.; HyenaID2022, LeopardID2022, NyalaData, SeaStarReID2023, WhaleSharkID, "
                    "ZindiTurtleRecall) and SalamanderID2025 (AnimalCLEF 2025, shown with the dataset team's permission).")
PROFILE_BY_KEY = {p.key: p for p in PAPER_PROFILES}
ROOT_KEYS = {"CzechLynx_v2": "czechlynx", "WildlifeReID-10k": "wildlife", "SalamanderID2025": "salamander"}


def default_renders() -> List[Dict[str, str]]:
    """The synthetic match demo's renders: tag, path, identity and role."""
    out: List[Dict[str, str]] = []
    for identity, sides in INDIVIDUALS.items():
        for role, rel in sides.items():
            out.append({"tag": f"{'Q' if role == 'query' else 'G'}_{identity}", "path": rel,
                        "identity": identity, "role": role})
    return out


def write_renders_csv(path: Path, renders: Sequence[Dict[str, str]]) -> None:
    """The row list the SAM 3 script segments (it needs a ``path`` column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "identity", "role", "tag"])
        writer.writeheader()
        for row in renders:
            writer.writerow({k: row[k] for k in ("path", "identity", "role", "tag")})


def _metadata_row(profile, path: str) -> Dict[str, str]:
    csv.field_size_limit(1 << 30)
    with Path(profile.metadata).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("path") == path:
                return row
    raise FileNotFoundError(f"{profile.key}: {path} not in {profile.metadata}")


def real_items(match_examples: Path = MATCH_EXAMPLES_JSON) -> List[Dict[str, Any]]:
    """The match-figure photos (query and top-1 per dataset) with raw path and reference mask."""
    examples = json.loads(match_examples.read_text(encoding="utf-8"))["examples"]
    items: List[Dict[str, Any]] = []
    for example in examples:
        profile = PROFILE_BY_KEY[example["dataset"]]
        root_key = ROOT_KEYS[profile.dataset_name]
        for role, meta_path in (("query", example["query_path"]), ("gallery", example["gallery_path"])):
            row = _metadata_row(profile, meta_path)
            if profile.mask_col:                       # CzechLynx: raw photo + RLE mask shipped with the dataset
                raw, reference = meta_path, {"type": "rle", "label": "dataset mask (RLE)"}
            elif row.get("original_path"):             # SalamanderID2025: masks were computed by us -> no reference
                raw, reference = row["original_path"], None
            else:                                      # WildlifeReID-10k: pre-masked images computed by us -> no reference
                raw = "images/" + meta_path[len("masked_images/"):] if meta_path.startswith("masked_images/") else meta_path
                reference = None
            items.append({"tag": f"{example['dataset']}_{role}", "dataset": example["dataset"], "dataset_label": example["label"],
                          "identity": str(row.get(profile.identity_col, "")), "role": role, "root_key": root_key,
                          "root": str(profile.root), "path": raw, "metadata_path": meta_path, "reference": reference,
                          "metadata_mask": row.get("mask") or ""})
    return items


def write_real_render_csvs(out_dir: Path, items: Sequence[Dict[str, Any]]) -> Dict[str, Path]:
    """One render list per dataset root for the SAM 3 script (paths relative to that root)."""
    written: Dict[str, Path] = {}
    for root_key in sorted({i["root_key"] for i in items}):
        rows = [i for i in items if i["root_key"] == root_key]
        path = out_dir / root_key / "renders.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "dataset", "role", "tag"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in ("path", "dataset", "role", "tag")})
        written[root_key] = path
        print(f"[masking-demo] {root_key}: {len(rows)} photos, root {rows[0]['root']}")
    return written


def reference_mask(item: Dict[str, Any], size_hw: Sequence[int]) -> Optional[np.ndarray]:
    """The dataset's own segmentation mask, or None when the dataset ships none."""
    reference = item.get("reference")
    if not reference:
        return None
    if reference["type"] == "rle":
        return decode_rle(item["metadata_mask"])
    raise ValueError(f"{item['tag']}: unknown reference type {reference['type']!r}")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def web_size(size: Sequence[int], long_side: int = WEB_LONG_SIDE) -> tuple:
    scale = min(1.0, long_side / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


def mask_png(mask: np.ndarray, size: Sequence[int], out_path: Path) -> None:
    """Binary mask resized to the web size (nearest neighbour) as an 8-bit PNG."""
    image = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize(tuple(size), Image.NEAREST)
    image.save(out_path, format="PNG", optimize=True)


def load_sam3_rows(sam3_dir: Path, paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    masks_csv = sam3_dir / "masks.csv"
    if not masks_csv.is_file():
        raise FileNotFoundError(f"SAM 3 output not found: {masks_csv} (run scripts/segment_with_sam3.py --segment first)")
    csv.field_size_limit(1 << 30)
    found: Dict[str, Dict[str, str]] = {}
    with masks_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("path") in set(paths):
                found[row["path"]] = row
    missing = set(paths) - set(found)
    if missing:
        raise FileNotFoundError(f"renders missing from {masks_csv}: {sorted(missing)}")
    return found


def item_record(render: Dict[str, Any], size: Sequence[int], sam3_row: Dict[str, str], sam3_mask: np.ndarray,
                dataset_mask: Optional[np.ndarray]) -> Dict[str, Any]:
    reference = render.get("reference") if "reference" in render else {"type": "rle", "label": "dataset mask (RLE)"}
    record = {
        "tag": render["tag"], "identity": render["identity"], "role": render["role"], "source": render["path"],
        "dataset": render.get("dataset", "synthetic"), "dataset_label": render.get("dataset_label", "CzechLynx synthetic"),
        "reference_mask_source": reference["label"] if reference else None,
        "image": {"file": f"{render['tag']}.jpg", "width": int(size[0]), "height": int(size[1])},
        "sam3_mask": f"{render['tag']}_sam3.png",
        "dataset_mask": f"{render['tag']}_dataset.png" if dataset_mask is not None else None,
        "sam3": {"prompt": sam3_row.get("prompt_used", ""), "threshold": _float(sam3_row.get("threshold_used")),
                 "score": _float(sam3_row.get("best_score")), "instances": _int(sam3_row.get("n_detections")),
                 "merge": sam3_row.get("merge", ""), "foreground_fraction": round(float(sam3_mask.mean()), 4)},
        "dataset_foreground_fraction": round(float(dataset_mask.astype(bool).mean()), 4) if dataset_mask is not None else None,
        "iou_with_dataset_mask": round(iou(sam3_mask, dataset_mask), 4) if dataset_mask is not None else None,
    }
    return record


def _float(value: Optional[str]) -> Optional[float]:
    try:
        return round(float(value), 4) if value not in (None, "") else None
    except ValueError:
        return None


def _int(value: Optional[str]) -> Optional[int]:
    try:
        return int(float(value)) if value not in (None, "") else None
    except ValueError:
        return None


def build_payload(groups: List[Dict[str, Any]], run: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the demo JSON: one group per image source, each with its own summary."""
    for group in groups:
        ious = [i["iou_with_dataset_mask"] for i in group["items"] if i["iou_with_dataset_mask"] is not None]
        group["summary"] = {"items": len(group["items"]), "items_with_reference": len(ious),
                            "mean_iou_with_reference": round(float(np.mean(ious)), 4) if ious else None,
                            "min_iou_with_reference": round(float(np.min(ious)), 4) if ious else None}
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "scripts/export_masking_demo.py",
        "method": "Text-prompted SAM 3 segmentation; all detected instances merged into one mask; pixels outside the mask set to black",
        "run": run,
        "groups": groups,
    }
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"demo payload contains a private path fragment {needle!r}")
    return payload


def export_items(root: Path, sam3_rows: Dict[str, Dict[str, str]], out_dir: Path, renders: Sequence[Dict[str, Any]],
                 dataset_masks: Dict[str, np.ndarray] | None, prompts: set) -> List[Dict[str, Any]]:
    """Web copies, mask PNGs and records for one list of renders under one root."""
    items: List[Dict[str, Any]] = []
    for render in renders:
        image = Image.open(root / render["path"]).convert("RGB")
        size = web_size(image.size)
        image.resize(size, Image.LANCZOS).save(out_dir / f"{render['tag']}.jpg", format="JPEG", quality=88,
                                              optimize=True, progressive=True)
        sam3_row = sam3_rows[render["path"]]
        sam3_mask = decode_rle(sam3_row["mask"]) if sam3_row.get("mask") else np.zeros((image.height, image.width), np.uint8)
        ref = dataset_masks[render["path"]] if dataset_masks is not None else reference_mask(render, (image.height, image.width))
        if sam3_mask.shape != (image.height, image.width) or (ref is not None and ref.shape != sam3_mask.shape):
            raise ValueError(f"{render['tag']}: mask sizes do not match the photo {image.size}")
        mask_png(sam3_mask, size, out_dir / f"{render['tag']}_sam3.png")
        if ref is not None:
            mask_png(ref, size, out_dir / f"{render['tag']}_dataset.png")
        item = item_record(render, size, sam3_row, sam3_mask, ref)
        prompts.add(item["sam3"]["prompt"])
        items.append(item)
        iou_text = f"IoU vs reference {item['iou_with_dataset_mask']:.3f}" if ref is not None else "no reference mask"
        print(f"[masking-demo] {render['tag']}: score {item['sam3']['score']}, {item['sam3']['instances']} instance(s), "
              f"fg {item['sam3']['foreground_fraction']:.2f}, {iou_text}", flush=True)
    return items


def export(root: Path, metadata: str, sam3_dir: Path, out_dir: Path, renders: Sequence[Dict[str, str]],
           sam3_real_dir: Optional[Path] = None, real: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts: set = set()
    groups: List[Dict[str, Any]] = []

    paths = [r["path"] for r in renders]
    dataset_rows = load_rows(root, metadata, paths)
    for render in renders:
        if identity_of(render["path"]) != render["identity"]:
            raise ValueError(f"{render['tag']}: path belongs to another individual")
    dataset_masks = {p: decode_rle(dataset_rows[p]["mask"]) for p in paths}
    synthetic_items = export_items(root, load_sam3_rows(sam3_dir, paths), out_dir, renders, dataset_masks, prompts)
    groups.append({"key": "synthetic", "label": "Synthetic renders", "synthetic": True, "attribution": ATTRIBUTION,
                   "reference": "mask shipped with the CzechLynx synthetic subset", "items": synthetic_items})

    if sam3_real_dir is not None and real:
        real_group_items: List[Dict[str, Any]] = []
        for root_key in sorted({i["root_key"] for i in real}):
            subset = [i for i in real if i["root_key"] == root_key]
            sam3_rows = load_sam3_rows(sam3_real_dir / root_key, [i["path"] for i in subset])
            real_group_items.extend(export_items(Path(subset[0]["root"]), sam3_rows, out_dir, subset, None, prompts))
        order = [e["dataset"] for e in json.loads(MATCH_EXAMPLES_JSON.read_text(encoding="utf-8"))["examples"]]
        real_group_items.sort(key=lambda i: (order.index(i["dataset"]), i["role"] != "query"))
        groups.append({"key": "real", "label": "Camera-trap and field photographs", "synthetic": False,
                       "attribution": REAL_ATTRIBUTION,
                       "reference": "the dataset's own segmentation mask where one ships (CzechLynx); none otherwise",
                       "items": real_group_items})

    run = {"segmenter": "SAM 3 (text prompt)", "prompts": sorted(p for p in prompts if p),
           "script": "scripts/segment_with_sam3.py --segment", "masks_csv": "masks.csv"}
    payload = build_payload(groups, run)
    (out_dir / "masking_demo.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[masking-demo] wrote {out_dir / 'masking_demo.json'}: " +
          ", ".join(f"{g['key']} {g['summary']['items']} items, mean IoU {g['summary']['mean_iou_with_reference']}" for g in groups))
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--sam3-dir", type=Path, default=DEFAULT_SAM3_DIR,
                        help="output directory of segment_with_sam3.py --segment for the demo renders")
    parser.add_argument("--renders", type=Path, default=None,
                        help="CSV with path,identity,role,tag (default: the synthetic match demo's renders)")
    parser.add_argument("--write-renders-csv", type=Path, default=None,
                        help="only write the render list for the SAM 3 run to this path and exit")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sam3-real-dir", type=Path, default=DEFAULT_SAM3_REAL_DIR,
                        help="parent of per-root SAM 3 output dirs (czechlynx/, wildlife/, salamander/) for the photographs")
    parser.add_argument("--real", action="store_true", help="also export the real-photo group (match-figure pairs)")
    parser.add_argument("--write-real-csvs", action="store_true",
                        help="only write the per-root render lists for the real-photo SAM 3 runs and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)
    renders = default_renders()
    if args.renders is not None:
        with args.renders.open(newline="", encoding="utf-8") as handle:
            renders = [dict(row) for row in csv.DictReader(handle)]
    if args.write_renders_csv is not None:
        write_renders_csv(args.write_renders_csv, renders)
        print(f"[masking-demo] wrote {args.write_renders_csv} ({len(renders)} renders)")
        return 0
    if args.write_real_csvs:
        write_real_render_csvs(args.sam3_real_dir, real_items())
        return 0
    export(args.root, args.metadata, args.sam3_dir, args.out, renders,
           sam3_real_dir=args.sam3_real_dir if args.real else None, real=real_items() if args.real else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
