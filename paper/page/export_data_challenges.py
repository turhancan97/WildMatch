#!/usr/bin/env python3
"""Project-page "Data challenges" page: candidate contact sheets for review by eye, and the
export of the confirmed examples plus the audit summary.

Two subcommands:

* ``candidates`` ranks images per challenge category from the image-quality audit CSVs
  (``experiments/image-quality/<dataset>.csv``, written by ``wildmatch audit``),
  resolves the RAW photo of each (never the masked model input, so a problem must be visible
  in the photo itself) and writes one contact sheet per category under
  ``reports/project_page/data_challenges/`` with ``dataset:row`` labels and the ranking
  metric. Flags are heuristic rankings, not labels: nothing goes on the page before a human
  has confirmed it on these sheets.
* ``render`` takes the confirmed ``EXAMPLES`` list, writes web-sized raw JPEGs and
  ``challenges.json`` (per image: dataset, row, side, identity, audit measurements, flags and
  the query's Top-1 rate over completed runs) to ``docs/assets/datasets/challenges/`` and the
  audit summary table to ``docs/data/image_quality_summary.json``.

Categories follow the paper's figure and AGENTS.md: dataset noise (overexposure, empty frame,
insufficient detail, corruption, blur) and inherent difficulties (night and infrared frames,
occlusion). Mask-only problems are described in text and have no category here. BelugaID is
not a paper dataset and is skipped; ``lynx_open`` audits the same images as ``lynx_closed``.
CPU only (pandas, Pillow).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]

from wildmatch.reporting.paper_datasets import PAPER_PROFILES, PaperProfile  # noqa: E402

AUDIT_DIR = REPO_ROOT / "experiments" / "image-quality"
SHEET_DIR = REPO_ROOT / "reports" / "project_page" / "data_challenges"
WEB_DIR = REPO_ROOT / "docs" / "assets" / "datasets" / "challenges"
SUMMARY_OUT = REPO_ROOT / "docs" / "data" / "image_quality_summary.json"
WEB_LONG_SIDE = 900
PAGE_DATASETS = ["lynx_closed", "hyena", "leopard", "nyala", "salamander", "sea_star", "whale_shark", "turtle"]
PROFILES: Dict[str, PaperProfile] = {p.key: p for p in PAPER_PROFILES}
METRIC_COLUMNS = [
    "foreground_fraction",
    "mask_components",
    "largest_component_fraction",
    "mean_luma",
    "std_luma",
    "saturated_fraction",
    "sharpness",
    "grayscale",
    "width",
    "height",
]


class Category(NamedTuple):
    key: str
    group: str  # "noise" or "difficulty"
    title: str
    # (dataset key, audit flag column or expression, sort metric, ascending, count)
    sources: Tuple[Tuple[str, str, str, bool, int], ...]


CATEGORIES: List[Category] = [
    Category(
        "overexposure",
        "noise",
        "Overexposure",
        (
            ("lynx_closed", "flag_overexposed", "saturated_fraction", False, 12),
            ("leopard", "flag_overexposed", "saturated_fraction", False, 8),
            ("hyena", "flag_overexposed", "saturated_fraction", False, 4),
            ("turtle", "flag_overexposed", "saturated_fraction", False, 3),
        ),
    ),
    Category(
        "empty_frame",
        "noise",
        "Empty frame",
        (
            ("lynx_closed", "flag_tiny_foreground", "foreground_fraction", True, 16),
            ("leopard", "flag_empty_foreground", "foreground_fraction", True, 4),
            ("turtle", "flag_empty_foreground", "foreground_fraction", True, 1),
            ("hyena", "flag_empty_foreground", "foreground_fraction", True, 1),
        ),
    ),
    Category(
        "insufficient_detail",
        "noise",
        "Insufficient detail",
        (
            ("turtle", "flag_tiny_foreground", "foreground_fraction", True, 8),
            ("whale_shark", "flag_tiny_foreground", "foreground_fraction", True, 8),
            ("leopard", "flag_tiny_foreground", "foreground_fraction", True, 6),
            ("nyala", "flag_tiny_foreground", "foreground_fraction", True, 4),
        ),
    ),
    Category(
        "corruption",
        "noise",
        "Corruption",
        (
            ("lynx_closed", "flag_low_contrast & ~grayscale", "std_luma", True, 12),
            ("leopard", "flag_low_contrast", "std_luma", True, 6),
            ("whale_shark", "flag_low_contrast", "std_luma", True, 6),
        ),
    ),
    Category(
        "blur",
        "noise",
        "Blur",
        (
            ("hyena", "flag_blurry", "sharpness", True, 6),
            ("salamander", "flag_blurry", "sharpness", True, 6),
            ("leopard", "flag_blurry", "sharpness", True, 4),
            ("turtle", "flag_blurry", "sharpness", True, 4),
            ("nyala", "flag_blurry", "sharpness", True, 3),
            ("whale_shark", "flag_blurry", "sharpness", True, 3),
        ),
    ),
    Category(
        "night_infrared",
        "difficulty",
        "Night and infrared frames",
        (
            ("hyena", "flag_underexposed", "mean_luma", True, 8),
            ("lynx_closed", "grayscale & ~flag_overexposed & ~flag_underexposed", "mean_luma", True, 8),
            ("lynx_closed", "flag_underexposed", "mean_luma", True, 4),
            ("leopard", "flag_underexposed", "mean_luma", True, 4),
            ("turtle", "flag_underexposed", "mean_luma", True, 4),
        ),
    ),
    Category(
        "occlusion",
        "difficulty",
        "Occlusion",
        (
            ("leopard", "flag_fragmented_mask", "largest_component_fraction", True, 12),
            ("lynx_closed", "flag_fragmented_mask", "largest_component_fraction", True, 6),
            ("salamander", "sam3_n_instances > 1", "sam3_n_instances", False, 10),
        ),
    ),
]


class Example(NamedTuple):
    category: str
    dataset: str
    row: int
    note: str  # provenance; not drawn on the page


# Filled after the contact sheets have been reviewed by eye (see AGENTS.md "Data challenges").
EXAMPLES: List[Example] = [
    # Overexposure: washed-out flash or infrared frames (approved by the user 2026-10-02).
    Example("overexposure", "lynx_closed", 38308, "fully saturated IR frame (paper figure)"),
    Example("overexposure", "lynx_closed", 20211, "saturated IR frame"),
    Example("overexposure", "lynx_closed", 31806, "colour frame, flash blast on the animal, scene visible"),
    Example("overexposure", "lynx_closed", 36620, "saturated IR frame"),
    Example("overexposure", "leopard", 2824, "IR flash glare (paper figure)"),
    Example("overexposure", "leopard", 3800, "IR flash, coat nearly white"),
    Example("overexposure", "turtle", 6395, "shell washed out by flash"),
    # Empty frame: no visible animal.
    Example("empty_frame", "lynx_closed", 36426, "stick, no lynx (paper figure)"),
    Example("empty_frame", "lynx_closed", 16663, "log and corner glare, one frame of a repeated sequence"),
    Example("empty_frame", "lynx_closed", 10928, "forest scene, no animal"),
    Example("empty_frame", "lynx_closed", 6327, "snowy bushes, no animal"),
    Example("empty_frame", "lynx_closed", 36092, "snowy branches, no animal"),
    Example("empty_frame", "lynx_closed", 14688, "rock and snow, no animal"),
    # Insufficient detail: the animal is too small or featureless to carry its markings.
    Example(
        "insufficient_detail", "turtle", 11453, "person holding the turtle far away; stored rotated (paper figure)"
    ),
    Example("insufficient_detail", "turtle", 5638, "handler and ID tag, turtle a few pixels"),
    Example("insufficient_detail", "turtle", 5256, "people with the tag, turtle small"),
    Example("insufficient_detail", "turtle", 11220, "turtle small in a wide scene"),
    Example("insufficient_detail", "whale_shark", 614, "hazy backlit silhouette, no spot pattern (paper figure)"),
    Example("insufficient_detail", "whale_shark", 7214, "distant fin at the surface"),
    Example("insufficient_detail", "whale_shark", 3670, "distant shark below the surface"),
    # Corruption.
    Example("corruption", "lynx_closed", 2307, "colour banding (paper figure)"),
    Example("corruption", "lynx_closed", 28881, "posterised blue block"),
    # Blur.
    Example("blur", "hyena", 550, "no fine detail (paper figure)"),
    Example("blur", "hyena", 2606, "blurred IR frame"),
    Example("blur", "salamander", 249, "flash close-up out of focus (paper note)"),
    Example("blur", "salamander", 112, "flash close-up out of focus"),
    Example("blur", "salamander", 423, "flash close-up out of focus"),
    Example("blur", "turtle", 2697, "blurred shell"),
    Example("blur", "whale_shark", 5910, "soft spots through water"),
    # Night and infrared frames: identifiable, but a different appearance.
    Example("night_infrared", "hyena", 1356, "dark night frame, hyena identifiable"),
    Example("night_infrared", "hyena", 0, "dark night frame, hyena identifiable"),
    Example("night_infrared", "lynx_closed", 14870, "grey IR frame"),
    Example("night_infrared", "lynx_closed", 35567, "purple-cast IR frame"),
    Example("night_infrared", "lynx_closed", 18339, "dark colour frame, lynx on a rock"),
    Example("night_infrared", "lynx_closed", 18272, "IR close-up of the face"),
    Example("night_infrared", "leopard", 4742, "dark frame, leopard visible"),
    Example("night_infrared", "leopard", 4083, "dusk frame"),
    Example("night_infrared", "turtle", 12714, "dark shell"),
    # Occlusion: vegetation or the handler's hand in front of the pattern.
    Example("occlusion", "leopard", 698, "grass in front of the coat"),
    Example("occlusion", "leopard", 882, "foliage"),
    Example("occlusion", "leopard", 2047, "branches"),
    Example("occlusion", "leopard", 1124, "leaves"),
    Example("occlusion", "lynx_closed", 20994, "branches in front of the lynx"),
    Example("occlusion", "lynx_closed", 28185, "tree trunk and branches in front"),
    Example("occlusion", "salamander", 1238, "handler's finger splits the pattern (paper note)"),
    Example("occlusion", "salamander", 696, "handler's finger"),
    Example("occlusion", "salamander", 1274, "handler's finger"),
    Example("occlusion", "salamander", 542, "handler's finger"),
]


# --------------------------------------------------------------------------- audit + photos
def load_audit(dataset: str) -> pd.DataFrame:
    path = AUDIT_DIR / f"{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"audit CSV missing: {path}")
    table = pd.read_csv(path)
    table["grayscale"] = table["grayscale"].astype(bool)
    return table


def load_metadata(profile: PaperProfile) -> pd.DataFrame:
    return pd.read_csv(profile.metadata)


def raw_photo_path(profile: PaperProfile, record: pd.Series) -> Path:
    """The raw photo behind an audit row (same rule as plot_data_quality_examples.py)."""
    if "original_path" in record.index and isinstance(record.get("original_path"), str):
        relative = str(record["original_path"])
    else:
        relative = str(record["path"]).replace("masked_images/", "images/", 1)
    return Path(profile.root) / relative


def extra_columns(dataset: str, metadata: pd.DataFrame) -> pd.DataFrame:
    """Metadata columns merged into the audit table for expressions (Salamander SAM 3 fields)."""
    cols = [c for c in ("sam3_n_instances", "sam3_score", "original_path", "path") if c in metadata.columns]
    frame = metadata[cols].copy()
    frame["row_index"] = range(len(frame))
    return frame


def rank_candidates(category: Category) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dataset, expression, metric, ascending, count in category.sources:
        profile = PROFILES[dataset]
        audit = load_audit(dataset)
        metadata = load_metadata(profile)
        # The audit keeps the original metadata row index and drops rows outside the two split sides
        # (490 unlabeled ZindiTurtleRecall rows), so check containment, not equal length.
        if int(audit["row_index"].max()) >= len(metadata):
            raise ValueError(f"{dataset}: audit row_index exceeds metadata length {len(metadata)}")
        merged = audit.merge(extra_columns(dataset, metadata), on="row_index", suffixes=("", "_meta"))
        selected = merged[merged.eval(expression)].sort_values(metric, ascending=ascending).head(count)
        for _, record in selected.iterrows():
            meta_record = metadata.iloc[int(record["row_index"])]
            rows.append(
                {
                    "dataset": dataset,
                    "row": int(record["row_index"]),
                    "metric": metric,
                    "value": record[metric],
                    "side": record["side"],
                    "photo": raw_photo_path(profile, meta_record),
                }
            )
    return rows


def contact_sheet(items: Sequence[Dict[str, Any]], output: Path, columns: int = 6, cell: int = 300) -> None:
    rows_n = max(1, (len(items) + columns - 1) // columns)
    label_h = 44
    sheet = Image.new("RGB", (columns * cell, rows_n * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(items):
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_h)
        try:
            with Image.open(item["photo"]) as img:
                photo = ImageOps.exif_transpose(img).convert("RGB")
                photo.thumbnail((cell - 8, cell - 8))
                sheet.paste(photo, (x + 4 + (cell - 8 - photo.width) // 2, y + 4 + (cell - 8 - photo.height) // 2))
        except OSError:
            draw.rectangle((x + 4, y + 4, x + cell - 4, y + cell - 4), outline="red")
        value = item["value"]
        text = (
            f"#{index}  {item['dataset']}:{item['row']} ({item['side']})\n{item['metric']}={value:.3g}"
            if isinstance(value, (int, float))
            else f"#{index}  {item['dataset']}:{item['row']}"
        )
        draw.multiline_text((x + 6, y + cell + 2), text, fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=85)


def command_candidates(args: argparse.Namespace) -> None:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for category in CATEGORIES:
        if args.categories and category.key not in args.categories:
            continue
        items = rank_candidates(category)
        contact_sheet(items, args.sheet_dir / f"{category.key}.jpg")
        index[category.key] = [
            {
                k: (
                    str(v)
                    if isinstance(v, Path)
                    else (float(v) if hasattr(v, "__float__") and not isinstance(v, str) else v)
                )
                for k, v in item.items()
            }
            for item in items
        ]
        print(
            f"[data-challenges] {category.key}: {len(items)} candidates -> {args.sheet_dir / (category.key + '.jpg')}"
        )
    (args.sheet_dir / "candidates.json").write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- render
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_photo(source: Path, output: Path) -> Tuple[int, int]:
    with Image.open(source) as img:
        photo = ImageOps.exif_transpose(img).convert("RGB")
        photo.thumbnail((WEB_LONG_SIDE, WEB_LONG_SIDE))
        output.parent.mkdir(parents=True, exist_ok=True)
        photo.save(output, "JPEG", quality=86, optimize=True, progressive=True)
        return photo.size


def summary_payload(summary_csv: Path) -> Dict[str, Any]:
    table = pd.read_csv(summary_csv)
    rows = []
    for _, r in table.iterrows():
        if r["dataset"] not in PAGE_DATASETS:
            continue
        rows.append(
            {
                k: (
                    None
                    if pd.isna(v)
                    else (
                        int(v)
                        if isinstance(v, (int,))
                        or (
                            hasattr(v, "is_integer")
                            and float(v).is_integer()
                            and "fraction" not in k
                            and "rate" not in k
                            and "cut" not in k
                            and "top1" not in k
                        )
                        else (float(v) if isinstance(v, float) else v)
                    )
                )
                for k, v in r.items()
            }
        )
    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_data_challenges.py",
        "source": "experiments/image-quality/summary.csv",
        "source_sha256": sha256_file(summary_csv),
        "flag_rules": json.loads((AUDIT_DIR / "manifest.json").read_text(encoding="utf-8"))["flag_rules"],
        "note": "Flags are heuristic rankings on the model input; pre-masked WildlifeReID-10k files use a brightness "
        "threshold for the foreground, so very dark animals can be flagged as tiny or empty. query_top1 columns "
        "average each flagged or clean query's Top-1 over all completed runs of all methods.",
        "rows": rows,
    }


def command_render(args: argparse.Namespace) -> None:
    if not EXAMPLES:
        raise SystemExit("EXAMPLES is empty: review the candidate sheets and fill it before rendering")
    audits = {d: load_audit(d) for d in {e.dataset for e in EXAMPLES}}
    metadata = {d: load_metadata(PROFILES[d]) for d in audits}
    items: List[Dict[str, Any]] = []
    for example in EXAMPLES:
        category = next(c for c in CATEGORIES if c.key == example.category)
        audit_row = audits[example.dataset].loc[audits[example.dataset]["row_index"] == example.row]
        if len(audit_row) != 1:
            raise ValueError(f"{example.dataset} row {example.row}: not exactly one audit row")
        record = audit_row.iloc[0]
        meta_record = metadata[example.dataset].iloc[example.row]
        source = raw_photo_path(PROFILES[example.dataset], meta_record)
        if not source.is_file():
            raise FileNotFoundError(source)
        file_name = f"{example.dataset}_{example.row}.jpg"
        width, height = export_photo(source, args.web_dir / file_name)
        measurements = {
            k: (None if pd.isna(record[k]) else (bool(record[k]) if k == "grayscale" else float(record[k])))
            for k in METRIC_COLUMNS
            if k in record.index
        }
        items.append(
            {
                "category": category.key,
                "group": category.group,
                "title": category.title,
                "dataset": example.dataset,
                "dataset_label": PROFILES[example.dataset].label,
                "row": example.row,
                "side": record["side"],
                "identity": str(record["identity"]),
                "note": example.note,
                "flags": [f for f in str(record["flags"]).split(";") if f and f != "nan"],
                "measurements": measurements,
                "query_runs": None if pd.isna(record.get("query_runs")) else int(record["query_runs"]),
                "query_top1_rate": None if pd.isna(record.get("query_top1_rate")) else float(record["query_top1_rate"]),
                "sam3_n_instances": (
                    int(meta_record["sam3_n_instances"]) if "sam3_n_instances" in meta_record.index else None
                ),
                "image": {"file": file_name, "width": width, "height": height},
                "source_sha256": sha256_file(source),
            }
        )
    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "generated_by": "paper/page/export_data_challenges.py",
        "categories": [{"key": c.key, "group": c.group, "title": c.title} for c in CATEGORIES],
        "attribution": "Raw photographs from CzechLynx (Picek et al.), WildlifeReID-10k and SalamanderID2025 "
        "(AnimalCLEF 2025), the latter shown with the dataset team's permission.",
        "items": items,
    }
    text = json.dumps(payload, indent=1)
    for fragment in ("/shared/", "/home/"):
        if fragment in text:
            raise ValueError(f"private path fragment {fragment!r} in export")
    (args.web_dir / "challenges.json").write_text(text + "\n", encoding="utf-8")
    summary = summary_payload(AUDIT_DIR / "summary.csv")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"[data-challenges] wrote {len(items)} examples to {args.web_dir} and {args.summary_out}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cand = sub.add_parser("candidates", help="write review contact sheets per category")
    cand.add_argument("--sheet-dir", type=Path, default=SHEET_DIR)
    cand.add_argument("--categories", nargs="*", default=None)
    cand.set_defaults(func=command_candidates)
    rend = sub.add_parser("render", help="export the confirmed EXAMPLES and the audit summary")
    rend.add_argument("--web-dir", type=Path, default=WEB_DIR)
    rend.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    rend.set_defaults(func=command_render)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
