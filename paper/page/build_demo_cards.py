#!/usr/bin/env python
"""Build the card thumbnails of the project page's Demo hub (CPU only).

Each card is a 480 x 300 JPEG under ``docs/assets/demo/cards/`` composed from photos
that the demo exports already committed under ``docs/assets/demo/<demo>/``; no dataset
access, model, or GPU is needed, and nothing is re-matched. The hub page
(``docs/demo/index.md``) shows one card per demo page.

Cards (in the hub's narrative order):

* ``before-after``: the Czech Lynx query and its top-1 photo from the before/after demo.
* ``rank-changes``: a rescued query followed by its fine-tuned top-3 gallery photos.
* ``mined-pairs``: an anchor with two positives (blue rule) and two hard negatives (red).
* ``masking``: a synthetic render, raw on the left and SAM 3-masked on the right.
* ``synthetic``: the two renders of one synthetic lynx.

Run ``python paper/page/build_demo_cards.py`` after re-exporting a demo; the script fails
closed when a referenced export is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "docs" / "assets" / "demo"
CARD_DIR = DEMO_ROOT / "cards"
CARD_SIZE = (480, 300)
GUTTER = 6
BLUE = (58, 126, 171)
RED = (207, 72, 50)
GREY = (209, 211, 212)
CARD_NAMES = ("before-after", "rank-changes", "mined-pairs", "masking", "synthetic")


def cover(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize ``image`` to fill ``size`` and centre-crop the overflow (CSS ``object-fit: cover``)."""
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def compose_row(
    images: Sequence[Image.Image],
    size: Tuple[int, int] = CARD_SIZE,
    gutter: int = GUTTER,
    rules: Sequence[Tuple[int, int, int] | None] | None = None,
    rule_height: int = 6,
) -> Image.Image:
    """Lay ``images`` side by side in equal columns; ``rules`` draws a coloured bar under a column."""
    if not images:
        raise ValueError("compose_row needs at least one image")
    width, height = size
    canvas = Image.new("RGB", size, GREY)
    columns = len(images)
    column_width = (width - gutter * (columns - 1)) // columns
    x = 0
    for index, image in enumerate(images):
        rule = rules[index] if rules else None
        cell_height = height - (rule_height if rule else 0)
        canvas.paste(cover(image.convert("RGB"), (column_width, cell_height)), (x, 0))
        if rule:
            ImageDraw.Draw(canvas).rectangle((x, height - rule_height, x + column_width - 1, height - 1), fill=rule)
        x += column_width + gutter
    if x - gutter < width:  # fill the rounding remainder with the last column's edge colour
        ImageDraw.Draw(canvas).rectangle((x - gutter, 0, width - 1, height - 1), fill=GREY)
    return canvas


def compose_split(
    raw: Image.Image, mask: Image.Image, size: Tuple[int, int] = CARD_SIZE, divider: Tuple[int, int, int] = RED
) -> Image.Image:
    """Left half raw photo, right half the masked model input (black outside the mask)."""
    raw = cover(raw.convert("RGB"), size)
    mask = cover(mask.convert("L"), size).point(lambda v: 255 if v > 127 else 0)
    masked = Image.composite(raw, Image.new("RGB", size, (0, 0, 0)), mask)
    half = size[0] // 2
    canvas = raw.copy()
    canvas.paste(masked.crop((half, 0, size[0], size[1])), (half, 0))
    ImageDraw.Draw(canvas).rectangle((half - 1, 0, half + 1, size[1] - 1), fill=divider)
    return canvas


def _load_json(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"demo export missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _open(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"demo photo missing: {path}")
    return Image.open(path)


def _photo_file(record: Dict) -> str:
    """Exports store web photos as ``{"image": {"file": ...}}`` or ``{"file": ...}``."""
    image = record.get("image", record)
    if isinstance(image, dict) and "file" in image:
        return str(image["file"])
    raise KeyError(f"no photo file in record keys {sorted(record)}")


def card_before_after(root: Path = DEMO_ROOT) -> Image.Image:
    data = _load_json(root / "before_after" / "before_after.json")
    pair = next((p for p in data["pairs"] if p.get("dataset") == "lynx_closed"), data["pairs"][0])
    folder = root / "before_after"
    return compose_row([_open(folder / _photo_file(pair["query"])), _open(folder / _photo_file(pair["gallery"]))])


def card_rank_changes(root: Path = DEMO_ROOT) -> Image.Image:
    data = _load_json(root / "rank_change" / "rank_change.json")
    query = next((q for q in data["queries"] if q.get("category") == "rescued"), data["queries"][0])
    photos = data["photos"]
    folder = root / "rank_change"
    top = query["rankings"]["finetuned"]["top5"][:3]
    tags = [query["tag"]] + [entry["tag"] for entry in top]
    images = [_open(folder / _photo_file(photos[tag])) for tag in tags]
    rules: List[Tuple[int, int, int] | None] = [BLUE] + [BLUE if entry.get("correct") else None for entry in top]
    return compose_row(images, rules=rules)


def card_mined_pairs(root: Path = DEMO_ROOT) -> Image.Image:
    data = _load_json(root / "mined_pairs" / "mined_pairs.json")
    anchor = data["anchors"][0]
    folder = root / "mined_pairs"
    records = [anchor["anchor"]] + anchor["positives"][:2] + anchor["negatives"][:2]
    images = [_open(folder / _photo_file(record)) for record in records]
    return compose_row(images, rules=[None, BLUE, BLUE, RED, RED])


def card_masking(root: Path = DEMO_ROOT, tag: str = "G_lynx_173") -> Image.Image:
    folder = root / "masking"
    return compose_split(_open(folder / f"{tag}.jpg"), _open(folder / f"{tag}_sam3.png"))


def card_synthetic(root: Path = DEMO_ROOT) -> Image.Image:
    data = _load_json(root / "synthetic" / "synthetic_demo.json")
    folder = root / "synthetic"
    query = data["queries"][0]
    gallery = next(g for g in data["gallery"] if g["identity"] == query["identity"])
    return compose_row([_open(folder / _photo_file(query)), _open(folder / _photo_file(gallery))])


BUILDERS = {
    "before-after": card_before_after,
    "rank-changes": card_rank_changes,
    "mined-pairs": card_mined_pairs,
    "masking": card_masking,
    "synthetic": card_synthetic,
}


def build_cards(output_dir: Path = CARD_DIR, root: Path = DEMO_ROOT, quality: int = 84) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in CARD_NAMES:
        image = BUILDERS[name](root)
        if image.size != CARD_SIZE:
            raise RuntimeError(f"{name}: card size {image.size} != {CARD_SIZE}")
        path = output_dir / f"{name}.jpg"
        image.save(path, "JPEG", quality=quality, optimize=True, progressive=True)
        written.append(path)
        print(f"[demo-cards] wrote {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=CARD_DIR)
    parser.add_argument("--demo-root", type=Path, default=DEMO_ROOT)
    args = parser.parse_args(argv)
    build_cards(args.output_dir, args.demo_root)


if __name__ == "__main__":
    main()
