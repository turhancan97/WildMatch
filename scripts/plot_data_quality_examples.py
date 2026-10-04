#!/usr/bin/env python
"""Plot confirmed low-quality dataset examples as raw source photos.

Every example below was flagged by ``scripts/audit_image_quality.py`` and then
confirmed by eye against the raw source file. Only raw photos are shown: masked
model inputs are left out because background removal (partly our own SAM3 runs)
could be read as our error rather than a property of the source data. Every panel
must therefore show a problem visible in the raw photo itself. Edit ``EXAMPLES``
only after confirming a new image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]

from wildmatch.reporting.paper_datasets import PAPER_PROFILES  # noqa: E402


class Example(NamedTuple):
    group: str
    dataset: str
    row: int
    note: str  # provenance only; not drawn
    # Trim the black padding baked into the source file, then centre-crop to a square
    # so the photo fills its panel. Only for padding, never to hide image content.
    fill: bool = False


# Raw photos confirmed by eye on 2026-09-23.
EXAMPLES = [
    Example("(a) Overexposure", "lynx_closed", 38308, "fully saturated"),
    Example("(a) Overexposure", "leopard", 2824, "IR flash glare"),
    Example("(b) Insufficient detail", "turtle", 11453, "turtle a few dozen pixels at model input size"),
    Example("(b) Insufficient detail", "whale_shark", 614, "hazy backlit silhouette, no spot pattern"),
    Example("(c) Empty frame", "lynx_closed", 36426, "stick, no lynx", fill=True),
    Example("(d) Corruption", "lynx_closed", 2307, "colour banding"),
    Example("(e) Blur", "hyena", 550, "no fine detail"),
]
DATASET_NAMES = {
    "lynx_closed": "CzechLynx",
    "leopard": "LeopardID2022",
    "hyena": "HyenaID2022",
    "sea_star": "SeaStarReID2023",
    "whale_shark": "WhaleSharkID",
    "salamander": "SalamanderID2025",
    "turtle": "ZindiTurtleRecall",
    "nyala": "NyalaData",
}
THUMBNAIL = 480
PADDING_THRESHOLD = 16


def raw_image_path(profile, frame: pd.DataFrame, row: int) -> Path:
    """Unmasked source photo for a metadata row."""
    record = frame.iloc[row]
    if "original_path" in frame.columns:
        # SalamanderID2025 records the unmasked source explicitly.
        relative = str(record["original_path"])
    elif profile.mask_col is None:
        # Pre-masked WildlifeReID-10k files mirror the raw tree under images/.
        relative = str(record["path"]).replace("masked_images/", "images/", 1)
    else:
        # CzechLynx paths are raw images; its mask is applied only at load time.
        relative = str(record["path"])
    path = Path(relative)
    return path if path.is_absolute() else profile.root / path


def trim_and_fill(image: Image.Image) -> Image.Image:
    """Drop near-black source padding, then centre-crop the content to a square."""
    content = np.asarray(image).max(axis=2) > PADDING_THRESHOLD
    ys, xs = np.nonzero(content)
    if len(xs):
        image = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def square(image: Image.Image) -> Image.Image:
    """Letterbox onto a white square so every panel has identical geometry."""
    scale = THUMBNAIL / max(image.size)
    image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (THUMBNAIL, THUMBNAIL), (255, 255, 255))
    canvas.paste(image, ((THUMBNAIL - image.width) // 2, (THUMBNAIL - image.height) // 2))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--width", type=float, default=6.875, help="Figure width in inches (CVPR full text width)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Times to match the CVPR body text; STIX (bundled with matplotlib) is a
    # Times-compatible fallback when Times New Roman is not installed.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    profiles = {profile.key: profile for profile in PAPER_PROFILES}
    frames: dict = {}
    columns = len(EXAMPLES)
    left_margin, right_margin, gap = 0.01, 0.01, 0.012
    cell = args.width * (1 - left_margin - right_margin - gap * (columns - 1)) / columns
    # Inches for the group label + dataset name; no footer, the group label says it all.
    header, footer = 0.42, 0.03
    height = cell + header + footer
    figure, axes = plt.subplots(1, columns, figsize=(args.width, height), squeeze=False)
    axes = axes[0]
    figure.subplots_adjust(
        left=left_margin, right=1 - right_margin,
        top=1 - header / height, bottom=footer / height,
        wspace=gap * columns / (1 - left_margin - right_margin - gap * (columns - 1)),
    )
    for axis, example in zip(axes, EXAMPLES):
        profile = profiles[example.dataset]
        frame = frames.setdefault(example.dataset, pd.read_csv(profile.metadata, low_memory=False))
        with Image.open(raw_image_path(profile, frame, example.row)) as handle:
            image = handle.convert("RGB")
            if example.fill:
                image = trim_and_fill(image)
            axis.imshow(square(image))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
        axis.set_title(DATASET_NAMES[example.dataset], fontsize=7.5, pad=2)

    # Group labels centred over each run of same-group columns. A label wider than
    # its columns wraps onto two lines instead of running into its neighbour.
    rule_y = 1 - 0.2 / height
    renderer = figure.canvas.get_renderer()
    start = 0
    for index in range(1, columns + 1):
        if index == columns or EXAMPLES[index].group != EXAMPLES[start].group:
            left = axes[start].get_position().x0
            right = axes[index - 1].get_position().x1
            label = figure.text((left + right) / 2, rule_y + 0.035 / height, EXAMPLES[start].group,
                                ha="center", va="bottom", fontsize=8, fontweight="bold", linespacing=1.1)
            span = (right - left) * figure.bbox.width
            if label.get_window_extent(renderer).width > span * 0.96:
                words = EXAMPLES[start].group.split(" ")
                label.set_text(" ".join(words[:-1]) + "\n" + words[-1])
            figure.add_artist(plt.Line2D([left + 0.004, right - 0.004], [rule_y, rule_y], color="0.35", linewidth=0.6))
            start = index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        output = args.output_dir / f"data_quality_examples.{fmt}"
        figure.savefig(output, dpi=600 if fmt == "png" else 300)
        print(f"[data-quality] {output}")


if __name__ == "__main__":
    main()
