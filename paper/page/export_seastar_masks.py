#!/usr/bin/env python3
"""Export the project page's Sea star mask figure (CPU): one test query whose paper-input mask kept almost
nothing of the photo, shown as the original photo, the paper's masked input and the SAM 3 masked input.

The candidates are the SeaStarReID2023 test queries whose paper-input file (``registry.paper_inputs``,
``masked_images/``) keeps less than ``--max-foreground`` of its pixels (foreground = max(RGB) > 12, the
image-quality audit's rule for pre-masked files); one is picked by seeded random sampling, never by score.
Writes ``docs/assets/beyond/seastar_masks.jpg`` and ``seastar_masks.json`` (candidate count, chosen file,
foreground shares; dataset-relative paths only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "assets" / "beyond"


def foreground_share(path: Path) -> float:
    with Image.open(path) as handle:
        rgb = np.asarray(handle.convert("RGB"))
    return float((rgb.max(axis=2) > 12).mean())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-foreground", type=float, default=0.01)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    import pandas as pd

    from wildmatch.data.registry import load_dataset

    entry = load_dataset("seastarreid2023")
    root = Path(str(entry.root))
    paper = pd.read_csv(root / str(entry.registry.paper_inputs.metadata_file))
    sam3 = pd.read_csv(root / str(entry.metadata_file))
    split_col, query = str(entry.split_col), str(entry.query_split_value)
    queries = paper[paper[split_col].astype(str) == query]
    # Both tables name the same release file after their first folder (masked_images/ or masked_images_sam3/);
    # the original photo is that file under images/.
    sam3_by_key = {str(row["path"]).split("/", 1)[1]: str(row["path"]) for row in sam3.to_dict("records")}
    candidates = []
    for row in queries.to_dict("records"):
        old = str(row["path"])
        key = old.split("/", 1)[1]
        raw = "images/" + key
        if key not in sam3_by_key or not (root / raw).is_file():
            continue
        share = foreground_share(root / old)
        if share < args.max_foreground:
            candidates.append((old, raw, sam3_by_key[key], share))
    if not candidates:
        raise SystemExit("no candidate query found")
    rng = np.random.default_rng(args.seed)
    old, raw, new, share = candidates[int(rng.integers(len(candidates)))]

    panels = []
    for rel in (raw, old, new):
        with Image.open(root / rel) as handle:
            image = handle.convert("RGB")
        scale = args.height / image.height
        panels.append(image.resize((max(1, round(image.width * scale)), args.height), Image.LANCZOS))
    gutter = 8
    canvas = Image.new("RGB", (sum(p.width for p in panels) + gutter * 2, args.height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gutter
    args.out.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out / "seastar_masks.jpg", quality=88, optimize=True, progressive=True)
    payload = {
        "generated_by": "paper/page/export_seastar_masks.py",
        "selection": f"seeded random (seed {args.seed}) among {len(candidates)} test queries whose paper-input "
        f"mask keeps < {args.max_foreground:.0%} of the photo",
        "candidates": len(candidates),
        "chosen": {"raw": raw, "paper_input": old, "sam3_input": new},
        "foreground_share": {"paper_input": round(share, 5), "sam3_input": round(foreground_share(root / new), 5)},
    }
    (args.out / "seastar_masks.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
