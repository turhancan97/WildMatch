#!/usr/bin/env python3
"""Build the CPU demo data shipped in ``src/wildmatch/demo/`` (run once; kept for provenance).

Takes six synthetic individuals with at least four renders each from the CzechLynx synthetic
subset (Picek et al., Zenodo record 17592004, CC BY 4.0), chosen by seeded random sampling,
applies each render's own mask (background black, as the paper's inputs), resizes to a 512 px
long side, and writes ``images/``, ``metadata.csv`` (three database renders and one query per
individual) and ``ATTRIBUTION.md``. Synthetic coats share one texture model, so the demo shows
that the pipeline runs; it is not a benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

ATTRIBUTION = (
    "Synthetic lynx renders from the CzechLynx synthetic subset (Picek et al.), "
    "Zenodo record 17592004, CC BY 4.0. Masked and resized for the WildMatch CPU demo."
)


def build(root: Path, output: Path, individuals: int = 6, per_individual: int = 4, seed: int = 0) -> pd.DataFrame:
    table = pd.read_csv(root / "CzechLynxDataset-Metadata-Synthetic.csv")
    counts = table["unique_name"].value_counts()
    eligible = sorted(counts[counts >= per_individual].index)
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(eligible, size=individuals, replace=False))
    rows = []
    (output / "images").mkdir(parents=True, exist_ok=True)
    for identity in chosen:
        picks = table[table["unique_name"] == identity].sample(per_individual, random_state=seed)
        for n, record in enumerate(picks.itertuples()):
            image = np.asarray(Image.open(root / record.path).convert("RGB")).copy()
            rle = json.loads(record.mask)
            mask = mask_utils.decode(rle).astype(bool)
            image[~mask] = 0
            picture = Image.fromarray(image)
            picture.thumbnail((512, 512))
            name = f"{identity}_{n}.jpg"
            picture.save(output / "images" / name, quality=90)
            rows.append(
                {
                    "identity": identity,
                    "path": f"images/{name}",
                    "split": "query" if n == per_individual - 1 else "database",
                    "source_path": record.path,
                }
            )
    metadata = pd.DataFrame(rows)
    metadata.to_csv(output / "metadata.csv", index=False)
    (output / "ATTRIBUTION.md").write_text(f"# Demo images\n\n{ATTRIBUTION}\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="CzechLynx folder with the synthetic metadata")
    parser.add_argument("--output", type=Path, default=Path("src/wildmatch/demo"))
    args = parser.parse_args()
    metadata = build(args.root, args.output)
    print(metadata.groupby("split").size().to_dict(), "individuals:", metadata["identity"].nunique())


if __name__ == "__main__":
    main()
