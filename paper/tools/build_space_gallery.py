#!/usr/bin/env python3
"""Build the Hugging Face Space's galleries and examples (GPU recommended for the embeddings; run once).

Two galleries, both CC BY 4.0, so they may be redistributed with attribution (the Space must not carry
WildlifeReID-10k, SalamanderID2025 or JaguarReID photos; see AGENTS.md, "Future-work checklist"):

``czechlynx_unseen``
    The database side of the CzechLynx unseen-individual split (registry entry ``czechlynx_unseen_eval``,
    160 photos of 44 lynx), the paper's unseen-identity protocol: none of these individuals was in the
    fine-tuning data of the CzechLynx open checkpoints the Space uses for this gallery. Model inputs are the
    photos with the dataset's own masks applied (black background), as in every CzechLynx run; the raw photos
    are kept for drawing.
``synthetic``
    The 24 synthetic lynx renders bundled with ``wildmatch demo`` (18 gallery renders, 6 queries); they are
    already masked, so the masked render is also drawn. Not a benchmark: synthetic coats share one texture.

Example queries are chosen by seeded random sampling from the unseen split's query side (never by score),
plus one same-lynx day/infrared pair picked by appearance for the pair tab. Writes ``space/gallery/<name>/``
(``metadata.csv``, ``raw/``, ``masked/``, ``embeddings.npz``, ``ATTRIBUTION.md``) and ``space/examples/``;
refuses to replace existing outputs without ``--overwrite``. Only dataset-relative paths are written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACE = REPO_ROOT / "space"
LONG_SIDE = 640
CZECHLYNX_LICENCE = "CC BY 4.0"
CZECHLYNX_CREDIT = "CzechLynx dataset (Picek et al.; Kaggle picekl/czechlynx), CC BY 4.0"
SYNTHETIC_CREDIT = "CzechLynx synthetic subset (Picek et al.; Zenodo record 17592004), CC BY 4.0"
# The pair example: one lynx in a colour frame and an infrared night frame, chosen by appearance (2026-10-10).
PAIR_EXAMPLE = ("CzechLynx/snpa/lynx_096/37856_lynx_096.jpg", "CzechLynx/snpa/lynx_096/38991_lynx_096.jpg")


def thumbnail(image: Image.Image, long_side: int = LONG_SIDE) -> Image.Image:
    scale = long_side / max(image.size)
    if scale >= 1:
        return image.copy()
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def masked_photo(image: Image.Image, mask: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    if mask.shape != array.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match photo {array.shape[:2]}")
    array[~mask] = 0
    return Image.fromarray(array)


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90, optimize=True)


def write_embeddings(folder: Path, masked: Sequence[Path]) -> None:
    """MegaDescriptor-L embeddings of the stored masked photos, computed exactly as the Space embeds a query."""
    sys.path.insert(0, str(SPACE))
    from wildmatch_space import retrieval

    rows = []
    for start in range(0, len(masked), 16):
        rows.append(retrieval.embed([Image.open(p).convert("RGB") for p in masked[start : start + 16]]))
    np.savez_compressed(folder / "embeddings.npz", embeddings=np.concatenate(rows).astype(np.float32))


def build_unseen(out: Path, examples: Path, n_queries: int, seed: int) -> List[Path]:
    from wildmatch.data.prepare.sam3_masks import decode_mask
    from wildmatch.data.registry import load_dataset

    entry = load_dataset("czechlynx_unseen_eval")
    root = Path(str(entry.root))
    frame = pd.read_csv(root / str(entry.metadata_file))
    split = frame[str(entry.split_col)].astype(str)
    database = frame[split == str(entry.database_split_value)].reset_index(drop=True)
    queries = frame[split == str(entry.query_split_value)].reset_index(drop=True)
    rows, masked_paths = [], []
    for index, row in database.iterrows():
        raw = Image.open(root / row["path"]).convert("RGB")
        masked = masked_photo(raw, decode_mask(row[str(entry.mask_col)]))
        name = f"{index:03d}.jpg"
        save_jpeg(thumbnail(raw), out / "raw" / name)
        save_jpeg(thumbnail(masked), out / "masked" / name)
        masked_paths.append(out / "masked" / name)
        rows.append(
            {
                "identity": row["unique_name"],
                "raw": f"raw/{name}",
                "masked": f"masked/{name}",
                "source": f"CzechLynx: {row['path']}",
                "licence": CZECHLYNX_LICENCE,
            }
        )
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    (out / "ATTRIBUTION.md").write_text(
        f"# CzechLynx unseen individuals\n\n{CZECHLYNX_CREDIT}. The database side of the CzechLynx unseen-"
        "individual evaluation split (160 photos of 44 lynx), resized to a 640 px long side; `masked/` applies "
        "the dataset's own masks.\n",
        encoding="utf-8",
    )
    rng = np.random.default_rng(seed)
    chosen = sorted(int(i) for i in rng.choice(len(queries), size=n_queries, replace=False))
    example_rows = []
    for k, index in enumerate(chosen):
        row = queries.iloc[index]
        name = f"lynx_query_{k}.jpg"
        save_jpeg(thumbnail(Image.open(root / row["path"]).convert("RGB")), examples / name)
        example_rows.append({"file": name, "identity": row["unique_name"], "source": f"CzechLynx: {row['path']}"})
    for k, rel in enumerate(PAIR_EXAMPLE):
        name = f"lynx_pair_{k}.jpg"
        save_jpeg(thumbnail(Image.open(root / rel).convert("RGB")), examples / name)
        identity = rel.split("/")[2]
        example_rows.append({"file": name, "identity": identity, "source": f"CzechLynx: {rel}"})
    pd.DataFrame(example_rows).to_csv(examples / "examples.csv", index=False)
    (examples / "ATTRIBUTION.md").write_text(
        f"# Example photos\n\n{CZECHLYNX_CREDIT}. Queries chosen by seeded random sampling (seed {seed}) from the "
        "unseen-individual split's query side; the pair (one lynx, colour and infrared) chosen by appearance.\n",
        encoding="utf-8",
    )
    return masked_paths


def build_synthetic(out: Path) -> List[Path]:
    demo = REPO_ROOT / "src" / "wildmatch" / "demo"
    frame = pd.read_csv(demo / "metadata.csv")
    rows, masked_paths = [], []
    for index, row in frame[frame["split"] == "database"].reset_index(drop=True).iterrows():
        name = f"{index:03d}.jpg"
        for kind in ("raw", "masked"):
            (out / kind).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(demo / row["path"], out / kind / name)
        masked_paths.append(out / "masked" / name)
        rows.append(
            {
                "identity": row["identity"],
                "raw": f"raw/{name}",
                "masked": f"masked/{name}",
                "source": f"CzechLynx synthetic: {row['source_path']}",
                "licence": CZECHLYNX_LICENCE,
            }
        )
    pd.DataFrame(rows).to_csv(out / "metadata.csv", index=False)
    queries = frame[frame["split"] == "query"].reset_index(drop=True)
    for k, row in queries.iterrows():
        shutil.copyfile(demo / row["path"], out.parent.parent / "examples" / f"synthetic_query_{k}.jpg")
    (out / "ATTRIBUTION.md").write_text(
        f"# Synthetic lynx\n\n{SYNTHETIC_CREDIT}. The renders bundled with `wildmatch demo` (masked, 512 px). "
        "Not a benchmark: synthetic coats share one texture model.\n",
        encoding="utf-8",
    )
    return masked_paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--queries", type=int, default=4, help="example queries from the unseen split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    targets = [SPACE / "gallery", SPACE / "examples"]
    existing = [t for t in targets if t.exists() and any(t.iterdir())]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to replace {', '.join(map(str, existing))} without --overwrite")
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    unseen = SPACE / "gallery" / "czechlynx_unseen"
    write_embeddings(unseen, build_unseen(unseen, SPACE / "examples", args.queries, args.seed))
    synthetic = SPACE / "gallery" / "synthetic"
    write_embeddings(synthetic, build_synthetic(synthetic))
    summary = {
        name: int(len(pd.read_csv(SPACE / "gallery" / name / "metadata.csv")))
        for name in ("czechlynx_unseen", "synthetic")
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
