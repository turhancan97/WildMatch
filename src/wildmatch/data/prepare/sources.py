"""Rebuild the prepared dataset tables from the published sources.

The rules were recovered from the tables the paper's runs read (2026-10-04) and reproduce them:

WildlifeReID-10k (``metadata.csv`` of the Kaggle release ``wildlifedatasets/wildlifereid-10k``)
    Per sub-dataset, in the release's row order: identities without the ``<dataset>_`` prefix
    (as integers when all are numeric, which changes the split order), and a closed-set split from wildlife-datasets, ``ClosedSetSplit(0.8, seed=666)``, which put
    the same images in ``train``/``test`` as the teammate's tables for all twelve sub-datasets.
    BelugaID keeps only its ``beluga/`` folder (not ``beluga-id-test/``). The old Zindi table also
    listed 490 images with identity ``unknown`` and no split; they are not in the release, no run
    used them, and leaving them out changes no split. Masks are new (SAM 3, prompt from the
    ``species`` column): the teammate's masking method is not recorded, so masked inputs, and with
    them scores, can differ from the paper's (compare with ``wildmatch prepare compare-masks``).

SalamanderID2025 (``metadata.csv`` of the AnimalCLEF2025 Kaggle competition)
    The labelled ``database`` photos of ``SalamanderID2025`` that have a date (four undated ones
    are dropped). For every individual photographed on two or more dates, all photos of its
    latest date become ``query``, the rest ``database``; ``cross_view`` marks a query whose
    orientation does not occur among that individual's database photos; rows are sorted by split,
    identity, date and image id. Reproduces ``split_time_closed.csv`` byte for byte (SHA-256
    ``0b832fc8...``); the images are copied unchanged into ``database/images`` and ``query/images``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

SPLIT_RATIO = 0.8
SPLIT_SEED = 666
# SAM 3 prompts for WildlifeReID-10k species values that are not plain nouns.
SPECIES_PROMPTS = {"whale": "beluga whale", "whaleshark": "whale shark"}


class SourceError(ValueError):
    """The source data does not have what the rebuild needs."""


def wildlifereid10k_table(metadata: pd.DataFrame, animal: str, include: Optional[str] = None) -> pd.DataFrame:
    """``identity, path, split, species, prompt`` for one sub-dataset (``path`` relative to ``images/``)."""
    from wildlife_datasets import splits

    rows = metadata[metadata["dataset"] == animal]
    if include:
        rows = rows[rows["path"].str.contains(include, regex=False)]
    if rows.empty:
        raise SourceError(f"no rows for {animal} in the WildlifeReID-10k metadata")
    rows = rows.reset_index(drop=True)
    prefix = f"{animal}_"
    identity = rows["identity"].astype(str)
    if not identity.str.startswith(prefix).all():
        raise SourceError(f"{animal} identities do not all start with {prefix!r}")
    identity = identity.str[len(prefix):]
    if identity.str.fullmatch(r"\d+").all():
        # Numeric identities were split as integers (ATRW, CowDataset, NyalaData); the
        # splitter sorts identities, and string order ("10" < "2") gives another split.
        identity = identity.astype(int)
    frame = pd.DataFrame({"identity": identity, "image_id": range(len(rows))})
    train, test = splits.ClosedSetSplit(SPLIT_RATIO, seed=SPLIT_SEED).split(frame)[0]
    split = pd.Series([None] * len(rows), dtype=object)
    split.iloc[train] = "train"
    split.iloc[test] = "test"
    if split.isna().any():
        raise SourceError(f"{int(split.isna().sum())} {animal} rows got no split")
    species = rows["species"].astype(str)
    return pd.DataFrame({
        "identity": identity,
        "path": rows["path"].astype(str).str.replace(r"^images/", "", regex=True),
        "split": split,
        "species": species,
        "prompt": species.map(lambda s: SPECIES_PROMPTS.get(s, s)),
    })


def salamander_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """The ``split_time_closed.csv`` table from the AnimalCLEF2025 metadata."""
    rows = metadata[(metadata["dataset"] == "SalamanderID2025") & (metadata["split"] == "database")
                    & metadata["date"].notna()].copy()
    if rows.empty:
        raise SourceError("no dated SalamanderID2025 database rows in the AnimalCLEF2025 metadata")
    date = pd.to_datetime(rows["date"])
    latest = date.groupby(rows["identity"]).transform("max")
    dates = date.groupby(rows["identity"]).transform("nunique")
    rows["split"] = ((dates > 1) & (date == latest)).map({True: "query", False: "database"})
    rows["path"] = rows["split"] + "/images/" + rows["path"].str.split("/").str[-1]
    database_views = rows[rows["split"] == "database"].groupby("identity")["orientation"].agg(set)
    rows["cross_view"] = [split == "query" and view not in database_views.get(identity, set())
                          for split, view, identity in zip(rows["split"], rows["orientation"], rows["identity"])]
    columns = ["image_id", "identity", "path", "date", "orientation", "split", "cross_view"]
    return rows.sort_values(["split", "identity", "date", "image_id"])[columns]


def copy_salamander_images(table: pd.DataFrame, source_metadata: pd.DataFrame, source_root: Path, root: Path) -> int:
    """Copy each table image from the competition layout to ``root/<split>/images``; returns the count."""
    origin = source_metadata.set_index(source_metadata["path"].str.split("/").str[-1])["path"]
    copied = 0
    for relative in table["path"]:
        name = relative.split("/")[-1]
        source = source_root / origin[name]
        target = root / relative
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise SourceError(f"{target} exists and differs from {source}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    return copied


def write_new(frame: pd.DataFrame, path: Path, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise SourceError(f"{path} already exists; pass --overwrite or choose --output-dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
