"""CzechLynx metadata, split, and canonical symlink-view utilities.

The public CzechLynx tree stores masked images as::

    CzechLynx_masked/<source>/<unique_name>/<original_filename>.jpg

while the Lynx tools expect split-prefixed paths with a sequence directory.
This module keeps the source data immutable and creates a reproducible view::

    <output>/<split>/<unique_name>/<source>/<encounter>/frame_XXXXXX.jpg

Encounter is the atomic unit used for split hygiene.  Benchmark collection
grouping is deliberately separate: callers can group the generated records by
``(source, unique_name)`` for the time-closed collection protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SOURCE_ROOT = Path("/shared/sets/datasets/vision/czechlynx/CzechLynx_v2")
DEFAULT_OUTPUT_ROOT = Path("/shared/sets/datasets/vision/czechlynx/CzechLynx_processed_time_closed")
SUPPORTED_SPLIT_COLUMNS = ("split-time_closed", "split-time_open")
DEFAULT_SPLIT_COLUMN = "split-time_closed"


@dataclass(frozen=True)
class CzechLynxRecord:
    split: str
    identity: str
    source: str
    encounter: str
    original_path: str
    masked_path: str
    relative_path: str
    metadata_split: str

    @property
    def collection(self) -> str:
        return f"{self.source}/{self.identity}"

    @property
    def encounter_key(self) -> tuple[str, str]:
        return self.source, self.encounter


@dataclass
class CzechLynxCollection:
    split: str
    identity: str
    source: str
    frame_paths: list[Path]

    @property
    def name(self) -> str:
        return f"{self.identity}/{self.source}"


def masked_relative_path(original_path: str) -> str:
    """Map a full-metadata path into the masked tree."""
    prefix = "CzechLynx/"
    if not original_path.startswith(prefix):
        raise ValueError(f"unexpected CzechLynx metadata path: {original_path}")
    return "CzechLynx_masked/" + original_path[len(prefix) :]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_metadata(
    source_root: Path,
    metadata_path: Path | None = None,
    masked_metadata_path: Path | None = None,
) -> list[dict[str, str]]:
    """Load and cross-check the full and masked metadata files."""
    metadata_path = metadata_path or source_root / "CzechLynxDataset-Metadata-Real.csv"
    masked_metadata_path = masked_metadata_path or source_root / "CzechLynxDataset-masked-Metadata-Real.csv"
    full_rows = _read_csv(metadata_path)
    masked_rows = _read_csv(masked_metadata_path)
    masked_by_path = {row["path"]: row for row in masked_rows}

    if len(masked_by_path) != len(masked_rows):
        raise ValueError("masked metadata contains duplicate paths")

    merged: list[dict[str, str]] = []
    for row in full_rows:
        masked_path = masked_relative_path(row["path"])
        masked = masked_by_path.get(masked_path)
        if masked is None:
            raise ValueError(f"full metadata path has no masked row: {masked_path}")
        if masked.get("unique_name") != row["unique_name"]:
            raise ValueError(
                f"identity mismatch for {masked_path}: {row['unique_name']} != {masked.get('unique_name')}"
            )
        merged.append({**row, "masked_path": masked_path})

    if len(merged) != len(masked_rows):
        raise ValueError(f"metadata row mismatch: full={len(merged)} masked={len(masked_rows)}")
    return merged


def assign_splits(
    rows: Iterable[dict[str, str]],
    *,
    split_column: str = DEFAULT_SPLIT_COLUMN,
    validation_fraction: float = 0.20,
    seed: int = 0,
) -> tuple[dict[tuple[str, str], str], list[tuple[str, str]]]:
    """Assign whole source/encounter groups to train, val, or test.

    The selected metadata split label is authoritative.  A mixed
    encounter is conservatively assigned to test so no test frame can enter a
    training or validation encounter.  Only all-train encounters participate
    in the deterministic validation holdout.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if split_column not in SUPPORTED_SPLIT_COLUMNS:
        raise ValueError(f"split_column must be one of {SUPPORTED_SPLIT_COLUMNS}, got {split_column!r}")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row.get(split_column, "")
        if label not in {"train", "test"}:
            raise ValueError(f"{split_column} must be train/test, got {label!r} for {row['path']}")
        groups[(row["source"], row["encounter"])].append(row)

    assignments: dict[tuple[str, str], str] = {}
    mixed: list[tuple[str, str]] = []
    train_groups: list[tuple[str, str]] = []
    for key, group in sorted(groups.items()):
        labels = {row[split_column] for row in group}
        if labels == {"test"} or labels == {"train", "test"}:
            assignments[key] = "test"
            if labels == {"train", "test"}:
                mixed.append(key)
        else:
            train_groups.append(key)

    rng = random.Random(seed)
    rng.shuffle(train_groups)
    n_val = math.ceil(len(train_groups) * validation_fraction)
    val_groups = set(train_groups[:n_val])
    for key in train_groups:
        assignments[key] = "val" if key in val_groups else "train"
    return assignments, mixed


def build_records(
    rows: list[dict[str, str]],
    assignments: dict[tuple[str, str], str],
    *,
    split_column: str = DEFAULT_SPLIT_COLUMN,
) -> list[CzechLynxRecord]:
    """Create deterministic canonical relative paths for all metadata rows."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            assignments[(row["source"], row["encounter"])],
            row["unique_name"],
            row["source"],
            row["encounter"],
        )
        grouped[key].append(row)

    records: list[CzechLynxRecord] = []
    for (split, identity, source, encounter), group in sorted(grouped.items()):
        group.sort(key=lambda row: row["path"])
        for index, row in enumerate(group):
            relative_path = f"{split}/{identity}/{source}/{encounter}/frame_{index:06d}.jpg"
            records.append(
                CzechLynxRecord(
                    split=split,
                    identity=identity,
                    source=source,
                    encounter=encounter,
                    original_path=row["path"],
                    masked_path=row["masked_path"],
                    relative_path=relative_path,
                    metadata_split=row[split_column],
                )
            )
    return records


def validate_records(
    records: Iterable[CzechLynxRecord],
    source_root: Path,
    output_root: Path | None = None,
) -> dict[str, object]:
    records = list(records)
    if not records:
        raise ValueError("no CzechLynx records")
    by_original: dict[str, CzechLynxRecord] = {}
    by_relative: dict[str, CzechLynxRecord] = {}
    for record in records:
        if record.original_path in by_original:
            raise ValueError(f"duplicate original path: {record.original_path}")
        if record.relative_path in by_relative:
            raise ValueError(f"duplicate canonical path: {record.relative_path}")
        by_original[record.original_path] = record
        by_relative[record.relative_path] = record
        target = source_root / record.masked_path
        if not target.is_file():
            raise FileNotFoundError(target)
        if output_root is not None:
            link = output_root / record.relative_path
            if not link.is_symlink():
                raise ValueError(f"canonical path is not a symlink: {link}")
            if link.resolve() != target.resolve():
                raise ValueError(f"symlink target mismatch: {link}")

    split_groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        split_groups[record.split].add(record.encounter_key)
    overlaps = set.intersection(*(groups for groups in split_groups.values())) if len(split_groups) > 1 else set()
    if overlaps:
        raise ValueError(f"encounter leakage across generated splits: {sorted(overlaps)[:5]}")

    return {
        "frames": len(records),
        "frames_by_split": dict(Counter(record.split for record in records)),
        "identities_by_split": {
            split: len({record.identity for record in records if record.split == split})
            for split in sorted(split_groups)
        },
        "encounters_by_split": {split: len(groups) for split, groups in sorted(split_groups.items())},
        "collections_by_split": {
            split: len({record.collection for record in records if record.split == split})
            for split in sorted(split_groups)
        },
    }


def write_view(
    records: list[CzechLynxRecord],
    source_root: Path,
    output_root: Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    for record in records:
        target = (source_root / record.masked_path).resolve()
        link = output_root / record.relative_path
        if dry_run:
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if not force:
                if link.is_symlink() and link.resolve() == target:
                    continue
                raise FileExistsError(f"refusing to replace existing path {link}; use --force")
            link.unlink()
        os.symlink(target, link)


def prepare(args: argparse.Namespace) -> dict[str, object]:
    rows = load_metadata(args.source_root, args.metadata, args.masked_metadata)
    assignments, mixed = assign_splits(
        rows,
        split_column=args.split_column,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    records = build_records(rows, assignments, split_column=args.split_column)
    summary = validate_records(records, args.source_root)
    summary.update(
        {
            "dataset": "CzechLynx",
            "protocol": args.split_column,
            "split_column": args.split_column,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "mixed_encounters_assigned_to_test": [list(key) for key in mixed],
            "source_root": str(args.source_root),
            "output_root": str(args.output_root),
        }
    )
    write_view(
        records,
        args.source_root,
        args.output_root,
        dry_run=args.dry_run,
        force=args.force,
    )
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "manifest.json").write_text(
            json.dumps(
                {
                    **summary,
                    "records": [asdict(record) for record in records],
                },
                indent=2,
            )
        )
        # A compact manifest is convenient for tools that only need the split
        # and path mapping and do not want to load the full summary first.
        (args.output_root / "records.jsonl").write_text(
            "".join(json.dumps(asdict(record)) + "\n" for record in records)
        )
        summary = validate_records(records, args.source_root, args.output_root) | summary
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--masked_metadata", type=Path, default=None)
    parser.add_argument(
        "--split_column",
        choices=SUPPORTED_SPLIT_COLUMNS,
        default=DEFAULT_SPLIT_COLUMN,
        help="Official metadata split column used to create train/test assignments.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Validate an existing canonical view")
    args = parser.parse_args()
    if args.validate:
        manifest = json.loads((args.output_root / "manifest.json").read_text())
        records = [CzechLynxRecord(**row) for row in manifest["records"]]
        print(json.dumps(validate_records(records, args.source_root, args.output_root), indent=2))
        return args
    prepare(args)
    return args


if __name__ == "__main__":
    parse_args()
