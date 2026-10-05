"""Configuration-driven dataset preparation for WildlifeReID-10k.

The adapter turns the heterogeneous metadata/image trees into a stable,
symlink-based layout consumed by the existing mining, cache, and evaluation
tools.  It deliberately does not copy image data.
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

DEFAULT_ROOT = Path("/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k")


@dataclass(frozen=True)
class WildlifeConfig:
    dataset_id: str
    metadata_csv: str
    source_root: str
    image_prefix: str = "masked_images"
    identity_column: str = "identity"
    split_column: str = "split"
    collection_rule: str = "identity"
    validation_fraction: float = 0.20
    seed: int = 0
    excluded_identities: tuple[str, ...] = ("", "unknown")
    allowed_splits: tuple[str, ...] = ("train", "test")

    @property
    def metadata_path(self) -> Path:
        return Path(self.metadata_csv).expanduser()

    @property
    def root(self) -> Path:
        return Path(self.source_root).expanduser()


@dataclass(frozen=True)
class WildlifeRecord:
    dataset_id: str
    identity: str
    collection: str
    original_path: str
    canonical_path: str
    official_split: str
    generated_split: str
    included: bool = True
    exclusion_reason: str = ""


@dataclass(frozen=True)
class WildlifeCollection:
    split: str
    identity: str
    collection: str
    frame_paths: list[Path]

    @property
    def name(self) -> str:
        return f"{self.identity}/{self.collection}"


def load_config(path: Path) -> WildlifeConfig:
    raw = json.loads(Path(path).read_text())
    required = {"dataset_id", "metadata_csv"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing config fields: {sorted(missing)}")
    root = Path(raw.get("source_root", DEFAULT_ROOT)).expanduser()
    metadata = Path(raw["metadata_csv"]).expanduser()
    if not metadata.is_absolute():
        metadata = root / metadata
    return WildlifeConfig(
        dataset_id=str(raw["dataset_id"]),
        metadata_csv=str(metadata),
        source_root=str(root),
        image_prefix=str(raw.get("image_prefix", "masked_images")).strip("/"),
        identity_column=str(raw.get("identity_column", "identity")),
        split_column=str(raw.get("split_column", "split")),
        collection_rule=str(raw.get("collection_rule", "identity")),
        validation_fraction=float(raw.get("validation_fraction", 0.20)),
        seed=int(raw.get("seed", 0)),
        excluded_identities=tuple(str(x) for x in raw.get("excluded_identities", ["", "unknown"])),
        allowed_splits=tuple(str(x) for x in raw.get("allowed_splits", ["train", "test"])),
    )


def config_from_registry(key: str, profile: str | None = None, inputs: str = "paper") -> WildlifeConfig:
    """The mining config of registry entry `key` (replaces the former configs/wildlife/*.json).

    `inputs="paper"` reads the table the paper's views, caches and indices were built from
    (`registry.paper_inputs.metadata_file`, the WildlifeReID-10k team masks; the entry's own table
    when it has no separate paper inputs). Mining on another table needs its own view, cache and
    index folders (caches are keyed by frame path), so `inputs="current"` is refused where the two
    tables differ. Optional per-entry settings live in `registry.mining` (e.g. `split_column`).
    """
    from wildmatch.data.registry import load_dataset

    entry = load_dataset(key, profile)
    if entry.name == "CzechLynx_v2":
        raise ValueError(f"{key} is a CzechLynx entry; its views come from wildmatch.mining.czechlynx_dataset")
    registry = entry.get("registry") or {}
    paper = (registry.get("paper_inputs") or {}).get("metadata_file")
    if inputs == "paper":
        metadata = paper or entry.metadata_file
    elif inputs == "current":
        if paper and paper != entry.metadata_file:
            raise ValueError(
                f"{key}: mining on the current inputs ({entry.metadata_file}) is not set up; it needs view, cache "
                f"and index folders separate from the paper inputs' ({paper})"
            )
        metadata = entry.metadata_file
    else:
        raise ValueError(f"inputs must be paper or current, got {inputs!r}")
    mining = registry.get("mining") or {}
    root = Path(str(entry.root))
    return WildlifeConfig(
        dataset_id=str(entry.animal),
        metadata_csv=str(root / str(metadata)),
        source_root=str(root),
        image_prefix=str(mining.get("image_prefix", "masked_images")).strip("/"),
        identity_column=str(mining.get("identity_column", "identity")),
        split_column=str(mining.get("split_column", "split")),
        collection_rule=str(mining.get("collection_rule", "identity")),
        validation_fraction=float(mining.get("validation_fraction", 0.20)),
        seed=int(mining.get("seed", 0)),
        excluded_identities=tuple(str(x) for x in mining.get("excluded_identities", ["", "unknown"])),
        allowed_splits=tuple(str(x) for x in mining.get("allowed_splits", ["train", "test"])),
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--registry", help="dataset registry key, e.g. salamander (the usual source)")
    group.add_argument("--config", type=Path, help="a JSON mining config, for datasets outside the registry")
    parser.add_argument("--paths", default=None, help="path profile for --registry")
    parser.add_argument("--inputs", choices=["paper", "current"], default="paper", help="input table for --registry")


def config_from_arguments(args: argparse.Namespace) -> WildlifeConfig:
    if args.registry:
        return config_from_registry(args.registry, args.paths, args.inputs)
    return load_config(args.config)


def read_rows(config: WildlifeConfig) -> list[dict[str, str]]:
    with config.metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metadata is empty: {config.metadata_path}")
    fields = set(rows[0])
    for field in (config.identity_column, config.split_column, "path"):
        if field not in fields:
            raise ValueError(f"{config.metadata_path}: missing required column {field!r}")
    return rows


def resolve_image(config: WildlifeConfig, metadata_path: str) -> Path:
    path = Path(metadata_path)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == config.image_prefix:
        return config.root / Path(*parts)
    return config.root / config.image_prefix / path


def collection_name(config: WildlifeConfig, row: dict[str, str], identity: str) -> str:
    rule = config.collection_rule
    path = Path(row["path"])
    if rule == "identity":
        return identity
    if rule == "parent":
        return path.parent.name or identity
    if rule.startswith("path_component:"):
        try:
            index = int(rule.split(":", 1)[1])
            return path.parts[index]
        except (ValueError, IndexError):
            raise ValueError(f"invalid collection_rule {rule!r} for path {path}")
    raise ValueError(f"unsupported collection_rule: {rule!r}")


def assign_generated_splits(
    rows: list[dict[str, str]], config: WildlifeConfig, protocol: str
) -> tuple[dict[int, str], list[dict[str, str]]]:
    if protocol not in {"strict", "legacy"}:
        raise ValueError("protocol must be 'strict' or 'legacy'")
    assignments: dict[int, str] = {}
    exclusions: list[dict[str, str]] = []
    valid: list[tuple[int, dict[str, str]]] = []
    for index, row in enumerate(rows):
        identity = str(row.get(config.identity_column, "")).strip()
        split = str(row.get(config.split_column, "")).strip().lower()
        reasons = []
        if identity in config.excluded_identities:
            reasons.append("excluded_identity")
        if split not in config.allowed_splits:
            reasons.append("invalid_or_unlabeled_split")
        if not resolve_image(config, row["path"]).is_file():
            reasons.append("missing_masked_image")
        if reasons:
            exclusions.append({"row_index": str(index), "path": row["path"], "reason": ";".join(reasons)})
        else:
            valid.append((index, row))

    if protocol == "legacy":
        for index, row in valid:
            assignments[index] = str(row[config.split_column]).strip().lower()
        return assignments, exclusions

    train_by_identity: dict[str, list[int]] = defaultdict(list)
    for index, row in valid:
        if str(row[config.split_column]).strip().lower() == "train":
            train_by_identity[str(row[config.identity_column]).strip()].append(index)
        else:
            assignments[index] = "test"

    rng = random.Random(config.seed)
    for identity, indices in sorted(train_by_identity.items()):
        indices = sorted(indices, key=lambda i: rows[i]["path"])
        rng.shuffle(indices)
        if len(indices) < 2:
            val_count = 0
        else:
            val_count = min(len(indices) - 1, max(1, math.ceil(len(indices) * config.validation_fraction)))
        val_indices = set(indices[:val_count])
        for index in indices:
            assignments[index] = "val" if index in val_indices else "train"
    return assignments, exclusions


def build_records(
    rows: list[dict[str, str]],
    config: WildlifeConfig,
    assignments: dict[int, str],
    exclusions: list[dict[str, str]],
) -> list[WildlifeRecord]:
    records: list[WildlifeRecord] = []
    by_group: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, generated_split in assignments.items():
        row = rows[index]
        identity = str(row[config.identity_column]).strip()
        collection = collection_name(config, row, identity)
        by_group[(generated_split, identity, collection)].append((index, row))
    for (split, identity, collection), group in sorted(by_group.items()):
        group.sort(key=lambda item: item[1]["path"])
        for frame_index, (index, row) in enumerate(group):
            records.append(
                WildlifeRecord(
                    dataset_id=config.dataset_id,
                    identity=identity,
                    collection=collection,
                    original_path=row["path"],
                    canonical_path=f"{split}/{identity}/{collection}/frame_{frame_index:06d}.jpg",
                    official_split=str(row[config.split_column]).strip().lower(),
                    generated_split=split,
                )
            )
    return records


def write_view(records: list[WildlifeRecord], config: WildlifeConfig, output_root: Path, force: bool = False) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir(exist_ok=True)
    for record in records:
        target = resolve_image(config, record.original_path).resolve()
        link = output_root / record.canonical_path
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.is_symlink() and link.resolve() == target:
                continue
            if not force:
                raise FileExistsError(f"refusing to replace {link}; use --force")
            link.unlink()
        os.symlink(target, link)


def validate_records(records: list[WildlifeRecord], config: WildlifeConfig, output_root: Path) -> dict[str, object]:
    if not records:
        raise ValueError(f"{config.dataset_id}: no usable records")
    canonical = set()
    original = set()
    split_identities: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.canonical_path in canonical:
            raise ValueError(f"duplicate canonical path: {record.canonical_path}")
        if record.original_path in original:
            raise ValueError(f"duplicate original path: {record.original_path}")
        canonical.add(record.canonical_path)
        original.add(record.original_path)
        target = resolve_image(config, record.original_path)
        link = output_root / record.canonical_path
        if not target.is_file():
            raise FileNotFoundError(target)
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise ValueError(f"invalid symlink: {link}")
        split_identities[record.generated_split].add(record.identity)
    return {
        "dataset_id": config.dataset_id,
        "frames": len(records),
        "frames_by_split": dict(Counter(record.generated_split for record in records)),
        "identities_by_split": {split: len(ids) for split, ids in sorted(split_identities.items())},
        "collections_by_split": {
            split: len({(r.identity, r.collection) for r in records if r.generated_split == split})
            for split in sorted(split_identities)
        },
    }


def prepare(
    config: WildlifeConfig, output_root: Path, protocol: str, force: bool = False, dry_run: bool = False
) -> dict[str, object]:
    rows = read_rows(config)
    assignments, exclusions = assign_generated_splits(rows, config, protocol)
    records = build_records(rows, config, assignments, exclusions)
    if not records:
        raise ValueError(
            f"{config.dataset_id}: no usable masked records remain after validation; "
            "check the metadata identity/split columns and masked_images path. "
            "Unmasked images are never used as a fallback."
        )
    if not dry_run:
        write_view(records, config, output_root, force=force)
        output_root.mkdir(parents=True, exist_ok=True)
    base_summary = (
        validate_records(records, config, output_root)
        if not dry_run
        else {
            "dataset_id": config.dataset_id,
            "frames": len(records),
            "frames_by_split": dict(Counter(r.generated_split for r in records)),
        }
    )
    summary = {
        **base_summary,
        "protocol": protocol,
        "source_root": str(config.root),
        "output_root": str(output_root),
        "excluded_rows": len(exclusions),
        "exclusions": exclusions,
        "config": asdict(config),
        "records": [asdict(record) for record in records],
    }
    if not dry_run:
        (output_root / "manifest.json").write_text(json.dumps(summary, indent=2))
        (output_root / "records.jsonl").write_text("".join(json.dumps(asdict(record)) + "\n" for record in records))
    return summary


def list_collections(root: Path, split: str) -> list[WildlifeCollection]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split {split!r} not found under {root}")
    output: list[WildlifeCollection] = []
    for identity_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for collection_dir in sorted(p for p in identity_dir.iterdir() if p.is_dir()):
            frames = sorted(collection_dir.glob("frame_*.jpg"))
            if frames:
                output.append(WildlifeCollection(split, identity_dir.name, collection_dir.name, frames))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["strict", "legacy"], default="strict")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    config = config_from_arguments(args)
    summary = prepare(config, args.output_root, args.protocol, force=args.force, dry_run=args.dry_run)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
