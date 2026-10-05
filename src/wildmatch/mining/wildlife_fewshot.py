"""Few-shot training views for the WildlifeReID-10k workflow.

Starting from a canonical view written by ``scripts.wildlife_dataset`` (the full
dataset), this builds a second symlink view in which every identity keeps only a
fraction of its training frames while ``test`` (and ``val``) stay untouched:

* the training split keeps ``round(fraction * N_train)`` frames in total — the
  requested fraction is the primary constraint;
* the budget is spread proportionally to each identity's frame count, rounded with
  the largest-remainder rule so the total is exact;
* an identity is never reduced below ``min_per_identity`` (default 2) frames, so
  every remaining training frame still has at least one positive.  Identities whose
  proportional share is below that minimum are pinned at the minimum and the other
  identities share the remaining budget proportionally (water-filling), so the
  global fraction is still met.  Only when the minimums alone exceed the budget is
  the effective fraction larger than requested.  Identities that already have a
  single frame in the full data are kept as they are — they never produce training
  entries (the aggregator drops query frames without positives) but they remain
  gallery entries, exactly as in the full view;
* the frames kept for an identity are a prefix of one seeded permutation, so the
  views for 1/8, 1/4 and 1/2 of the data are nested (1/8 ⊂ 1/4 ⊂ 1/2 ⊂ full).

Canonical frame names (``train/<identity>/<collection>/frame_XXXXXX.jpg``) are the
ones of the source view, so the RDD/LoMa feature caches built for the full view stay
valid for every few-shot view (the cache is keyed by that relative path).  What does
change is the gallery, and with it the mined positives/negatives — mining and
fine-tuning are re-run per view, into their own index/checkpoint directories.

The same selection is also written into a metadata CSV for the probe benchmark
(explainable_individual_reidentification): a copy of the dataset metadata with one
extra column per view (``split_frac0.125_seed0`` ...) holding ``train`` for the kept
training rows, ``unused`` for the dropped ones and the original split otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from scripts.wildlife_dataset import WildlifeRecord, load_config

DEFAULT_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
DEFAULT_MIN_PER_IDENTITY = 2
UNUSED_SPLIT = "unused"


def fraction_tag(fraction: float) -> str:
    """Stable text for a fraction: 0.125 -> '0.125', 0.5 -> '0.5', 1 -> '1.0'."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    text = f"{fraction:.6f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def view_name(fraction: float, seed: int) -> str:
    return f"frac{fraction_tag(fraction)}-seed{seed}"


def split_column(fraction: float, seed: int) -> str:
    return f"split_frac{fraction_tag(fraction)}_seed{seed}"


def allocate_budget(
    counts: dict[str, int], fraction: float, min_per_identity: int = DEFAULT_MIN_PER_IDENTITY
) -> dict[str, int]:
    """Frames to keep per identity for a global budget of ``round(fraction * total)``.

    Water-filling: every identity gets ``clip(s * n, min(n, min_per_identity), n)`` for
    the scale ``s`` that makes the total equal the budget (``s == fraction`` when no
    identity has to be lifted to the minimum), and the free identities are rounded with
    the largest-remainder rule.  When the minimums alone exceed the budget they are
    returned as they are and the effective fraction is larger than requested — the
    caller reports that.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if min_per_identity < 1:
        raise ValueError("min_per_identity must be >= 1")
    total = sum(counts.values())
    budget = int(round(fraction * total))
    minimum = {name: min(n, min_per_identity) for name, n in counts.items()}
    kept = dict(minimum)
    if sum(minimum.values()) >= budget:
        return kept

    pinned: set[str] = set()
    while True:
        free = [name for name in counts if name not in pinned]
        remaining = budget - sum(kept[name] for name in pinned)
        scale = remaining / sum(counts[name] for name in free)
        newly_pinned = False
        for name in free:
            target = scale * counts[name]
            if target < minimum[name]:
                kept[name] = minimum[name]
                pinned.add(name)
                newly_pinned = True
            elif target > counts[name]:
                kept[name] = counts[name]
                pinned.add(name)
                newly_pinned = True
        if not newly_pinned:
            break

    free = [name for name in counts if name not in pinned]
    remaining = budget - sum(kept[name] for name in pinned)
    scale = remaining / sum(counts[name] for name in free)
    target = {name: scale * counts[name] for name in free}
    for name in free:
        kept[name] = math.floor(target[name])
    deficit = remaining - sum(kept[name] for name in free)
    for name in sorted(free, key=lambda name: (-(target[name] - math.floor(target[name])), -counts[name], name)):
        if deficit == 0:
            break
        if kept[name] < counts[name]:
            kept[name] += 1
            deficit -= 1
    assert deficit == 0 and sum(kept.values()) == budget, "budget allocation failed"
    return kept


def select_frames(frames: list[str], keep: int, seed: int, identity: str) -> list[str]:
    """Prefix of a seeded permutation of ``frames``; nested across ``keep`` values."""
    order = sorted(frames)
    random.Random(f"{seed}/{identity}").shuffle(order)
    return sorted(order[:keep])


def load_source_records(source_view: Path) -> tuple[dict, list[WildlifeRecord]]:
    """Records of a WildlifeReID canonical view (scripts.wildlife_dataset)."""
    manifest_path = source_view / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source view has no manifest.json: {source_view}")
    manifest = json.loads(manifest_path.read_text())
    records = [WildlifeRecord(**item) for item in manifest["records"] if item.get("included", True)]
    if not records:
        raise ValueError(f"source view has no included records: {source_view}")
    return manifest, records


def load_czechlynx_records(source_view: Path) -> tuple[dict, list[WildlifeRecord]]:
    """Records of a CzechLynx canonical view (scripts.czechlynx_dataset), normalised to the
    same record type: canonical path = ``relative_path`` (``<split>/<identity>/<source>/
    <encounter>/frame_XXXXXX.jpg``), collection = ``<source>/<encounter>``, original path =
    the metadata ``path`` column, generated split = the view split (train/val/test)."""
    manifest_path = source_view / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"source view has no manifest.json: {source_view}")
    manifest = json.loads(manifest_path.read_text())
    records = [
        WildlifeRecord(
            dataset_id="CzechLynx", identity=item["identity"],
            collection=f"{item['source']}/{item['encounter']}",
            original_path=item["original_path"], canonical_path=item["relative_path"],
            official_split=item["metadata_split"], generated_split=item["split"],
        )
        for item in manifest["records"]
    ]
    if not records:
        raise ValueError(f"source view has no records: {source_view}")
    manifest.setdefault("dataset_id", "CzechLynx")
    manifest.setdefault("config", {"dataset_id": "CzechLynx"})
    return manifest, records


def build_subset(
    records: list[WildlifeRecord], fraction: float, seed: int, min_per_identity: int
) -> tuple[list[WildlifeRecord], dict[str, object]]:
    """Return the records of the few-shot view and a summary of the selection."""
    train_frames: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.generated_split == "train":
            train_frames[record.identity].append(record.canonical_path)
    counts = {identity: len(frames) for identity, frames in train_frames.items()}
    if len(counts) < 2:
        raise ValueError("the training split needs at least two identities to provide negatives")
    kept_counts = allocate_budget(counts, fraction, min_per_identity)
    kept_paths: set[str] = set()
    for identity, frames in train_frames.items():
        kept_paths.update(select_frames(frames, kept_counts[identity], seed, identity))

    subset = [r for r in records if r.generated_split != "train" or r.canonical_path in kept_paths]
    total = sum(counts.values())
    kept_total = sum(kept_counts.values())
    budget = int(round(fraction * total))
    singletons = sorted(name for name, n in counts.items() if n == 1)
    lifted = sorted(
        name for name, n in counts.items()
        if n >= 2 and math.floor(fraction * n) < min(n, min_per_identity)
    )
    summary = {
        "fraction": fraction,
        "fraction_tag": fraction_tag(fraction),
        "seed": seed,
        "min_per_identity": min_per_identity,
        "train_frames_full": total,
        "train_frames_budget": budget,
        "train_frames_kept": kept_total,
        "effective_fraction": kept_total / total,
        "budget_feasible": kept_total == budget,
        "train_identities": len(counts),
        "train_identities_with_positives": sum(1 for n in kept_counts.values() if n >= 2),
        "singleton_identities": len(singletons),
        "identities_lifted_to_minimum": len(lifted),
        "per_identity": {name: {"full": counts[name], "kept": kept_counts[name]} for name in sorted(counts)},
    }
    return subset, summary


def write_view(source_view: Path, output_root: Path, records: list[WildlifeRecord], force: bool = False) -> None:
    """Symlink every record of the subset, copying the source view's link targets."""
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir(exist_ok=True)
    for record in records:
        source_link = source_view / record.canonical_path
        if not source_link.is_symlink() and not source_link.is_file():
            raise FileNotFoundError(f"source frame missing: {source_link}")
        target = os.readlink(source_link) if source_link.is_symlink() else str(source_link.resolve())
        link = output_root / record.canonical_path
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink() and os.readlink(link) == target:
                continue
            if not force:
                raise FileExistsError(f"refusing to replace {link}; use --force")
            link.unlink()
        os.symlink(target, link)


def view_is_complete(output_root: Path, records: list[WildlifeRecord], spot_checks: int = 16) -> bool:
    """True when ``records.jsonl`` (written last by a finished run) holds exactly ``records``.

    Re-verifying every symlink of an existing view costs one NFS round trip per frame
    (tens of minutes for CzechLynx when the file server is busy), so a complete view with
    the same selection is reused as is; only a few evenly spaced links are stat-ed.
    """
    listing = output_root / "records.jsonl"
    if not listing.is_file():
        return False
    lines = listing.read_text().splitlines()
    if len(lines) != len(records):
        return False
    if any(json.loads(line) != asdict(record) for line, record in zip(lines, records)):
        return False
    step = max(1, len(records) // spot_checks)
    return all((output_root / record.canonical_path).is_symlink() for record in records[::step])


def prune_stale_links(output_root: Path, records: list[WildlifeRecord]) -> int:
    """Remove links under train/ that a previous run created but this subset drops."""
    wanted = {record.canonical_path for record in records}
    removed = 0
    train_root = output_root / "train"
    if not train_root.is_dir():
        return 0
    for link in sorted(train_root.rglob("frame_*.jpg")):
        if str(link.relative_to(output_root)) not in wanted:
            link.unlink()
            removed += 1
    for directory in sorted((p for p in train_root.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return removed


def summarize_records(records: list[WildlifeRecord]) -> dict[str, object]:
    identities: dict[str, set[str]] = defaultdict(set)
    collections: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        identities[record.generated_split].add(record.identity)
        collections[record.generated_split].add((record.identity, record.collection))
    return {
        "frames": len(records),
        "frames_by_split": dict(Counter(record.generated_split for record in records)),
        "identities_by_split": {split: len(ids) for split, ids in sorted(identities.items())},
        "collections_by_split": {split: len(items) for split, items in sorted(collections.items())},
    }


class _FileLock:
    """Exclusive lock through an O_EXCL lock file (atomic on NFS, unlike flock)."""

    def __init__(self, target: Path, timeout_s: float = 120.0, stale_s: float = 300.0) -> None:
        self.path = target.with_name(target.name + ".lock")
        self.timeout_s = timeout_s
        self.stale_s = stale_s

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}@{os.uname().nodename}\n".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        self.path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not acquire {self.path} within {self.timeout_s}s")
                time.sleep(0.5 + random.random())

    def __exit__(self, *exc: object) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def write_metadata_column(
    metadata_csv: Path, output_csv: Path, column: str, records: list[WildlifeRecord],
    split_column_name: str = "split",
) -> dict[str, int]:
    """Add/replace ``column`` in a copy of the dataset metadata used by the probe.

    Rows are matched to records through the metadata ``path`` column (the record's
    ``original_path``).  Kept training rows get ``train``, every other row that the
    dataset marks as ``train`` gets ``unused`` (dropped frames and rows the canonical
    view excluded), and all remaining rows keep their original split value.  When
    ``output_csv`` already exists (columns for other fractions/seeds), it is used as
    the base so that one file accumulates every view; the read-modify-write is guarded
    by a lock file because the views of several fractions are usually built by
    concurrent jobs.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with _FileLock(output_csv):
        return _write_metadata_column_locked(metadata_csv, output_csv, column, records, split_column_name)


def _write_metadata_column_locked(
    metadata_csv: Path, output_csv: Path, column: str, records: list[WildlifeRecord],
    split_column_name: str = "split",
) -> dict[str, int]:
    base = output_csv if output_csv.is_file() else metadata_csv
    with base.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "path" not in fields or split_column_name not in fields:
        raise ValueError(f"{base}: metadata needs 'path' and {split_column_name!r} columns")
    kept = {r.original_path for r in records if r.generated_split == "train"}
    values = Counter()
    for row in rows:
        original = str(row[split_column_name]).strip().lower()
        if row["path"] in kept:
            value = "train"
        elif original == "train":
            value = UNUSED_SPLIT
        else:
            value = row[split_column_name]
        row[column] = value
        values[value] += 1
    if len(kept) != values["train"]:
        raise ValueError(
            f"{len(kept)} kept training frames but {values['train']} metadata rows matched; "
            "the metadata CSV does not correspond to the source view"
        )
    if column not in fields:
        fields.append(column)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_csv)
    return dict(values)


def prepare_fewshot(
    source_view: Path,
    output_root: Path,
    fraction: float,
    seed: int,
    min_per_identity: int = DEFAULT_MIN_PER_IDENTITY,
    metadata_csv: Path | None = None,
    metadata_out: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    loader=load_source_records,
    split_column_name: str = "split",
) -> dict[str, object]:
    source_manifest, records = loader(source_view)
    subset, selection = build_subset(records, fraction, seed, min_per_identity)
    summary: dict[str, object] = {
        **summarize_records(subset),
        "dataset_id": source_manifest.get("dataset_id"),
        "protocol": source_manifest.get("protocol"),
        "source_view": str(source_view),
        "output_root": str(output_root),
        "view": view_name(fraction, seed),
        "split_column": split_column(fraction, seed),
        "fewshot": selection,
        "config": source_manifest.get("config"),
    }
    if dry_run:
        return summary
    existing = output_root / "fewshot.json"
    if existing.is_file() and not force:
        previous = json.loads(existing.read_text()).get("fewshot", {})
        keys = ("fraction", "seed", "min_per_identity")
        if any(previous.get(key) != selection[key] for key in keys):
            raise FileExistsError(
                f"{output_root} already holds a view with different parameters "
                f"({ {key: previous.get(key) for key in keys} }); use --force to replace it"
            )
    if not force and existing.is_file() and view_is_complete(output_root, subset):
        print(f"reusing the complete view {output_root} (links not re-verified; --force rewrites)", file=sys.stderr)
        summary["fewshot"]["reused_existing_view"] = True
        summary["fewshot"]["stale_links_removed"] = 0
    else:
        write_view(source_view, output_root, subset, force=force)
        summary["fewshot"]["stale_links_removed"] = prune_stale_links(output_root, subset)
    if metadata_out is not None:
        csv_source = metadata_csv or Path(str(source_manifest["config"]["metadata_csv"]))
        summary["metadata_csv"] = str(metadata_out)
        summary["metadata_split_counts"] = write_metadata_column(
            csv_source, metadata_out, split_column(fraction, seed), subset, split_column_name
        )
    (output_root / "fewshot.json").write_text(json.dumps(summary, indent=2))
    (output_root / "manifest.json").write_text(
        json.dumps({**summary, "records": [asdict(record) for record in subset]}, indent=2)
    )
    (output_root / "records.jsonl").write_text(
        "".join(json.dumps(asdict(record)) + "\n" for record in subset)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/wildlife/<dataset>.json")
    parser.add_argument("--protocol", choices=["strict", "legacy"], default="legacy")
    parser.add_argument("--fraction", type=float, required=True, help="share of training frames to keep, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_per_identity", type=int, default=DEFAULT_MIN_PER_IDENTITY)
    parser.add_argument("--source_view", type=Path, default=None,
                        help="full canonical view (default: $WILDLIFE_PROCESSED_ROOT/<dataset>/<protocol>)")
    parser.add_argument("--output_root", type=Path, default=None,
                        help="few-shot view (default: $FEWSHOT_ROOT/views/<dataset>/<protocol>/<view>)")
    parser.add_argument("--metadata_out", type=Path, default=None,
                        help="probe metadata copy receiving the split column "
                             "(default: <source_root>/metadata_fewshot/metadata_<dataset>.csv; 'none' disables)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(os.environ.get("CZECHLYNX_DATA_ROOT", "/shared/sets/datasets/vision/czechlynx"))
    processed_root = Path(os.environ.get("WILDLIFE_PROCESSED_ROOT", data_root / "wildlife_processed"))
    fewshot_root = Path(os.environ.get("FEWSHOT_ROOT", data_root / "fewshot"))
    source_view = args.source_view or processed_root / config.dataset_id / args.protocol
    output_root = args.output_root or fewshot_root / "views" / config.dataset_id / args.protocol / view_name(args.fraction, args.seed)
    if args.metadata_out is None:
        metadata_out: Path | None = config.root / "metadata_fewshot" / f"metadata_{config.dataset_id}.csv"
    elif str(args.metadata_out).lower() == "none":
        metadata_out = None
    else:
        metadata_out = args.metadata_out
    summary = prepare_fewshot(
        source_view=source_view,
        output_root=output_root,
        fraction=args.fraction,
        seed=args.seed,
        min_per_identity=args.min_per_identity,
        metadata_csv=config.metadata_path,
        metadata_out=metadata_out,
        force=args.force,
        dry_run=args.dry_run,
    )
    printable = {key: value for key, value in summary.items() if key not in {"config"}}
    printable["fewshot"] = {k: v for k, v in summary["fewshot"].items() if k != "per_identity"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
