"""Aggregate generic WildlifeReID mining shards into trainer indices."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path


def relative(path: str, root: Path) -> str:
    value = Path(path)
    if value.is_absolute():
        return str(value.relative_to(root))
    return str(value)


def aggregate(prefix: Path, split: str, root: Path) -> list[dict]:
    reports = [
        Path(path)
        for path in sorted(glob(f"{prefix}_{split}_*.json"))
        if not path.endswith("_combined.json") and not path.endswith(".metadata.json")
    ]
    entries: list[dict] = []
    skipped = 0
    for report_path in reports:
        data = json.loads(report_path.read_text())
        for frame in data.get("selected_frames", []):
            if not frame.get("positives") or not frame.get("negatives"):
                skipped += 1
                continue
            entries.append(
                {
                    "query_frame": relative(frame["query_frame"], root),
                    "positives": [relative(item["frame"], root) for item in frame["positives"]],
                    "negatives": [relative(item["frame"], root) for item in frame["negatives"]],
                }
            )
    if skipped:
        print(f"{split}: skipped {skipped} query frames without both candidate classes")
    return entries


def write_entries(prefix: Path, split: str, root: Path, metadata: dict) -> int:
    entries = aggregate(prefix, split, root)
    output = Path(f"{prefix}_{split}_combined.json")
    output.write_text(json.dumps(entries, indent=2))
    output.with_suffix(".metadata.json").write_text(
        json.dumps({**metadata, "split": split, "entries": len(entries)}, indent=2)
    )
    print(f"{split}: wrote {len(entries)} entries to {output}")
    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_report", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--dataset_id", default="wildlife-reid-10k")
    parser.add_argument("--protocol", choices=["strict", "legacy"], default="strict")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--backend", choices=["rdd", "loma"], default="rdd")
    parser.add_argument("--variant", default="loma-b")
    parser.add_argument("--weights", default="")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--frames_per_collection", type=int, default=20)
    parser.add_argument("--top_k_frames", type=int, default=5)
    parser.add_argument("--top_m", type=int, default=10)
    args = parser.parse_args()
    metadata = {
        "dataset": args.dataset_id,
        "protocol": args.protocol,
        "backend": args.backend,
        "variant": args.variant if args.backend == "loma" else None,
        "weights": args.weights,
        "cache_dir": args.cache_dir,
        "frames_per_collection": args.frames_per_collection,
        "top_k_frames": args.top_k_frames,
        "top_m": args.top_m,
        "gallery_split": "train",
        "query_gallery_rule": "train gallery; split queries",
    }
    requested = list(dict.fromkeys(args.splits))
    for split in requested:
        if split == "val" and args.protocol == "legacy":
            test_output = Path(f"{args.dump_report}_test_combined.json")
            if not test_output.is_file():
                write_entries(args.dump_report, "test", args.dataset_root, metadata)
            entries = json.loads(test_output.read_text())
            output = Path(f"{args.dump_report}_val_combined.json")
            output.write_text(json.dumps(entries, indent=2))
            output.with_suffix(".metadata.json").write_text(
                json.dumps({**metadata, "split": "val", "entries": len(entries), "legacy_alias_of": "test"}, indent=2)
            )
            print(f"val: legacy alias of test ({len(entries)} entries) -> {output}")
        else:
            write_entries(args.dump_report, split, args.dataset_root, metadata)


if __name__ == "__main__":
    main()
