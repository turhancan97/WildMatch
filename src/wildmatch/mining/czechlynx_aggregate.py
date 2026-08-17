"""Aggregate CzechLynx per-query mining reports into flat triplet indices."""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path


def relative(path: str, root: Path) -> str:
    return str(Path(path).relative_to(root))


def aggregate(prefix: Path, split: str, root: Path) -> list[dict]:
    reports = [
        Path(path)
        for path in sorted(glob(f"{prefix}_{split}_*.json"))
        if not path.endswith("_combined.json")
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
        print(f"{split}: skipped {skipped} reports without both candidate classes")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_report", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    for split in args.splits:
        entries = aggregate(args.dump_report, split, args.dataset_root)
        if not entries:
            print(f"{split}: no usable reports found")
            continue
        output = Path(f"{args.dump_report}_{split}_combined.json")
        output.write_text(json.dumps(entries, indent=2))
        print(f"{split}: wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
