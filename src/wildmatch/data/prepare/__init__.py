"""Dataset preparation: ``wildmatch prepare status|download|jaguar|unseen-split``.

``status`` checks what each registry entry needs on disk (root, metadata, split values, a
sample of image files, the mask column when masks are applied at load time) and prints the
entry's data source (``registry.download``). ``download`` fetches raw data where a public
route exists (WildlifeReID-10k from Kaggle through wildlife-datasets; Kaggle API credentials
required). ``jaguar`` runs the JaguarReID preparation steps and ``unseen-split`` rebuilds the
CzechLynx unseen-identity evaluation split with the paper's parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from wildmatch.data.registry import dataset_keys, load_dataset
from wildmatch.paths import active_profile

# Parameters of the paper's unseen-identity split (its manifest, 2026-09-23).
UNSEEN_SOURCE = "czechlynx_open"
UNSEEN_OUTPUT = Path("metadata") / "czechlynx-unseen-eval"
UNSEEN_ARGS = ["--group-col", "encounter", "--order-col", "date", "--path-col", "path"]
# Registry entries whose raw data wildlife-datasets can download, and the class that does it.
DOWNLOADERS = {"WildlifeReID-10k": "WildlifeReID10k"}


def _metadata_path(entry) -> Path:
    path = Path(str(entry.metadata_file))
    return path if path.is_absolute() else Path(str(entry.root)) / path


def check(key: str, profile: Optional[str] = None, sample: int = 20) -> Dict[str, object]:
    """What is present for one registry entry; ``ready`` is True when a run can start."""
    entry = load_dataset(key, profile)
    root, metadata = Path(str(entry.root)), _metadata_path(entry)
    report: Dict[str, object] = {"key": key, "root": str(root), "metadata": str(metadata), "problems": []}
    problems: List[str] = report["problems"]  # type: ignore[assignment]
    if not root.is_dir():
        problems.append("root missing")
    if not metadata.is_file():
        problems.append("metadata missing")
    else:
        table = pd.read_csv(metadata, low_memory=False)
        report["rows"] = len(table)
        needed = [str(entry.label_col), str(entry.split_col), "path"]
        if bool(entry.no_background):
            needed.append(str(entry.mask_col))
        missing = [column for column in needed if column not in table.columns]
        if missing:
            problems.append(f"columns missing: {', '.join(missing)}")
        else:
            values = set(table[str(entry.split_col)].astype(str))
            for side in (str(entry.database_split_value), str(entry.query_split_value)):
                if side not in values:
                    problems.append(f"split value '{side}' absent")
            paths = table["path"].astype(str)
            picked = paths.iloc[:: max(1, len(paths) // max(1, sample))].head(sample)
            absent = [p for p in picked if not (Path(p) if Path(p).is_absolute() else root / p).is_file()]
            if absent:
                problems.append(f"{len(absent)} of {len(picked)} sampled images missing (e.g. {absent[0]})")
    download = entry.registry.get("download") or {}
    report["raw"] = download.get("raw")
    report["derived"] = download.get("derived")
    report["ready"] = not problems
    return report


def download_raw(key: str, profile: Optional[str] = None) -> Path:
    entry = load_dataset(key, profile)
    name = str(entry.name)
    if name not in DOWNLOADERS:
        raw = (entry.registry.get("download") or {}).get("raw")
        raise SystemExit(f"no automatic download for {key}; source: {raw}")
    import wildlife_datasets.datasets as datasets

    root = Path(str(entry.root))
    getattr(datasets, DOWNLOADERS[name]).get_data(str(root))
    return root


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Check, download or build datasets.")
    parser.add_argument("--paths", help="path profile (default: WILDMATCH_PATHS, wildmatch.local.yaml, default)")
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", help="what is on disk for each registry entry")
    status.add_argument("--dataset", action="append", default=[], help="registry key (repeatable; default all)")
    status.add_argument("--sample", type=int, default=20, help="image files checked per entry")
    download = sub.add_parser("download", help="fetch raw data where a public route exists")
    download.add_argument("dataset", help="registry key")
    jaguar = sub.add_parser("jaguar", help="JaguarReID steps: prepare, embed, split (options as in the module)")
    jaguar.add_argument("arguments", nargs=argparse.REMAINDER)
    unseen = sub.add_parser("unseen-split", help="rebuild the CzechLynx unseen-identity split")
    unseen.add_argument("--output-dir", type=Path, help=f"default: <{UNSEEN_SOURCE} root>/{UNSEEN_OUTPUT}")
    unseen.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    profile = active_profile(args.paths)

    if args.action == "status":
        keys = args.dataset or dataset_keys()
        unknown = set(keys) - set(dataset_keys())
        if unknown:
            parser.exit(2, f"{parser.prog}: unknown dataset(s): {', '.join(sorted(unknown))}\n")
        reports = [check(key, profile, args.sample) for key in keys]
        for report in reports:
            state = "ready" if report["ready"] else "MISSING"
            print(f"{state:<8} {report['key']:<22} {'; '.join(report['problems']) or report['metadata']}")
            if not report["ready"]:
                print(f"         raw: {report['raw']}\n         derived: {report['derived']}")
        print(f"paths profile: {profile}")
        return 0 if all(report["ready"] for report in reports) else 1
    if args.action == "download":
        print(f"downloaded raw data into {download_raw(args.dataset, profile)}; "
              f"run `wildmatch prepare status --dataset {args.dataset}` for what is still missing")
        return 0
    if args.action == "jaguar":
        from wildmatch.data.prepare import jaguar as jaguar_module

        rest = [a for a in args.arguments if a != "--"]
        jaguar_module.main(rest, prog=f"{parser.prog} jaguar")
        return 0
    source = load_dataset(UNSEEN_SOURCE, profile)
    output = args.output_dir or Path(str(source.root)) / UNSEEN_OUTPUT
    from wildmatch.data import unseen_split

    argv_unseen = ["--metadata", str(_metadata_path(source)), "--output-dir", str(output),
                   "--label-col", str(source.label_col), "--source-split-col", str(source.split_col),
                   "--dataset", str(source.name), "--database-value", str(source.database_split_value),
                   "--query-value", str(source.query_split_value), "--root", str(source.root), *UNSEEN_ARGS]
    if args.force:
        argv_unseen.append("--force")
    return unseen_split.main(argv_unseen, prog=f"{parser.prog} unseen-split")
