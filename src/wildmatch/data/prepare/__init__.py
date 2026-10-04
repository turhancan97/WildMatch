"""Dataset preparation: ``wildmatch prepare status|download|build|finish|compare-masks|jaguar|unseen-split``.

``status`` checks what each registry entry needs on disk (root, metadata, split values, a
sample of image files, the mask column when masks are applied at load time) and prints the
entry's data source (``registry.download``). ``download`` fetches raw data where a public
route exists (WildlifeReID-10k and CzechLynx from Kaggle through wildlife-datasets; Kaggle API
credentials required). ``build`` rebuilds an entry's split table from its source (rules in
:mod:`.sources`) and prints the SAM 3 masking command (GPU, SAM 3 environment); ``finish`` joins
the masks into the metadata file the registry names; ``compare-masks`` measures how far new
masks are from the masked files already on disk. ``jaguar`` runs the JaguarReID preparation steps and ``unseen-split`` rebuilds the
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
DOWNLOADERS = {"WildlifeReID-10k": "WildlifeReID10k", "CzechLynx_v2": "CzechLynx"}
SAM3_SCRIPT = Path("src") / "wildmatch" / "data" / "prepare" / "sam3_masks.py"


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


def _prepare_block(entry) -> Dict[str, object]:
    block = entry.registry.get("prepare")
    return dict(block) if block is not None else {}


def _split_table_path(entry, out: Path) -> Path:
    if _prepare_block(entry)["builder"] == "wildlifereid10k":
        return out / "wildmatch_prepare" / f"{entry.animal}_split.csv"
    return out / "split_time_closed.csv"


def _sam3_arguments(entry, out: Path) -> List[str]:
    block, root = _prepare_block(entry), Path(str(entry.root))
    if block["builder"] == "wildlifereid10k":
        return ["--root", str(root / "images"), "--csv", str(_split_table_path(entry, out)), "--out-dir", str(out),
                "--masks-csv", f"masks_{entry.animal}.csv", "--prompt-column", "prompt"]
    arguments = ["--root", str(out), "--csv", _split_table_path(entry, out).name, "--out-dir", str(out),
                 "--prompt", str(block["prompt"])]
    for item in block.get("threshold_override") or []:
        arguments += ["--threshold-override", str(item)]
    return arguments


def build(key: str, profile: Optional[str], source: Optional[Path], output: Optional[Path], overwrite: bool) -> List[str]:
    """Write the split table (and, for Salamander, copy the images); returns the SAM 3 command."""
    from wildmatch.data.prepare import sources

    entry = load_dataset(key, profile)
    block = _prepare_block(entry)
    builder = block.get("builder")
    out = output or Path(str(entry.root))
    if builder == "wildlifereid10k":
        source = source or Path(str(entry.root))
        metadata = pd.read_csv(source / "metadata.csv", low_memory=False)
        table = sources.wildlifereid10k_table(metadata, str(entry.animal), block.get("include"))
        written = sources.write_new(table, _split_table_path(entry, out), overwrite)
    elif builder == "salamander":
        if source is None:
            raise SystemExit("pass --source <extracted animal-clef-2025 folder> (metadata.csv and images/)")
        metadata = pd.read_csv(source / "metadata.csv")
        table = sources.salamander_table(metadata)
        copied = sources.copy_salamander_images(table, metadata, source, out)
        print(f"copied {copied} images into {out}")
        written = sources.write_new(table, _split_table_path(entry, out), overwrite)
    else:
        raise SystemExit(f"{key} has no build step (prepare.builder: {builder}); see `wildmatch prepare --help`")
    print(f"wrote {written} ({len(table)} rows)")
    return ["python", str(SAM3_SCRIPT), *_sam3_arguments(entry, out), "--segment"]


def finish(key: str, profile: Optional[str], output: Optional[Path], overwrite: bool) -> Path:
    """Join the SAM 3 masks into the metadata file the registry entry reads."""
    from wildmatch.data.prepare import sam3_masks

    entry = load_dataset(key, profile)
    block = _prepare_block(entry)
    out = output or Path(str(entry.root))
    source = pd.read_csv(_split_table_path(entry, out))
    masks_name = f"masks_{entry.animal}.csv" if block["builder"] == "wildlifereid10k" else sam3_masks.MASKS_FILE
    masks = pd.read_csv(out / masks_name)
    mapping = sam3_masks.parse_mapping(str(block.get("split_map") or "")) or None
    metadata = sam3_masks.build_masked_metadata(source, masks, out, split_col=str(entry.split_col),
                                                split_map=mapping, split_map_column=block.get("split_map_column"))
    target = out / str(entry.metadata_file)
    if target.exists() and not overwrite:
        raise SystemExit(f"{target} already exists; pass --overwrite or choose --output-dir")
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(target, index=False)
    return target


def compare_masks(key: str, profile: Optional[str], masks_csv: Path, threshold: int = 12) -> pd.DataFrame:
    """IoU between new SAM 3 masks and the foreground of the masked files the registry entry reads.

    The old foreground is ``max(RGB) > threshold`` on the masked JPEG (the image-quality audit's
    rule), so very dark animal pixels count as background and IoU is a lower bound there.
    """
    import numpy as np
    from PIL import Image

    from wildmatch.data.prepare import sam3_masks

    entry = load_dataset(key, profile)
    root = Path(str(entry.root))
    old = pd.read_csv(_metadata_path(entry))
    prefix = "masked_images/"
    old_by_source = {str(p)[len(prefix):] if str(p).startswith(prefix) else str(p): str(p) for p in old["path"]}
    rows = []
    for record in pd.read_csv(masks_csv).itertuples():
        name = str(record.path)
        if name not in old_by_source:
            continue
        previous = np.asarray(Image.open(root / old_by_source[name]).convert("RGB")).max(axis=2) > threshold
        current = sam3_masks.decode_mask(record.mask)
        if previous.shape != current.shape:
            rows.append({"path": name, "iou": float("nan"), "note": "size differs"})
            continue
        union = np.logical_or(previous, current).sum()
        rows.append({"path": name, "iou": float(np.logical_and(previous, current).sum() / union) if union else 1.0,
                     "old_fg": float(previous.mean()), "new_fg": float(current.mean())})
    return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Check, download or build datasets.")
    parser.add_argument("--paths", help="path profile (default: WILDMATCH_PATHS, wildmatch.local.yaml, default)")
    sub = parser.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", help="what is on disk for each registry entry")
    status.add_argument("--dataset", action="append", default=[], help="registry key (repeatable; default all)")
    status.add_argument("--sample", type=int, default=20, help="image files checked per entry")
    download = sub.add_parser("download", help="fetch raw data where a public route exists")
    download.add_argument("dataset", help="registry key")
    build_parser = sub.add_parser("build", help="rebuild the split table from the source and print the SAM 3 command")
    build_parser.add_argument("dataset", help="registry key")
    build_parser.add_argument("--source", type=Path, help="source folder (WildlifeReID-10k: the dataset root; "
                              "Salamander: the extracted animal-clef-2025 competition folder)")
    finish_parser = sub.add_parser("finish", help="join the SAM 3 masks into the registry's metadata file")
    finish_parser.add_argument("dataset", help="registry key")
    for command in (build_parser, finish_parser):
        command.add_argument("--output-dir", type=Path, help="write here instead of the registry root (trials)")
        command.add_argument("--overwrite", action="store_true")
    compare = sub.add_parser("compare-masks", help="IoU of new SAM 3 masks against the masked files on disk")
    compare.add_argument("dataset", help="registry key")
    compare.add_argument("--masks-csv", type=Path, required=True)
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
    if args.action == "build":
        command = build(args.dataset, profile, args.source, args.output_dir, args.overwrite)
        print("Next, on a GPU node in the SAM 3 environment (A100/H100), from the repository root:\n  "
              + " ".join(command) + f"\nthen: wildmatch prepare finish {args.dataset}"
              + (f" --output-dir {args.output_dir}" if args.output_dir else ""))
        return 0
    if args.action == "finish":
        print(f"wrote {finish(args.dataset, profile, args.output_dir, args.overwrite)}")
        return 0
    if args.action == "compare-masks":
        table = compare_masks(args.dataset, profile, args.masks_csv)
        if table.empty:
            parser.exit(1, f"{parser.prog}: no rows of {args.masks_csv} match {args.dataset}'s metadata\n")
        iou = table["iou"].dropna()
        print(f"{len(table)} images: IoU mean {iou.mean():.3f}, median {iou.median():.3f}, "
              f"share >= 0.9: {(iou >= 0.9).mean():.2f}, share < 0.5: {(iou < 0.5).mean():.2f}")
        print(table.sort_values("iou").head(10).to_string(index=False))
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
