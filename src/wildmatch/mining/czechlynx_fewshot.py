"""Few-shot training views for the CzechLynx time-closed workflow.

Same construction as ``wildmatch.mining.wildlife_fewshot`` (exact global budget, proportional per
identity, never below two frames per identity, seeded nested selections, canonical frame
names preserved so the RDD/LoMa caches of the full view stay valid) applied to the
CzechLynx canonical view: the training split is subsampled frame-wise per identity, the
``val`` and ``test`` splits are copied unchanged.

The probe metadata copy is ``CzechLynx_v2/metadata_fewshot/CzechLynxDataset-Metadata-Real.csv``
with one column ``split_frac<f>_seed<s>`` per view: ``train`` for the kept training
frames, ``unused`` for every other row whose ``split-time_closed`` is ``train`` (dropped
frames and the view's validation holdout), ``test`` unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wildmatch.mining.wildlife_fewshot import (
    DEFAULT_MIN_PER_IDENTITY,
    load_czechlynx_records,
    prepare_fewshot,
    view_name,
)

METADATA_SPLIT_COLUMN = "split-time_closed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fraction", type=float, required=True, help="share of training frames to keep, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_per_identity", type=int, default=DEFAULT_MIN_PER_IDENTITY)
    parser.add_argument("--protocol", default="legacy", help="name component of the output view")
    parser.add_argument(
        "--source_view", type=Path, default=None, help="full canonical view (default: $CZECHLYNX_VIEW_ROOT)"
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="few-shot view (default: $FEWSHOT_ROOT/views/CzechLynx/<protocol>/<view>)",
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=None,
        help="source metadata (default: $CZECHLYNX_DATA_ROOT/CzechLynx_v2/CzechLynxDataset-Metadata-Real.csv)",
    )
    parser.add_argument(
        "--metadata_out",
        type=Path,
        default=None,
        help="probe metadata copy (default: <CzechLynx_v2>/metadata_fewshot/CzechLynxDataset-Metadata-Real.csv; 'none' disables)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(os.environ.get("CZECHLYNX_DATA_ROOT", "/shared/sets/datasets/vision/czechlynx"))
    source_view = args.source_view or Path(
        os.environ.get("CZECHLYNX_VIEW_ROOT", data_root / "CzechLynx_processed_time_closed")
    )
    fewshot_root = Path(os.environ.get("FEWSHOT_ROOT", data_root / "fewshot"))
    output_root = args.output_root or fewshot_root / "views" / "CzechLynx" / args.protocol / view_name(
        args.fraction, args.seed
    )
    metadata_csv = args.metadata_csv or data_root / "CzechLynx_v2" / "CzechLynxDataset-Metadata-Real.csv"
    if args.metadata_out is None:
        metadata_out: Path | None = metadata_csv.parent / "metadata_fewshot" / metadata_csv.name
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
        metadata_csv=metadata_csv,
        metadata_out=metadata_out,
        force=args.force,
        dry_run=args.dry_run,
        loader=load_czechlynx_records,
        split_column_name=METADATA_SPLIT_COLUMN,
    )
    printable = {key: value for key, value in summary.items() if key != "config"}
    printable["fewshot"] = {k: v for k, v in summary["fewshot"].items() if k != "per_identity"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
