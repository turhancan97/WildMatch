"""``wildmatch demo``: the evaluation pipeline end to end on 24 synthetic lynx renders, CPU only.

The demo data (``metadata.csv`` and ``images/``, built by ``paper/tools/build_demo_data.py``) holds
six synthetic individuals from the CzechLynx synthetic subset (CC BY 4.0, see ATTRIBUTION.md),
three gallery renders and one query each, already masked. The command runs ``wildmatch evaluate``
with the demo as the dataset, the CPU as device, and every output (run directory, caches, run
index) under ``--output`` (default ``demo_runs/``), so it never touches real caches or indexes.
Synthetic coats share one texture model: the demo shows that installation and pipeline work, and
its numbers are not a benchmark. First use downloads the backbone (and, for ``--method loma``, the
LoMa weights).
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path
from typing import List, Optional, Sequence

METHODS = ("cosine", "loma", "rdd-lightglue")


def data_root() -> Path:
    return Path(str(resources.files("wildmatch") / "demo"))


def overrides(method: str, model: str, output: Path) -> List[str]:
    output = output.resolve()
    arguments = [
        "dataset.name=WildMatchDemo",
        "dataset.animal=SyntheticLynx",
        f"dataset.root={data_root()}",
        "dataset.metadata_file=metadata.csv",
        "dataset.label_col=identity",
        "dataset.split_col=split",
        "dataset.database_split_value=database",
        "dataset.query_split_value=query",
        "dataset.no_background=false",
        "dataset.image_variant=no_background",
        "dataset.calibration_size=10",
        f"model.type={model}",
        "model.device=cpu",
        "model.num_workers=0",
        "benchmark.candidate_k=10",
        f"benchmark.cache.dir={output}/cache/features",
        f"output.experiment_root={output}/experiments",
        f"output.run_dir={output}/benchmark_runs",
        f"output.csv_path={output}/benchmark_results.csv",
        f"reporting.index_path={output}/runs.csv",
        "wandb.enabled=false",
    ]
    if method == "cosine":
        arguments.append("benchmark.method=cosine")
    else:
        arguments += [
            "benchmark.method=vismatch",
            f"benchmark.methods.vismatch.matcher={method}",
            "benchmark.methods.vismatch.device=cpu",
            "benchmark.methods.vismatch.checkpoint_source=default",
            f"benchmark.methods.vismatch.cache_dir={output}/cache/vismatch",
        ]
        if method == "loma":
            arguments.append("benchmark.methods.vismatch.loma_arch=LoMa-B")
    return arguments


def main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="cosine",
        help="cosine retrieval, or a default Vismatch matcher over the cosine candidates",
    )
    parser.add_argument(
        "--model",
        default="megadescriptor-t",
        help="backbone (megadescriptor-t downloads ~110 MB; the paper uses megadescriptor-l)",
    )
    parser.add_argument("--output", type=Path, default=Path("demo_runs"))
    args = parser.parse_args(argv)
    from wildmatch.entrypoints import probe

    sys.argv = [prog or "wildmatch demo", "paths=default", *overrides(args.method, args.model, args.output)]
    probe()
    print(
        f"\nDemo finished: run directory and metrics under {args.output.resolve()}/experiments "
        "(synthetic data; not a benchmark)."
    )
    return 0
