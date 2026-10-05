"""Small CLI helpers for resolving WildlifeReID configuration values."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from scripts.wildlife_dataset import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    values = {
        "WILDLIFE_DATASET_ID": config.dataset_id,
        "WILDLIFE_SOURCE_ROOT": str(config.root),
        "WILDLIFE_METADATA_CSV": str(config.metadata_path),
    }
    if args.shell:
        for key, value in values.items():
            print(f"{key}={shlex.quote(value)}")
    else:
        for key, value in values.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
