"""Small CLI helpers for resolving WildlifeReID configuration values."""

from __future__ import annotations

import argparse
import shlex

from wildmatch.mining.wildlife_dataset import add_config_arguments, config_from_arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = config_from_arguments(args)
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
