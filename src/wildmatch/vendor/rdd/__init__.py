"""RDD (xtcpete/rdd, Apache-2.0), vendored unchanged from the checkout matcher fine-tuning used.

See VENDORED.md for the source commit, the files kept and why.
"""

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"
RDD_WEIGHTS = "RDD-v2.pth"
LG_WEIGHTS = "RDD_lg-v2.pth"


def resolve_weights(value, filename: str) -> str:
    """An explicit weights path, else `filename` in the path profile's `external.rdd_weights_dir`."""
    if value:
        return str(value)
    from wildmatch.paths import path

    folder = path("external.rdd_weights_dir")
    if folder is None:
        raise SystemExit(
            f"no RDD weights given: pass the path explicitly or set external.rdd_weights_dir in the "
            f"path profile (WILDMATCH_RDD_WEIGHTS_DIR for the default profile) to the folder with {filename}"
        )
    return str(folder / filename)
