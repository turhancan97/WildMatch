"""Compare per-frame feature caches: old vs old (noise floor), old vs new, and each vs the paper's cache."""

import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
PAPER = {
    "loma": Path("/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025/loma-cache"),
    "rdd": Path("/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-cache"),
}


def compare(a: Path, b: Path, label: str) -> None:
    files = sorted(p.relative_to(a) for p in a.rglob("*.npz"))
    if not files:
        print(f"{label}: no frames in {a}")
        return
    exact = missing = 0
    worst: dict[str, float] = {}
    shape_mismatch = 0
    for rel in files:
        if not (b / rel).is_file():
            missing += 1
            continue
        with np.load(a / rel) as x, np.load(b / rel) as y:
            same = set(x.files) == set(y.files)
            for key in sorted(set(x.files) & set(y.files)):
                u, v = x[key], y[key]
                if u.shape != v.shape:
                    shape_mismatch += 1
                    same = False
                    continue
                if not np.array_equal(u, v):
                    same = False
                    if np.issubdtype(u.dtype, np.number):
                        d = float(np.max(np.abs(u.astype(np.float64) - v.astype(np.float64)))) if u.size else 0.0
                        worst[key] = max(worst.get(key, 0.0), d)
            exact += same
    detail = ", ".join(f"{k} {v:.2e}" for k, v in sorted(worst.items()))
    print(f"{label}: {exact}/{len(files)} frames bit-identical, {missing} missing, "
          f"{shape_mismatch} shape mismatches; max |diff| {detail or '-'}")


for backend in ("loma", "rdd"):
    old1, old2, new = (out / f"{backend}_{n}" for n in ("old1", "old2", "new"))
    compare(old1, old2, f"{backend} old1 vs old2")
    compare(old1, new, f"{backend} old1 vs new")
    compare(old1, PAPER[backend], f"{backend} old1 vs paper cache")
    compare(new, PAPER[backend], f"{backend} new vs paper cache")
