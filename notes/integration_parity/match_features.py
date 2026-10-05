"""Order-free feature comparison: nearest-neighbour keypoint matching, then descriptor cosine."""

import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
PAPER = {
    "loma": Path("/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025/loma-cache"),
    "rdd": Path("/shared/sets/datasets/vision/czechlynx/checkpoints/wildlife-reid-10k/SalamanderID2025/rdd-cache"),
}


def load(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    kp = d.get("keypoints", d.get("k")).astype(np.float64).reshape(-1, 2)
    de = d.get("descriptors", d.get("d")).astype(np.float64)
    de = de.reshape(kp.shape[0], -1) if de.shape[0] != kp.shape[0] else de
    size = d.get("image_size", d.get("hw"))
    return kp, de, d, size


def compare(a: Path, b: Path, label: str) -> None:
    rows = []
    for rel in sorted(p.relative_to(a) for p in a.rglob("*.npz")):
        ka, da, ra, size = load(a / rel)
        kb, db, rb, _ = load(b / rel)
        scale = 1.0
        if np.abs(ka).max() <= 1.5 and size is not None:  # normalized [-1, 1] -> pixels of the processed image
            h, w = (float(x) for x in np.asarray(size).ravel()[:2])
            ka = (ka + 1) / 2 * [w, h]
            kb = (kb + 1) / 2 * [w, h]
        dist = np.sqrt(((ka[:, None, :] - kb[None, :, :]) ** 2).sum(-1))
        j = dist.argmin(1)
        near = dist[np.arange(len(ka)), j]
        ok = near < 0.5
        na = da / np.linalg.norm(da, axis=1, keepdims=True).clip(1e-12)
        nb = db / np.linalg.norm(db, axis=1, keepdims=True).clip(1e-12)
        cos = (na[ok] * nb[j[ok]]).sum(1) if ok.any() else np.array([np.nan])
        rows.append((len(ka), len(kb), ok.mean(), np.median(near), float(np.nanmin(cos)), float(np.nanmedian(cos))))
    r = np.array(rows, dtype=float)
    print(f"{label}: {len(r)} frames; keypoints {r[:,0].mean():.0f} vs {r[:,1].mean():.0f}; "
          f"matched within 0.5 px {100*r[:,2].mean():.2f}% (worst frame {100*r[:,2].min():.1f}%); "
          f"median NN distance {np.median(r[:,3]):.2e} px; descriptor cosine median {np.nanmedian(r[:,5]):.6f}, "
          f"min {np.nanmin(r[:,4]):.4f}")


for backend in ("loma", "rdd"):
    old1, old2, new = (out / f"{backend}_{n}" for n in ("old1", "old2", "new"))
    compare(old1, new, f"{backend} old vs new  ")
    compare(old1, PAPER[backend], f"{backend} old vs paper")
    compare(new, PAPER[backend], f"{backend} new vs paper")
