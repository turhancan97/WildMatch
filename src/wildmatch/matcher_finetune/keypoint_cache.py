"""
On-disk cache of RDD's per-frame output (keypoints + descriptors), so a run
with a FROZEN RDD (`--trained_model lg`) never re-detects the same frame twice.

Why this exists
---------------
With `--trained_model lg` the RDD forward in the training step runs under
`no_grad` (see train_by_lg_matches.train_epoch_lg) and its weights never
change, so for a given frame it recomputes the same features every time that
frame comes up. Measured on an H100 at --resize 512 / --top_k 512, that forward
is ~65% of a training step (247 ms of 379 ms at batch 8) and ~80% of a
pseudo-accuracy eval batch, while the whole train split is only 116k distinct
frames — a 600-epoch index-40 run performs ~15.2M frame-forwards over them,
i.e. it recomputes each frame ~130 times.

What is cached
--------------
Exactly what `train_common.extract_train` returns for one image — `keypoints`
(N, 2) in resized-image pixel coordinates and `descriptors` (N, D) — plus the
resized (H, W) those coordinates live in, which `batch_features` needs for
LightGlue's keypoint normalization and which no longer arrives with the image.

Descriptors are stored fp16. The rounding error that costs (measured max 9e-5
on unit-norm descriptors) is *below* the run-to-run noise the live model
already has: RDD's output is not bit-stable across batch composition (max 1.4e-4
on descriptors, ~3e-3 px on keypoint positions, and the keypoint count itself
moves by +/-1 on frames with detections sitting on `detection_threshold`), so a
cached frame is no further from "what this step would have computed" than the
same frame in a differently-shuffled batch would have been. Keypoint
coordinates stay fp32 — fp16 would cost ~0.12 px there, well above that noise
floor, and they are only 0.8% of the bytes.

Cost: ~204 KiB/frame (mean 404 keypoints x (2 fp32 + 256 fp16)), so ~24 GB for
the 116k-frame train split, ~31 GB for train+test.

Validity
--------
The cache is only sound while RDD is frozen and sees the same input, so
`CacheSpec` pins everything that would change the answer — the RDD weights
(by content hash), `--resize`, `--top_k` and the detection threshold — and
opening a cache with a mismatched spec is a hard error rather than a silent
wrong-features run. Photometric `--augment` and `--multi_scale_*` change the
image RDD would have seen, so they are rejected by the training script when the
cache is on; `--keypoint_dropout` is fine, it is applied to features after
extraction.

Layout
------
One `.npz` per frame, mirroring the dataset tree, plus a `manifest.json` at the
root::

    <cache_root>/manifest.json
    <cache_root>/train/lynx_1/GoPo_95a/0009/frame_0000.npz

Per-frame files (rather than one packed blob) keep the build resumable and
trivially shardable across GPUs, and let a partial cache be diagnosed by simply
listing the tree.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

MANIFEST_NAME = "manifest.json"


def weights_fingerprint(path: str | Path) -> str:
    """SHA-256 of a weights file — what ties a cache to the RDD that built it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class CacheSpec:
    """Every setting that changes what RDD would produce for a given frame.

    Compared field-by-field when a cache is opened; a mismatch means the cached
    features are not the ones this run would have computed, which is a bug, not
    a warning.
    """
    rdd_weights_sha256: str
    resize: int
    top_k: int
    detection_threshold: float
    descriptor_dim: int

    def assert_matches(self, wanted: "CacheSpec", cache_root: Path) -> None:
        """Raise unless `wanted` (what the run needs) equals `self` (what the cache holds)."""
        mismatched = {
            f: (getattr(self, f), getattr(wanted, f))
            for f in self.__dataclass_fields__
            if getattr(self, f) != getattr(wanted, f)
        }
        if mismatched:
            lines = "\n".join(
                f"    {f}: cache has {have!r}, this run wants {want!r}"
                for f, (have, want) in mismatched.items()
            )
            raise ValueError(
                f"keypoint cache at {cache_root} was built with different settings:\n"
                f"{lines}\n"
                "  Rebuild it (contrastive_finetuning.build_keypoint_cache) or fix the flags."
            )


class KeypointCache:
    """Read/write access to a keypoint cache directory.

    Args:
        root: Cache directory (holds manifest.json and the mirrored frame tree).
        spec: When given, the spec this run needs; checked against the
            manifest's on open. Omit only when building the cache.
    """

    @classmethod
    def for_writing(cls, root: str | Path, spec: CacheSpec) -> "KeypointCache":
        """A cache handle for a directory that has no manifest yet (build time)."""
        obj = cls.__new__(cls)
        obj.root = Path(root)
        obj.spec = spec
        obj.manifest = None
        obj._image_hw_cache = {}
        return obj

    def __init__(self, root: str | Path, spec: CacheSpec | None = None) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {self.root} — not a keypoint cache "
                "(build one with `python -m contrastive_finetuning.build_keypoint_cache`)"
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.spec = CacheSpec(**self.manifest["spec"])
        self._image_hw_cache: dict[str, tuple[int, int]] = {}
        if spec is not None:
            self.spec.assert_matches(spec, self.root)

    # ── paths ────────────────────────────────────────────────────────────────
    def path_for(self, rel: str) -> Path:
        """Cache file for a dataset-relative frame path ('train/lynx_1/.../f.jpg')."""
        return self.root / Path(rel).with_suffix(".npz")

    def has(self, rel: str) -> bool:
        return self.path_for(rel).exists()

    def image_hw(self, rel: str) -> tuple[int, int]:
        """Return the cached resized image size without loading features."""
        cached = self._image_hw_cache.get(rel)
        if cached is not None:
            return cached
        path = self.path_for(rel)
        try:
            with np.load(path, allow_pickle=False) as z:
                hw = z["hw"]
        except FileNotFoundError:
            raise KeyError(
                f"frame {rel!r} is not in the keypoint cache at {self.root} "
                f"(expected {path})"
            ) from None
        if hw.shape != (2,):
            raise ValueError(f"{path} has invalid image size shape {hw.shape}")
        value = (int(hw[0]), int(hw[1]))
        self._image_hw_cache[rel] = value
        return value

    # ── read ─────────────────────────────────────────────────────────────────
    def load_padded(self, rel: str) -> dict[str, torch.Tensor]:
        """One frame's features, zero-padded to `top_k` so the default collate stacks them.

        Returns keypoints (top_k, 2) fp32, descriptors (top_k, D) fp16, the
        valid keypoint count, and the resized (H, W) the coordinates refer to.
        `unpad_cached_features` turns a collated batch of these back into the
        ragged list-of-dicts `batch_features` consumes — the same shape
        `extract_train` returns, so nothing downstream has to know which of the
        two produced it.

        Padding rather than ragged tensors keeps torch's default_collate (and
        its pin_memory path) working untouched; at a mean 404 of 512 keypoints
        the wasted transfer is ~0.13 MB per frame, against the ~10 ms of RDD it
        replaces.
        """
        path = self.path_for(rel)
        try:
            with np.load(path) as z:
                k, d, hw = z["k"], z["d"], z["hw"]
        except FileNotFoundError:
            raise KeyError(
                f"frame {rel!r} is not in the keypoint cache at {self.root} "
                f"(expected {path}). The cache must cover every frame the run can "
                "draw — with --random_negative_prob > 0 that is the whole candidate "
                "split, not just the index's frames."
            ) from None

        n, top_k = k.shape[0], self.spec.top_k
        if n > top_k:
            raise ValueError(f"{path} holds {n} keypoints, more than top_k={top_k}")
        kp = torch.zeros(top_k, 2, dtype=torch.float32)
        de = torch.zeros(top_k, self.spec.descriptor_dim, dtype=torch.float16)
        kp[:n] = torch.from_numpy(k)
        de[:n] = torch.from_numpy(d)
        return {
            "keypoints": kp,
            "descriptors": de,
            "n_keypoints": torch.tensor(n, dtype=torch.long),
            "image_hw": torch.from_numpy(hw.astype(np.int64)),
        }

    def load_padded_stack(self, rels: list[str]) -> dict[str, torch.Tensor]:
        """`load_padded` over several frames, stacked along a new leading dim.

        Used for the K>1 negative list and for a pseudo-accuracy query's whole
        candidate pool, mirroring how those stack images today.
        """
        items = [self.load_padded(r) for r in rels]
        return {key: torch.stack([it[key] for it in items]) for key in items[0]}

    # ── write ────────────────────────────────────────────────────────────────
    def save(self, rel: str, keypoints: torch.Tensor, descriptors: torch.Tensor,
             image_hw: tuple[int, int]) -> None:
        path = self.path_for(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name and rename, so an interrupted build never leaves
        # a truncated .npz that a later run would happily load.
        tmp = path.with_suffix(".npz.tmp")
        # Written through a file object, not a path: np.savez appends ".npz" to
        # any path that doesn't already end in it, which would leave the temp
        # file under a name the rename below never finds.
        with open(tmp, "wb") as f:
            np.savez(
                f,
                k=keypoints.detach().cpu().numpy().astype(np.float32),
                d=descriptors.detach().cpu().numpy().astype(np.float16),
                hw=np.asarray(image_hw, dtype=np.int32),
            )
        tmp.replace(path)

    @staticmethod
    def write_manifest(root: str | Path, spec: CacheSpec, extra: dict) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        with open(root / MANIFEST_NAME, "w") as f:
            json.dump({"spec": asdict(spec), **extra}, f, indent=2)


def open_cache_for_run(
    root: str | Path, rdd_weights: str | Path, resize: int, top_k: int,
) -> KeypointCache:
    """Open a cache, checked against what this run's flags imply RDD would do.

    The detection threshold comes from the same config `models.build_rdd` reads
    (nothing here needs the model instantiated), and the descriptor width from
    LightGlue's `input_dim` — 256, see `models.build_masked_lg` — which is what
    the run can actually consume regardless of what is on disk.
    """
    from rdd.RDD.utils import read_config  # local: pulls in the RDD package

    return KeypointCache(
        root,
        spec=CacheSpec(
            rdd_weights_sha256=weights_fingerprint(rdd_weights),
            resize=resize,
            top_k=top_k,
            detection_threshold=float(read_config("rdd/configs/default.yaml")["detection_threshold"]),
            descriptor_dim=256,
        ),
    )


def is_cached_batch(item) -> bool:
    """True when a DataLoader element is cached features rather than an image tensor."""
    return isinstance(item, dict) and "descriptors" in item


def unpad_cached_features(
    batched: dict[str, torch.Tensor], device: torch.device,
) -> tuple[list[dict], list[int], list[int]]:
    """Collated cache output -> features and per-frame H/W lists.

    Accepts any number of leading batch dims — (B, ...) for query/positive and
    (B, K, ...) for the K-negative and pseudo-accuracy-candidate stacks — and
    flattens them row-major, which is the same layout the image path produces
    via `negatives.reshape(-1, *shape[2:])`.

    Cached frames can have different aspect ratios, so one DataLoader batch
    may contain several resized image sizes. Keep the dimensions per frame;
    the LightGlue wrapper partitions pair batches by their two image sizes
    before forwarding them.
    """
    n = batched["n_keypoints"].reshape(-1)
    hw = batched["image_hw"].reshape(-1, 2)
    d_dim = batched["descriptors"].shape[-1]
    kpts = batched["keypoints"].reshape(-1, batched["keypoints"].shape[-2], 2).to(device, non_blocking=True)
    desc = batched["descriptors"].reshape(-1, batched["descriptors"].shape[-2], d_dim).to(device, non_blocking=True)
    # Descriptors are stored fp16; LightGlue runs in fp32 like the live path.
    desc = desc.float()
    counts = n.tolist()
    feats = [
        {"keypoints": kpts[i, :c], "descriptors": desc[i, :c]}
        for i, c in enumerate(counts)
    ]
    return feats, hw[:, 0].tolist(), hw[:, 1].tolist()
