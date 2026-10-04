"""Bring the Kaggle Jaguar Re-ID data into the shared dataset format as ``JaguarReID`` (CPU).

The Kaggle release has one labelled table (``train.csv``: ``filename``, ``ground_truth``)
and RGBA PNGs whose alpha channel removes the background. The RGB channels still hold
the original background under transparent pixels, and the shared loader reads images
with OpenCV's default flag, which drops alpha, so the files cannot be used directly.
Kaggle's ``test.csv`` is an unlabelled pair list and is not used.

The photos contain long bursts of near-identical frames, so a per-image random split puts
copies of a query in the gallery. The split therefore keeps whole bursts on one side.
Three steps, all writing into the dataset root:

``prepare``
    ``masked_images/<filename>`` (RGB multiplied by alpha, black background) and ``jaguar_reid_base.csv``: one row per
    labelled image with the masked ``path``, ``original_path``, ``identity``, a COCO-RLE
    ``mask`` from the alpha channel, a 256-bit difference hash ``dhash`` and the masked
    file's SHA-256; plus ``jaguar_reid_base_manifest.json``.

``embed``
    DINOv2-small CLS embeddings (the repository's ``dinov2`` backbone, a generic model
    that is not the MegaDescriptor backbone being evaluated) of every masked image, on
    the CPU: ``jaguar_reid_dinov2_small_cls.npz``.

``split``
    Joins photos of the same jaguar into one burst group when their hash distance is at
    most ``--threshold``, when their files are adjacent (``train_0688``/``train_0689``; the
    Kaggle file order follows capture order within a jaguar) and their embedding cosine is
    at least ``--adjacent-cos``, or when their cosine is at least ``--any-cos`` wherever they
    sit. Groups are joined transitively; per jaguar, whole groups go to ``query`` until about
    ``--query-ratio`` of its photos are there, and every jaguar stays on both sides. Writes
    ``jaguar_reid_v2_no_background.csv`` (split column ``split_v2``, plus
    ``split_train_test`` train/test for the lynx-finetuning wildlife pipeline) and
    ``jaguar_reid_v2_manifest.json``. The ``v2`` names are kept because the probe runs made
    on this split record that file and use ``split_v2`` as their split protocol.

Everything fails closed: an image without alpha, a near-identical pair under two labels, a
jaguar that cannot fill both sides, or embeddings that do not match the base table.

Example::

    python scripts/prepare_jaguar_metadata.py prepare
    python scripts/prepare_jaguar_metadata.py embed
    python scripts/prepare_jaguar_metadata.py split --threshold 32 --adjacent-cos 0.85 --any-cos 0.95
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from wildmatch.data.registry import load_dataset as _load_dataset  # noqa: E402

DEFAULT_ROOT = Path(str(_load_dataset("jaguar").root))  # registry entry `jaguar`, active path profile
TRAIN_CSV = "train.csv"
IMAGE_DIR = Path("train") / "train"
MASKED_DIR = "masked_images"
BASE_NAME = "jaguar_reid_base.csv"
BASE_MANIFEST_NAME = "jaguar_reid_base_manifest.json"
METADATA_NAME = "jaguar_reid_v2_no_background.csv"
MANIFEST_NAME = "jaguar_reid_v2_manifest.json"
EMBEDDINGS_NAME = "jaguar_reid_dinov2_small_cls.npz"
EMBEDDING_SPEC = ("dinov2 (facebook/dinov2-with-registers-small) post-layernorm CLS, L2-normalised; "
                  "masked image thumbnailed to 448 px, resized to 224x224, ImageNet normalisation; CPU")
SPLIT_COL = "split_v2"
OVERSHOOT = 0.2  # allowed relative overshoot of the per-jaguar query target
HASH_SIZE = 16  # 16 x 16 = 256-bit difference hash
HASH_SPEC = "dhash-256 of alpha-masked luminance, image reduced by 8 then bilinear 17x16"
BASE_COLUMNS = ["image_id", "identity", "path", "original_path", "mask", "dhash", "masked_sha256"]
METADATA_COLUMNS = [
    "image_id", "identity", "path", "original_path", "mask", "dup_group", "dhash",
    SPLIT_COL, "split_train_test", "masked_sha256",
]


class PreparationError(RuntimeError):
    """Raised when the source data violates an assumption; nothing is half-written."""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise PreparationError(f"{path} has mode {image.mode}; expected RGBA with an alpha mask")
        return np.asarray(image.convert("RGBA"))


def masked_rgb(rgba: np.ndarray) -> np.ndarray:
    """RGB multiplied by alpha, truncated to uint8, as the Kaggle runner wrote it."""
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgb * alpha).astype(np.uint8)


def alpha_mask(rgba: np.ndarray) -> np.ndarray:
    return rgba[..., 3] > 0


def dhash(rgba: np.ndarray, size: int = HASH_SIZE) -> np.ndarray:
    image = Image.fromarray(masked_rgb(rgba))
    factor = max(1, min(image.size) // (size * 4))
    if factor > 1:
        image = image.reduce(factor)
    grey = np.asarray(image.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (grey[:, 1:] > grey[:, :-1]).flatten()


def bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def hex_to_bits(text: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8)).astype(bool)


def encode_mask(mask: np.ndarray) -> str:
    """Boolean mask -> COCO-RLE JSON string (same layout as scripts/segment_with_sam3.py)."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return json.dumps(rle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Source table
# ---------------------------------------------------------------------------


def read_train_table(root: Path) -> List[Dict[str, str]]:
    path = root / TRAIN_CSV
    if not path.is_file():
        raise PreparationError(f"missing {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"filename", "ground_truth"}:
        raise PreparationError(f"{path} must have exactly the columns filename, ground_truth")
    names = [row["filename"] for row in rows]
    if len(set(names)) != len(names):
        raise PreparationError(f"{path} lists a filename twice")
    for row in rows:
        if not row["filename"].strip() or not row["ground_truth"].strip():
            raise PreparationError(f"{path} has an empty filename or identity: {row}")
        if not (root / IMAGE_DIR / row["filename"]).is_file():
            raise PreparationError(f"image listed in {path} does not exist: {IMAGE_DIR / row['filename']}")
    return rows


def _hash_one(path: str) -> str:
    return bits_to_hex(dhash(load_rgba(Path(path))))


def compute_hashes(root: Path, rows: Sequence[Mapping[str, str]], workers: int) -> List[str]:
    paths = [str(root / IMAGE_DIR / row["filename"]) for row in rows]
    if workers <= 1:
        return [_hash_one(path) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_hash_one, paths, chunksize=8))


def distance_matrix(hashes: Sequence[str]) -> np.ndarray:
    bits = np.stack([hex_to_bits(h) for h in hashes]).astype(np.uint8)
    # Hamming distance via dot products: d = |a| + |b| - 2 a.b
    ones = bits.sum(1).astype(np.int32)
    inner = bits.astype(np.int32) @ bits.T.astype(np.int32)
    return ones[:, None] + ones[None, :] - 2 * inner


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_split(
    identities: Sequence[str],
    groups: Sequence[int],
    order_keys: Sequence[str],
    *,
    query_ratio: float,
    min_query: int,
    seed: int,
) -> List[str]:
    """Per jaguar, move whole groups to ``query`` until the target is met.

    Groups are ordered by their smallest ``order_key`` and shuffled with one generator
    seeded by ``seed``, visiting jaguars in sorted order, so the split is deterministic.
    Groups that would push the query side more than ``OVERSHOOT`` above the target are
    skipped, and the database side always keeps at least one image of every jaguar.
    """
    if not 0.0 < query_ratio < 1.0:
        raise PreparationError("query_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    by_identity: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (identity, group) in enumerate(zip(identities, groups)):
        by_identity[identity][group].append(index)
    split = ["database"] * len(identities)
    for identity in sorted(by_identity):
        members = by_identity[identity]
        ordered = sorted(members.values(), key=lambda idx: min(order_keys[i] for i in idx))
        if len(ordered) < 2:
            raise PreparationError(f"jaguar {identity} has a single burst group; it cannot fill both sides")
        permutation = rng.permutation(len(ordered))
        shuffled = [ordered[k] for k in permutation]
        total = sum(len(g) for g in shuffled)
        target = max(int(min_query), int(round(total * query_ratio)))
        ceiling = max(target, int(np.ceil(target * (1.0 + OVERSHOOT))))
        taken = 0
        left_in_database = total
        chosen = set()
        # First pass: only groups that keep the query side within the ceiling.
        for position, group_members in enumerate(shuffled):
            if taken >= target:
                break
            size = len(group_members)
            if taken + size > ceiling or left_in_database - size < 1:
                continue
            chosen.add(position)
            taken += size
            left_in_database -= size
        # Second pass, only if the minimum is not met: smallest remaining groups first.
        for position in sorted(set(range(len(shuffled))) - chosen, key=lambda k: (len(shuffled[k]), k)):
            if taken >= min_query:
                break
            size = len(shuffled[position])
            if left_in_database - size < 1:
                continue
            chosen.add(position)
            taken += size
            left_in_database -= size
        for position in chosen:
            for index in shuffled[position]:
                split[index] = "query"
        if taken == 0:
            raise PreparationError(f"jaguar {identity} received no query image")
    return split


# ---------------------------------------------------------------------------
# prepare: masked images and the base table
# ---------------------------------------------------------------------------


def _write_masked(task: Tuple[str, str]) -> Tuple[str, str, Dict[str, int]]:
    src, dst = Path(task[0]), Path(task[1])
    rgba = load_rgba(src)
    alpha = rgba[..., 3]
    stats = {"nonbinary_alpha_pixels": int(((alpha > 0) & (alpha < 255)).sum()),
             "foreground_pixels": int((alpha > 0).sum()), "pixels": int(alpha.size)}
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(masked_rgb(rgba)).save(tmp, format="PNG")
    os.replace(tmp, dst)
    return encode_mask(alpha_mask(rgba)), sha256_file(dst), stats


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _count_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def prepare(root: Path, *, workers: int) -> Dict[str, object]:
    rows = sorted(read_train_table(root), key=lambda row: row["filename"])
    hashes = compute_hashes(root, rows, workers)
    masked_dir = root / MASKED_DIR
    masked_dir.mkdir(exist_ok=True)
    tasks = [(str(root / IMAGE_DIR / row["filename"]), str(masked_dir / row["filename"])) for row in rows]
    if workers <= 1:
        results = [_write_masked(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_write_masked, tasks, chunksize=4))
    base_rows = [{
        "image_id": Path(row["filename"]).stem,
        "identity": row["ground_truth"],
        "path": f"{MASKED_DIR}/{row['filename']}",
        "original_path": (IMAGE_DIR / row["filename"]).as_posix(),
        "mask": mask,
        "dhash": hashes[index],
        "masked_sha256": masked_sha,
    } for index, (row, (mask, masked_sha, _)) in enumerate(zip(rows, results))]
    base_path = root / BASE_NAME
    _write_csv(base_path, BASE_COLUMNS, base_rows)
    test_csv = root / "test.csv"
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/prepare_jaguar_metadata.py prepare",
        "dataset_root": str(root),
        "source": {"train_csv": TRAIN_CSV, "train_csv_sha256": sha256_file(root / TRAIN_CSV),
                   "image_dir": IMAGE_DIR.as_posix()},
        "excluded": {
            "kaggle_test": "test.csv is an unlabelled pair list; its images are not used",
            "kaggle_test_rows": _count_rows(test_csv) if test_csv.is_file() else None,
        },
        "hash": HASH_SPEC,
        "counts": {"images": len(base_rows), "identities": len({r["identity"] for r in base_rows})},
        "alpha": {
            "images_with_nonbinary_alpha": sum(1 for _, _, st in results if st["nonbinary_alpha_pixels"]),
            "mean_foreground_fraction": float(np.mean([st["foreground_pixels"] / st["pixels"] for _, _, st in results])),
        },
        "outputs": {"base": BASE_NAME, "base_sha256": sha256_file(base_path), "masked_dir": MASKED_DIR},
    }
    (root / BASE_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_base(root: Path) -> List[Dict[str, str]]:
    path = root / BASE_NAME
    if not path.is_file():
        raise PreparationError(f"missing {path}; run the prepare step first")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = set(BASE_COLUMNS) - set(rows[0]) if rows else set(BASE_COLUMNS)
    if missing:
        raise PreparationError(f"{path} lacks columns {sorted(missing)}")
    return sorted(rows, key=lambda row: row["image_id"])


# ---------------------------------------------------------------------------
# embed and split
# ---------------------------------------------------------------------------


def file_number(image_id: str) -> int:
    """``train_0688`` -> 688; the Kaggle file order follows capture order within a jaguar."""
    try:
        return int(image_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise PreparationError(f"image_id {image_id!r} does not end in _<number>") from error


def compute_embeddings(root: Path, rows: Sequence[Mapping[str, str]], workers: int) -> np.ndarray:
    import torch
    import torchvision.transforms as T

    from wildmatch.models.model import get_model

    model, _, mean, std, img_size, *_ = get_model("dinov2")
    model.eval()
    transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])

    class _Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(root / rows[index]["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((448, 448))
                return transform(image)

    loader = torch.utils.data.DataLoader(_Images(), batch_size=32, num_workers=max(0, workers))
    chunks = []
    with torch.no_grad():
        for batch in loader:
            chunks.append(torch.nn.functional.normalize(model(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def write_embeddings(root: Path, workers: int) -> Path:
    rows = read_base(root)
    embeddings = compute_embeddings(root, rows, workers)
    out = root / EMBEDDINGS_NAME
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, image_id=np.array([row["image_id"] for row in rows]), emb=embeddings,
             spec=np.array(EMBEDDING_SPEC))
    os.replace(tmp, out)
    return out


def load_embeddings(path: Path, image_ids: Sequence[str]) -> np.ndarray:
    if not path.is_file():
        raise PreparationError(f"missing {path}; run the embed step first")
    payload = np.load(path)
    if list(payload["image_id"].astype(str)) != list(image_ids):
        raise PreparationError(f"{path} does not cover the base table's images in the same order")
    embeddings = payload["emb"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise PreparationError(f"{path} holds embeddings that are not L2-normalised")
    return embeddings


def burst_groups(
    distances: np.ndarray,
    similarity: np.ndarray,
    numbers: Sequence[int],
    identities: Sequence[str],
    *,
    threshold: int,
    adjacent_gap: int,
    adjacent_cos: float,
    any_cos: float,
    cross_identity_limit: int = 12,
) -> List[int]:
    """Union-find over same-jaguar pairs that look like one burst (see the module docstring)."""
    labels = np.asarray(identities)
    same = np.equal.outer(labels, labels)
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    crossing = np.argwhere(upper & ~same & (distances <= cross_identity_limit))
    if len(crossing):
        sample = ", ".join(f"{i}/{j} ({identities[i]} vs {identities[j]})" for i, j in crossing[:5])
        raise PreparationError(
            f"{len(crossing)} near-identical pairs at distance <= {cross_identity_limit} carry different "
            f"labels (row indices {sample}); check them for label errors"
        )
    gap = np.abs(np.subtract.outer(np.asarray(numbers), np.asarray(numbers)))
    link = upper & same & (
        (distances <= threshold) | ((gap <= adjacent_gap) & (similarity >= adjacent_cos)) | (similarity >= any_cos)
    )
    parent = list(range(len(identities)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(link):
        root_i, root_j = find(int(i)), find(int(j))
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)
    return [find(i) for i in range(len(identities))]


def burst_leakage(similarity: np.ndarray, numbers: Sequence[int], identities: Sequence[str],
                  split: Sequence[str], *, adjacent_gap: int, adjacent_cos: float, any_cos: float) -> Dict[str, object]:
    labels = np.asarray(identities)
    sides = np.asarray(split)
    gap = np.abs(np.subtract.outer(np.asarray(numbers), np.asarray(numbers)))
    across = np.triu(np.equal.outer(labels, labels) & np.not_equal.outer(sides, sides), k=1)
    query = np.flatnonzero(sides == "query")
    database = np.flatnonzero(sides == "database")
    same_q_db = np.equal.outer(labels[query], labels[database])
    nearest = np.where(same_q_db, similarity[np.ix_(query, database)], -1.0).max(1)
    return {
        "cross_side_adjacent_similar_pairs": int((across & (gap <= adjacent_gap) & (similarity >= adjacent_cos)).sum()),
        "cross_side_pairs_at_any_cos": int((across & (similarity >= any_cos)).sum()),
        "query_nearest_same_identity_database_cos_percentiles": {
            str(p): round(float(np.percentile(nearest, p)), 4) for p in (10, 50, 90, 100)
        },
    }


def split_leakage(distances: np.ndarray, identities: Sequence[str], split: Sequence[str],
                  bands: Sequence[int] = (8, 16, 24, 32)) -> Dict[str, int]:
    """Same-jaguar pairs that end up on opposite sides, per maximum hash distance."""
    labels = np.asarray(identities)
    sides = np.asarray(split)
    across = np.triu(np.equal.outer(labels, labels) & np.not_equal.outer(sides, sides), k=1)
    return {f"cross_side_same_identity_pairs_le_{b}": int((across & (distances <= b)).sum()) for b in bands}


def split_dataset(root: Path, *, threshold: int, adjacent_gap: int, adjacent_cos: float, any_cos: float,
            query_ratio: float, min_query: int, seed: int, cross_identity_limit: int = 12) -> Dict[str, object]:
    rows = read_base(root)
    image_ids = [row["image_id"] for row in rows]
    identities = [row["identity"] for row in rows]
    numbers = [file_number(image_id) for image_id in image_ids]
    embeddings = load_embeddings(root / EMBEDDINGS_NAME, image_ids)
    similarity = embeddings @ embeddings.T
    distances = distance_matrix([row["dhash"] for row in rows])
    groups = burst_groups(distances, similarity, numbers, identities, threshold=threshold,
                          adjacent_gap=adjacent_gap, adjacent_cos=adjacent_cos, any_cos=any_cos,
                          cross_identity_limit=cross_identity_limit)
    split = assign_split(identities, groups, image_ids, query_ratio=query_ratio, min_query=min_query, seed=seed)

    group_names: Dict[int, str] = {}
    for index in range(len(rows)):
        group_names.setdefault(groups[index], f"b{len(group_names):04d}")
    out_rows = []
    for index, row in enumerate(rows):
        out = {column: row[column] for column in BASE_COLUMNS}
        out["dup_group"] = group_names[groups[index]]
        out[SPLIT_COL] = split[index]
        out["split_train_test"] = "train" if split[index] == "database" else "test"
        out_rows.append(out)
    metadata_path = root / METADATA_NAME
    _write_csv(metadata_path, METADATA_COLUMNS, out_rows)
    for row in out_rows:  # bookkeeping below reads "split"
        row["split"] = row[SPLIT_COL]

    group_sizes: Dict[str, int] = defaultdict(int)
    sides_per_group: Dict[str, set] = defaultdict(set)
    for row in out_rows:
        group_sizes[row["dup_group"]] += 1
        sides_per_group[row["dup_group"]].add(row["split"])
    per_identity_query = defaultdict(lambda: [0, 0])
    for row in out_rows:
        per_identity_query[row["identity"]][0] += row["split"] == "query"
        per_identity_query[row["identity"]][1] += 1
    shares = [q / n for q, n in per_identity_query.values()]
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "scripts/prepare_jaguar_metadata.py split",
        "dataset_root": str(root),
        "source": {"base": BASE_NAME, "base_sha256": sha256_file(root / BASE_NAME),
                   "embeddings": EMBEDDINGS_NAME, "embeddings_sha256": sha256_file(root / EMBEDDINGS_NAME),
                   "embedding_spec": EMBEDDING_SPEC},
        "parameters": {
            "split_column": SPLIT_COL, "hash": HASH_SPEC, "hash_threshold": threshold, "adjacent_file_gap": adjacent_gap,
            "adjacent_cos": adjacent_cos, "any_cos": any_cos, "cross_identity_limit": cross_identity_limit,
            "grouping": "same-identity pairs only, joined transitively",
            "query_ratio": query_ratio, "min_query_per_identity": min_query, "seed": seed,
        },
        "leakage": {**split_leakage(distances, identities, split),
                    **burst_leakage(similarity, numbers, identities, split, adjacent_gap=adjacent_gap,
                                    adjacent_cos=adjacent_cos, any_cos=any_cos)},
        "counts": {
            "images": len(out_rows), "identities": len(per_identity_query),
            "burst_groups": len(group_sizes), "largest_group": max(group_sizes.values()),
            "groups_on_both_sides": sum(1 for sides in sides_per_group.values() if len(sides) > 1),
            "query_share_per_identity": [round(min(shares), 3), round(max(shares), 3)],
            "fewest_queries_for_an_identity": min(q for q, _ in per_identity_query.values()),
            **{side: {"images": sum(r["split"] == side for r in out_rows),
                      "identities": len({r["identity"] for r in out_rows if r["split"] == side})}
               for side in ("database", "query")},
        },
        "outputs": {"metadata": METADATA_NAME, "metadata_sha256": sha256_file(metadata_path)},
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Kaggle Jaguar dataset root")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="write masked images and the base table")
    sub.add_parser("embed", help="DINOv2-small embeddings of the masked images (CPU)")
    split = sub.add_parser("split", help="burst-aware database/query split")
    split.add_argument("--threshold", type=int, default=32, help="hash distance that always joins two photos")
    split.add_argument("--adjacent-gap", type=int, default=1, help="largest file-number gap that counts as adjacent")
    split.add_argument("--adjacent-cos", type=float, default=0.85, help="embedding cosine that joins adjacent photos")
    split.add_argument("--any-cos", type=float, default=0.95, help="embedding cosine that joins photos anywhere")
    split.add_argument("--cross-identity-limit", type=int, default=12,
                       help="fail when two jaguars have a pair at or below this hash distance")
    split.add_argument("--query-ratio", type=float, default=0.25)
    split.add_argument("--min-query", type=int, default=2)
    split.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, workers=args.workers)["counts"], indent=2))
    elif args.command == "embed":
        print(f"[jaguar] wrote {write_embeddings(args.root, args.workers)}")
    else:
        manifest = split_dataset(args.root, threshold=args.threshold, adjacent_gap=args.adjacent_gap,
                                 adjacent_cos=args.adjacent_cos, any_cos=args.any_cos,
                                 query_ratio=args.query_ratio, min_query=args.min_query, seed=args.seed,
                                 cross_identity_limit=args.cross_identity_limit)
        print(json.dumps({"counts": manifest["counts"], "leakage": manifest["leakage"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except PreparationError as error:
        sys.exit(f"[jaguar] {error}")
