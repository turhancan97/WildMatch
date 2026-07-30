from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from tqdm import tqdm

class TripletImageFolder(Dataset):
    """ImageFolder wrapper that returns (anchor, positive, negative) triplets.

    Positive is sampled randomly from the same class as the anchor,
    negative from a randomly chosen different class.

    Args:
        root: Path to the image folder root (class subdirectories expected).
        transform: Transform applied to every image independently.
        anchor_transform: Optional separate transform for the anchor image.
    """

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
        anchor_transform: transforms.Compose | None = None,
    ) -> None:
        self._base = ImageFolder(root=str(root), transform=None)
        self.transform = transform
        self.anchor_transform = anchor_transform or transform

        # Build per-class index lists for fast sampling
        self._class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(self._base.samples):
            self._class_to_indices[label].append(idx)

        self.classes = self._base.classes
        self.class_to_idx = self._base.class_to_idx

    def __len__(self) -> int:
        return len(self._base)

    def _load(self, idx: int) -> torch.Tensor:
        path, _ = self._base.samples[idx]
        img = self._base.loader(path)
        return img

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_label = self._base.targets[index]

        # Positive: different index, same class
        pos_pool = self._class_to_indices[anchor_label]
        pos_idx = index
        while pos_idx == index and len(pos_pool) > 1:
            pos_idx = random.choice(pos_pool)

        # Negative: random different class
        neg_label = anchor_label
        while neg_label == anchor_label:
            neg_label = random.choice(list(self._class_to_indices.keys()))
        neg_idx = random.choice(self._class_to_indices[neg_label])

        anchor_img = self._load(index)
        pos_img = self._load(pos_idx)
        neg_img = self._load(neg_idx)

        if self.anchor_transform is not None:
            anchor_img = self.anchor_transform(anchor_img)
        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return anchor_img, pos_img, neg_img


class FixedTripletDataset(Dataset):
    """Deterministic triplet dataset: triplets are pre-sampled once at construction.

    Uses a private `random.Random` instance so it never pollutes global RNG state
    and always produces the same triplets regardless of training state.
    """

    def __init__(self, base: TripletImageFolder, n_samples: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        self._base = base

        anchor_indices = rng.sample(range(len(base)), min(n_samples, len(base)))

        self._triplets: list[tuple[int, int, int]] = []
        for anchor_idx in anchor_indices:
            anchor_label = base._base.targets[anchor_idx]

            pos_pool = base._class_to_indices[anchor_label]
            pos_idx = anchor_idx
            while pos_idx == anchor_idx and len(pos_pool) > 1:
                pos_idx = rng.choice(pos_pool)

            neg_label = anchor_label
            while neg_label == anchor_label:
                neg_label = rng.choice(list(base._class_to_indices.keys()))
            neg_idx = rng.choice(base._class_to_indices[neg_label])

            self._triplets.append((anchor_idx, pos_idx, neg_idx))

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx, pos_idx, neg_idx = self._triplets[index]
        base = self._base

        anchor_img = base._load(anchor_idx)
        pos_img    = base._load(pos_idx)
        neg_img    = base._load(neg_idx)

        if base.anchor_transform is not None:
            anchor_img = base.anchor_transform(anchor_img)
        if base.transform is not None:
            pos_img = base.transform(pos_img)
            neg_img = base.transform(neg_img)

        return anchor_img, pos_img, neg_img


class IndexAssignedTripletDataset(Dataset):
    """Triplet dataset driven by a pre-built JSON index.

    Each entry in the index describes one query frame together with a pool of
    positives and a pool of negatives.  At every call to ``__getitem__`` one
    positive and one negative are drawn at random from their respective pools.

    Expected JSON schema (list of objects)::

        [
          {
            "query_frame": "relative/path/to/query.jpg",
            "positives": ["rel/pos1.jpg", ...],
            "negatives": ["rel/neg1.jpg", ...]
          },
          ...
        ]

    Args:
        index_path: Path to the JSON index file.
        root: Root directory that is prepended to every relative path in the
            index.  If *None*, paths are used as-is (must then be absolute).
        transform: Transform applied to all three images (query, positive,
            negative).
        query_transform: Optional separate transform for the query image; falls
            back to *transform* when not provided.
        loader: Callable that loads a PIL image from a path.
        random_negative_prob: Probability of replacing the index-mined
            negative with a random image of a *different* lynx, drawn from
            the whole candidate pool (every lynx dir under the same split the
            index's own positives/negatives come from) instead of just this
            entry's top_m negatives. 0 (default) keeps the original
            index-only behavior. Requires `root` to be set, since the
            candidate pool is built by scanning the filesystem. Mutable at
            runtime (read fresh on every `__getitem__` call) so a training
            loop can move it over time — see train_by_lg_matches.py's
            `--moving_negative_prob`.
        negative_mining: When drawing a random negative (see
            random_negative_prob), bias which lynx it's drawn from toward
            candidates that have produced high LG confidence for that query
            lynx in the past, instead of picking uniformly. Weights come from
            an EMA difficulty matrix over (query_lynx, candidate_lynx) pairs
            that starts uniform (all-zero scores) and is only updated via
            `update_mining_stats` — nothing here is `torch.no_grad`-implicit,
            the dataset does no scoring itself. Requires random_negative_prob
            > 0.
        negative_mining_temperature: Softmax temperature over the difficulty
            matrix row when negative_mining is on; lower concentrates
            sampling on the single hardest known candidate lynx.
        negative_mining_decay: EMA decay used by `update_mining_stats` (closer
            to 1 remembers older observations longer).
        weak_queries: With probability weak_queries_prob, replace the
            index-driven (query, positive, negative) for an item with a fully
            random triplet: a random lynx, two distinct random frames of it
            (query + positive), and a random frame of a different random lynx
            (negative) — there's no index entry to mine a positive/negative
            pool from for a random query, so both are plain uniform draws
            from the candidate pool (independent of negative_mining, which
            only biases the *index*-query random-negative branch). Requires
            `root` to be set (same candidate pool as random_negative_prob).
        weak_queries_prob: Probability of drawing a weak_queries triplet
            instead of the index-driven one, per `__getitem__` call.
        return_meta: When True, `__getitem__` returns a 4th element: a dict
            with `neg_source` ("index" | "random"), `query_lynx`, `neg_lynx`,
            `is_weak_query`, and `query_frame` (the actual query path drawn —
            entry["query_frame"] for an index sample, the random one for a
            weak_queries sample), letting a training loop bucket LG confidence
            by where each sample came from (used by --moving_negative_prob,
            --negative_mining, --weak_queries) or track a specific query's
            outcome across epochs (used by train_by_lg_matches.py's
            pos-index dead-candidate recurrence tracking). Default False
            keeps the original 3-tuple for all other callers.
    """

    def __init__(
        self,
        index_path: str | Path,
        root: str | Path | None = None,
        transform: transforms.Compose | None = None,
        query_transform: transforms.Compose | None = None,
        loader=None,
        random_negative_prob: float = 0.0,
        negative_mining: bool = False,
        negative_mining_temperature: float = 1.0,
        negative_mining_decay: float = 0.9,
        weak_queries: bool = False,
        weak_queries_prob: float = 0.0,
        return_meta: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.query_transform = query_transform or transform
        self._loader = loader or default_loader
        self.random_negative_prob = random_negative_prob
        self.negative_mining = negative_mining
        self.negative_mining_temperature = negative_mining_temperature
        self.negative_mining_decay = negative_mining_decay
        self.weak_queries = weak_queries
        self.weak_queries_prob = weak_queries_prob
        self.return_meta = return_meta
        self._mining_scores: dict[tuple[str, str], float] = {}

        with open(index_path) as f:
            self._entries: list[dict] = json.load(f)

        for entry in self._entries:
            if not entry.get("positives"):
                raise ValueError(f"Entry for {entry['query_frame']} has no positives.")
            if not entry.get("negatives"):
                raise ValueError(f"Entry for {entry['query_frame']} has no negatives.")

        self._lynx_pool: dict[str, list[str]] = {}
        self._lynx_ids: list[str] = []
        if random_negative_prob > 0 or weak_queries:
            if self.root is None:
                raise ValueError(
                    "random_negative_prob > 0 or weak_queries=True requires `root` to be set"
                )
            self._lynx_pool = self._scan_lynx_pool()
            self._lynx_ids = list(self._lynx_pool)

    def _scan_lynx_pool(self) -> dict[str, list[str]]:
        """Every image under the split that positives/negatives are drawn
        from (e.g. 'train/'), grouped by lynx id — the same split used by
        every entry's own positives/negatives, whatever the query's split.
        """
        cand_split = Path(self._entries[0]["positives"][0]).parts[0]
        cand_root = self.root / cand_split
        pool: dict[str, list[str]] = defaultdict(list)
        for lynx_dir in sorted(cand_root.iterdir()):
            if not lynx_dir.is_dir():
                continue
            for img_path in lynx_dir.rglob("*.jpg"):
                pool[lynx_dir.name].append(str(img_path.relative_to(self.root)))
        return dict(pool)

    def _lynx_id(self, rel_path: str) -> str:
        return Path(rel_path).parts[1]

    def __len__(self) -> int:
        return len(self._entries)

    def _full_path(self, rel: str) -> Path:
        return self.root / rel if self.root is not None else Path(rel)

    def _sample_negative_lynx(self, query_lynx: str) -> str:
        """Picks a candidate lynx != query_lynx for the 'random negative' branch.

        Uniform unless negative_mining is on, in which case candidates are
        weighted by softmax(EMA difficulty score / temperature) — untried
        pairs default to score 0, so sampling starts uniform and only drifts
        toward historically-hard candidates as `update_mining_stats` feeds it
        real observations.
        """
        candidates = [l for l in self._lynx_ids if l != query_lynx]
        if not self.negative_mining:
            return random.choice(candidates)

        scores = [self._mining_scores.get((query_lynx, l), 0.0) for l in candidates]
        m = max(scores)
        exps = [math.exp((s - m) / self.negative_mining_temperature) for s in scores]
        total = sum(exps)
        weights = [e / total for e in exps]
        return random.choices(candidates, weights=weights, k=1)[0]

    def update_mining_stats(self, observations: list[tuple[str, str, float]]) -> None:
        """EMA-updates the (query_lynx, candidate_lynx) difficulty matrix used by
        negative_mining. `observations` is a list of (query_lynx, neg_lynx,
        lg_confidence) triples — typically an epoch's negatives that came from
        the 'random' branch (see `neg_source` in __getitem__'s meta dict) of an
        *index* query (weak_queries samples are excluded by the training loop,
        since they don't share the index-query random-negative distribution),
        collected and passed in once per epoch by the training loop.
        """
        for query_lynx, neg_lynx, conf in observations:
            key = (query_lynx, neg_lynx)
            prev = self._mining_scores.get(key)
            self._mining_scores[key] = (
                conf if prev is None
                else self.negative_mining_decay * prev + (1 - self.negative_mining_decay) * conf
            )

    def _sample_negative(self, entry: dict) -> tuple[str, dict]:
        query_lynx = self._lynx_id(entry["query_frame"])
        if self.random_negative_prob > 0 and random.random() < self.random_negative_prob:
            neg_lynx = self._sample_negative_lynx(query_lynx)
            neg_path = random.choice(self._lynx_pool[neg_lynx])
            return neg_path, {"neg_source": "random", "query_lynx": query_lynx, "neg_lynx": neg_lynx}
        neg_path = random.choice(entry["negatives"])
        return neg_path, {"neg_source": "index", "query_lynx": query_lynx, "neg_lynx": self._lynx_id(neg_path)}

    def _sample_weak_triplet(self) -> tuple[str, str, str, dict]:
        """Fully random (query, positive, negative), bypassing the JSON index
        entirely: a random lynx supplies two distinct frames (query +
        positive), a different random lynx supplies the negative frame.
        Deliberately plain `random.choice` throughout, NOT negative_mining-
        weighted — weak queries are meant as an unbiased exploration signal,
        independent of the mining curriculum (which only targets the
        index-query random-negative branch via _sample_negative_lynx).
        Requires weak_queries=True, which guarantees _lynx_pool is built.
        """
        query_lynx = random.choice(self._lynx_ids)
        pool = self._lynx_pool[query_lynx]
        query_rel = random.choice(pool)
        pos_rel = query_rel
        if len(pool) > 1:
            while pos_rel == query_rel:
                pos_rel = random.choice(pool)

        neg_lynx = query_lynx
        while neg_lynx == query_lynx:
            neg_lynx = random.choice(self._lynx_ids)
        neg_rel = random.choice(self._lynx_pool[neg_lynx])

        meta = {
            "neg_source": "random", "query_lynx": query_lynx, "neg_lynx": neg_lynx,
            "is_weak_query": True, "query_frame": query_rel,
        }
        return query_rel, pos_rel, neg_rel, meta

    def __getitem__(self, index: int):
        if self.weak_queries and random.random() < self.weak_queries_prob:
            query_rel, pos_rel, neg_rel, meta = self._sample_weak_triplet()
        else:
            entry = self._entries[index]
            query_rel = entry["query_frame"]
            pos_rel = random.choice(entry["positives"])
            neg_rel, meta = self._sample_negative(entry)
            meta["is_weak_query"] = False
            meta["query_frame"] = query_rel

        query_path = self._full_path(query_rel)
        pos_path   = self._full_path(pos_rel)
        neg_path   = self._full_path(neg_rel)

        query_img = self._loader(query_path)
        pos_img   = self._loader(pos_path)
        neg_img   = self._loader(neg_path)

        if self.query_transform is not None:
            query_img = self.query_transform(query_img)
        if self.transform is not None:
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        if self.return_meta:
            return query_img, pos_img, neg_img, meta
        return query_img, pos_img, neg_img


class PseudoAccuracyDataset(Dataset):
    """Query + its FULL candidate pool, for pseudo-accuracy eval.

    Unlike IndexAssignedTripletDataset (one random positive + one random
    negative per item), pseudo-accuracy needs every positive and every
    negative for a query to find the best-scoring one. Every entry must carry
    the same number of positives and the same number of negatives (true for
    indices built with a fixed top_k/top_m) so candidates stack into a
    uniform (n_pos + n_neg, C, H, W) tensor per item and batch across queries
    via the default collate_fn — same DataLoader/num_workers path as
    training, instead of loading images one at a time in the main process.

    Args:
        entries: List of index entries (query_frame/positives/negatives dicts).
        root: Root directory prepended to every relative path in the index.
        transform: Transform applied to every candidate image.
        query_transform: Optional separate transform for the query image;
            falls back to *transform* when not provided.
        loader: Callable that loads a PIL image from a path.
    """

    def __init__(
        self,
        entries: list[dict],
        root: str | Path | None = None,
        transform: transforms.Compose | None = None,
        query_transform: transforms.Compose | None = None,
        loader=None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.transform = transform
        self.query_transform = query_transform or transform
        self._loader = loader or default_loader
        self.entries = entries

        n_pos = {len(e["positives"]) for e in entries}
        n_neg = {len(e["negatives"]) for e in entries}
        if len(n_pos) > 1 or len(n_neg) > 1:
            raise ValueError(
                "PseudoAccuracyDataset requires every entry to have the same "
                f"number of positives/negatives; got positive counts {n_pos} "
                f"and negative counts {n_neg}"
            )
        self.n_pos = next(iter(n_pos)) if n_pos else 0
        self.n_neg = next(iter(n_neg)) if n_neg else 0

    def __len__(self) -> int:
        return len(self.entries)

    def _full_path(self, rel: str) -> Path:
        return self.root / rel if self.root is not None else Path(rel)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.entries[index]

        query_img = self._loader(self._full_path(entry["query_frame"]))
        if self.query_transform is not None:
            query_img = self.query_transform(query_img)

        cand_paths = list(entry["positives"]) + list(entry["negatives"])
        cand_imgs = [self._loader(self._full_path(p)) for p in cand_paths]
        if self.transform is not None:
            cand_imgs = [self.transform(img) for img in cand_imgs]

        return query_img, torch.stack(cand_imgs), index


class LabeledImageFolder(ImageFolder):
    """Plain ImageFolder that returns (image, label) pairs.

    Suitable for online triplet/contrastive mining where the loss function
    constructs pairs/triplets from a regular labeled batch.
    """

    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__(root=str(root), transform=transform)


class BalancedBatchSampler(Sampler):
    """Yields batches with exactly `n_classes` classes and `n_samples` per class.

    Commonly used with online triplet/contrastive loss to guarantee that every
    batch contains enough positives to mine from.

    Args:
        labels: Sequence of integer class labels, one per dataset item.
        n_classes: Number of distinct classes per batch.
        n_samples: Number of samples per class per batch.
    """

    def __init__(
        self,
        labels: list[int],
        n_classes: int,
        n_samples: int,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples

        self._class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self._class_to_indices[label].append(idx)

        self._classes = list(self._class_to_indices.keys())
        if len(self._classes) < n_classes:
            raise ValueError(
                f"Dataset has only {len(self._classes)} classes, "
                f"but n_classes={n_classes} was requested."
            )

        # Number of batches: how many times we can cycle through all classes
        self._n_batches = len(labels) // self.batch_size

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            chosen_classes = random.sample(self._classes, self.n_classes)
            batch: list[int] = []
            for cls in chosen_classes:
                pool = self._class_to_indices[cls]
                # Sample with replacement if the class has fewer than n_samples images
                batch.extend(
                    random.choices(pool, k=self.n_samples)
                    if len(pool) < self.n_samples
                    else random.sample(pool, self.n_samples)
                )
            yield batch


def get_loader(
    data: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 16,
    pin: bool = True,
    persistent_workers=True,
    seed: int | None = None,
):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    return loader
