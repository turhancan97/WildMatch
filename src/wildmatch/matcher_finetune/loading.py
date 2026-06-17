from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

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
