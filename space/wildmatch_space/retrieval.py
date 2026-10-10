"""Find an individual in a bundled gallery: MegaDescriptor-L candidate list, then feature matching.

Gallery folders (``space/gallery/<name>/``) are written by ``paper/tools/build_space_gallery.py``: ``metadata.csv``
(``identity``, ``raw``, ``masked``, ``source``, ``licence``), the photos, and ``embeddings.npz`` (L2-normalized
MegaDescriptor-L embeddings of the masked photos, rows in metadata order). The query is embedded the same way,
the ``k`` most similar gallery photos form the candidate list (descending cosine, lowest index on ties, the rule
of every ranking in the repository), and the matcher scores each candidate; candidates are then ordered by
matcher score with the same tie rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

GALLERY_ROOT = Path(__file__).resolve().parents[1] / "gallery"
EMBED_SIZE = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_embedder = None


@dataclass(frozen=True)
class Gallery:
    name: str
    root: Path
    metadata: pd.DataFrame
    embeddings: np.ndarray

    def raw(self, index: int) -> Image.Image:
        return Image.open(self.root / self.metadata.iloc[index]["raw"]).convert("RGB")

    def masked(self, index: int) -> Image.Image:
        return Image.open(self.root / self.metadata.iloc[index]["masked"]).convert("RGB")

    def foreground(self, index: int) -> np.ndarray:
        return np.asarray(self.masked(index)).max(axis=2) > 0


def available() -> List[str]:
    return (
        sorted(p.name for p in GALLERY_ROOT.iterdir() if (p / "metadata.csv").is_file())
        if GALLERY_ROOT.is_dir()
        else []
    )


@lru_cache(maxsize=None)
def load(name: str) -> Gallery:
    root = GALLERY_ROOT / name
    metadata = pd.read_csv(root / "metadata.csv")
    embeddings = np.load(root / "embeddings.npz")["embeddings"].astype(np.float32)
    if len(embeddings) != len(metadata):
        raise ValueError(f"gallery {name}: {len(embeddings)} embeddings for {len(metadata)} photos")
    return Gallery(name, root, metadata, embeddings)


def embedder():
    """MegaDescriptor-L (``BVRA/MegaDescriptor-L-384``) as ``wildmatch.models.model.get_model`` builds it."""
    global _embedder
    if _embedder is None:
        import torch

        from wildmatch.models.model import get_model

        backbone = get_model("megadescriptor-l")[0]
        _embedder = backbone.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return _embedder


def embed(images: List[Image.Image]) -> np.ndarray:
    """L2-normalized embeddings with the probe's evaluation transform (square resize, ImageNet normalization)."""
    import torch
    import torchvision.transforms as T

    model = embedder()
    device = next(model.parameters()).device
    transform = T.Compose([T.Resize([EMBED_SIZE, EMBED_SIZE]), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    with torch.inference_mode():
        batch = torch.stack([transform(image.convert("RGB")) for image in images]).to(device)
        out = model(batch).float()
    return torch.nn.functional.normalize(out, dim=1).cpu().numpy()


def candidates(query_embedding: np.ndarray, gallery: Gallery, k: int) -> np.ndarray:
    """Indices of the ``k`` gallery photos most similar to the query (descending cosine, lowest index on ties)."""
    from wildmatch.evaluate.ranking import stable_rank_1d

    similarity = gallery.embeddings @ np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    return stable_rank_1d(similarity.astype(np.float64))[: min(k, len(similarity))]


def order_by_score(indices: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """``indices`` reordered by descending matcher score; ties keep the lower gallery index first."""
    from wildmatch.evaluate.ranking import stable_rank_1d

    indices, scores = np.asarray(indices), np.asarray(scores, dtype=np.float64)
    by_index = np.argsort(indices, kind="stable")
    return indices[by_index][stable_rank_1d(scores[by_index])]
