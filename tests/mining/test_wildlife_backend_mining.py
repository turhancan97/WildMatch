from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wildmatch.mining.wildlife_aggregate import main as aggregate_main
from wildmatch.mining.wildlife_mine import validate_loma_cache


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_feature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=np.zeros((512, 2), dtype=np.float32),
        descriptors=np.zeros((512, 256), dtype=np.float32),
        scores=np.ones(512, dtype=np.float32),
        image_size=np.array([512, 512], dtype=np.int32),
    )


def test_loma_cache_metadata_and_checkpoint_are_validated(tmp_path: Path):
    weights = tmp_path / "loma_B.pt"
    weights.write_bytes(b"checkpoint")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {
                "backend": "loma",
                "variant": "loma-b",
                "resize": 512,
                "num_keypoints": 512,
                "patch_size": 14,
                "weights_sha256": _hash(weights),
            }
        )
    )
    root = tmp_path / "view"
    frame = root / "train" / "id" / "collection" / "frame_000000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"image")
    _write_feature(cache / "train" / "id" / "collection" / "frame_000000.npz")

    manifest = validate_loma_cache(cache, "loma-b", weights)
    assert manifest["backend"] == "loma"

    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights_sha256"):
        validate_loma_cache(cache, "loma-b", weights)


def test_aggregate_writes_backend_metadata_without_reading_sidecars(tmp_path: Path, monkeypatch):
    prefix = tmp_path / "indices" / "strong-matches"
    prefix.parent.mkdir()
    root = tmp_path / "view"
    report = {
        "backend": "loma",
        "selected_frames": [
            {
                "query_frame": "train/id/collection/frame_000000.jpg",
                "positives": [{"frame": "train/id/collection/frame_000001.jpg"}],
                "negatives": [{"frame": "train/other/collection/frame_000000.jpg"}],
            }
        ],
    }
    (Path(f"{prefix}_train_0.json")).write_text(json.dumps(report))
    (Path(f"{prefix}_train_0.metadata.json")).write_text(json.dumps({"backend": "loma"}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "wildlife_aggregate",
            "--dump_report",
            str(prefix),
            "--dataset_root",
            str(root),
            "--dataset_id",
            "BelugaID",
            "--protocol",
            "strict",
            "--backend",
            "loma",
            "--weights",
            "/tmp/loma_B.pt",
            "--cache_dir",
            "/tmp/loma-cache",
            "--splits",
            "train",
        ],
    )
    aggregate_main()
    output = json.loads(Path(f"{prefix}_train_combined.json").read_text())
    metadata = json.loads(Path(f"{prefix}_train_combined.metadata.json").read_text())
    assert len(output) == 1
    assert metadata["backend"] == "loma"
    assert metadata["dataset"] == "BelugaID"
