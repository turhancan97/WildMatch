"""Immutable sweep submissions: frozen config tree, task table and checkpoint identities.

``create_submission`` writes ``<logs>/submissions/<id>/`` with ``probe.yaml`` and its config
groups (``paths/``, ``dataset/``), the sweep spec copy (``sweep.yaml``), the task table
(``tasks.tsv``) and ``manifest.json``. Every task reads its settings from the manifest, never
from the mutable repository config, and ``validate_task`` re-checks the snapshot and the
custom checkpoint SHA-256 and owner before the task loads a model or writes a cache.

Moved from ``scripts/probe_parallel_manifest.py`` (2026-10-04); the manifest schema is
unchanged, so submissions made by the bash launchers stay valid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from wildmatch.sweep.spec import TASK_FIELDS, task_line

SCHEMA_VERSION = 1
# Hydra config groups that probe.yaml's defaults list loads; the submission snapshot freezes
# them next to probe.yaml so tasks compose the config with --config-path from the snapshot alone.
CONFIG_GROUPS = ("paths", "dataset")


def packaged_probe_config() -> Path:
    return Path(str(resources.files("wildmatch") / "conf" / "probe.yaml"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash a file or a directory deterministically."""
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"checkpoint path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def path_owner(path: str) -> str | None:
    """The animal a WildlifeReID-10k checkpoint path belongs to, when the layout names it."""
    match = re.search(r"/wildlife-reid-10k/([^/]+)/(?:rdd|loma)-finetuned/", path)
    return match.group(1) if match else None


def _check_owner(path_text: str, owner: str, animal: str) -> None:
    owner_in_path = path_owner(path_text)
    if owner_in_path and owner_in_path != owner:
        raise ValueError(f"checkpoint path owner mismatch: declared={owner} path={owner_in_path} path={path_text}")
    if not owner_in_path and owner != animal:
        raise ValueError(
            "dataset/checkpoint owner mismatch: "
            f"dataset={animal} checkpoint_owner={owner}; "
            "cross-species checkpoints must use an owner-identifiable path"
        )


def validate_checkpoint(task: Mapping[str, str]) -> Dict[str, Any]:
    """The manifest's checkpoint record; fails closed on any inconsistency."""
    label = task["checkpoint_label"]
    components = task.get("checkpoint_components", "-")
    if label == "descriptor-fine-tuned" and components != "descriptor_only":
        raise ValueError("descriptor-fine-tuned checkpoints must use checkpoint_components=descriptor_only")
    if label == "custom" and components == "descriptor_only":
        raise ValueError("descriptor_only checkpoints must use checkpoint_label=descriptor-fine-tuned")
    # Joint (descriptor + matcher) checkpoints are complete models; loading only
    # one part would evaluate weights that were never trained together.
    if label == "joint-fine-tuned" and components != "full":
        raise ValueError("joint-fine-tuned checkpoints must use checkpoint_components=full")
    if label == "custom" and components == "full":
        raise ValueError("full-model checkpoints must use checkpoint_label=joint-fine-tuned")
    if label == "default":
        return {"source": "default", "path": None, "owner": None, "sha256": None}
    owner = task["checkpoint_owner"]
    if not owner or owner == "-":
        raise ValueError(f"custom checkpoint owner is missing for {task['profile_id']}")
    path = Path(task["checkpoint_path"])
    if not path.exists():
        raise ValueError(f"{task['matcher']} custom checkpoint does not exist: {path}")
    _check_owner(task["checkpoint_path"], owner, task["animal"])
    return {"source": "custom", "path": str(path.resolve()), "owner": owner, "sha256": sha256_path(path)}


def _config_tree_files(config_dir: Path) -> List[Path]:
    files = [config_dir / "probe.yaml"]
    for group in CONFIG_GROUPS:
        files.extend(sorted((config_dir / group).glob("*.yaml")))
    return files


def config_tree_sha256(config_dir: Path) -> str:
    """One digest over probe.yaml and every group file (relative path and content)."""
    digest = hashlib.sha256()
    for path in _config_tree_files(config_dir):
        digest.update(path.relative_to(config_dir).as_posix().encode("utf-8") + b"\0")
        digest.update(_sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _snapshot_config_groups(config_file: Path, submission_dir: Path) -> None:
    """Copy the config groups next to the snapshot: from the config's own directory when it
    has them (the packaged conf/), else from the packaged configuration (for a custom
    probe.yaml such as a parity copy)."""
    for group in CONFIG_GROUPS:
        source = config_file.parent / group
        if not source.is_dir():
            source = packaged_probe_config().parent / group
        if not source.is_dir():
            raise ValueError(f"config group '{group}' not found next to {config_file} or in the package")
        target = submission_dir / group
        target.mkdir()
        for path in sorted(source.glob("*.yaml")):
            shutil.copy2(path, target / path.name)


def _task_record(index: int, raw: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "index": index,
        "profile_id": raw["profile_id"],
        "dataset": {
            "name": raw["dataset_name"],
            "animal": raw["animal"],
            "evaluation_animal": raw["evaluation_animal"],
            "root": raw["root"],
            "metadata_file": raw["metadata_file"],
            "label_col": raw["label_col"],
            "mask_col": raw["mask_col"],
            "no_background": raw["no_background"].lower() == "true",
            "image_variant": raw["image_variant"],
            "split_col": raw["split_col"],
            "split_protocol": raw["split_col"],
            "database_split_value": raw["database_split_value"],
            "query_split_value": raw["query_split_value"],
            "calibration_size": int(raw["calibration_size"]),
        },
        "benchmark": {
            "method": raw["method"],
            "matcher": raw["matcher"],
            "candidate_k": int(raw["candidate_k"]),
            "checkpoint_label": raw["checkpoint_label"],
            "checkpoint_components": raw["checkpoint_components"],
            "loma_arch": raw["loma_arch"],
            "train_mode": raw["train_mode"],
            "class_weighting": raw["class_weighting"],
        },
        "checkpoint": validate_checkpoint(raw),
    }


def create_submission(
    tasks: Iterable[Mapping[str, str]],
    submission_dir: Path,
    submission_id: str,
    spec_path: Path,
    paths_profile: str,
    config_file: Path | None = None,
    repository_dir: Path | None = None,
) -> Path:
    """Freeze a task table into ``submission_dir``; returns the manifest path.

    Every checkpoint is validated and hashed before anything is written, so a failing
    sweep leaves no partial submission behind.
    """
    tasks = list(tasks)
    if not tasks:
        raise ValueError("task table is empty")
    for task in tasks:
        missing = set(TASK_FIELDS) - set(task)
        if missing:
            raise ValueError(f"task is missing fields: {', '.join(sorted(missing))}")
    records = [_task_record(index, task) for index, task in enumerate(tasks)]
    config_file = (config_file or packaged_probe_config()).resolve()
    if not config_file.is_file():
        raise ValueError(f"probe config does not exist: {config_file}")
    submission_dir = submission_dir.resolve()
    if submission_dir.exists():
        raise ValueError(f"submission directory already exists: {submission_dir}")
    submission_dir.mkdir(parents=True)
    config_snapshot = submission_dir / "probe.yaml"
    shutil.copy2(config_file, config_snapshot)
    _snapshot_config_groups(config_file, submission_dir)
    spec_snapshot = submission_dir / "sweep.yaml"
    shutil.copy2(spec_path, spec_snapshot)
    (submission_dir / "tasks.tsv").write_text("".join(task_line(task) + "\n" for task in tasks), encoding="utf-8")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "submission_id": submission_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_dir": str((repository_dir or Path.cwd()).resolve()),
        "sweep_spec_path": str(Path(spec_path).resolve()),
        "sweep_spec_sha256": _sha256_file(spec_snapshot),
        "paths_profile": paths_profile,
        "config_snapshot": str(config_snapshot),
        "config_sha256": _sha256_file(config_snapshot),
        "config_tree_sha256": config_tree_sha256(submission_dir),
        "task_count": len(records),
        "tasks": records,
    }
    manifest_path = submission_dir / "manifest.json"
    atomic_json(manifest_path, payload)
    return manifest_path


def load_task(manifest_path: Path, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported parallel probe manifest schema")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or index < 0 or index >= len(tasks):
        last = len(tasks) - 1 if isinstance(tasks, list) else -1
        raise ValueError(f"task index {index} is outside 0..{last}")
    return payload, tasks[index]


def validate_task(payload: Mapping[str, Any], task: Mapping[str, Any]) -> None:
    """Re-check the frozen config and the custom checkpoint before the task runs."""
    config_snapshot = Path(payload["config_snapshot"])
    if not config_snapshot.is_file() or _sha256_file(config_snapshot) != payload["config_sha256"]:
        raise ValueError("immutable probe config snapshot is missing or changed")
    expected_tree = payload.get("config_tree_sha256")  # absent in manifests written before 2026-10-04
    if expected_tree is not None and config_tree_sha256(config_snapshot.parent) != expected_tree:
        raise ValueError("immutable config groups (paths/, dataset/) in the snapshot are missing or changed")
    checkpoint = task["checkpoint"]
    if checkpoint["source"] == "custom":
        path = Path(checkpoint["path"])
        if not path.exists():
            raise ValueError(f"custom checkpoint is missing: {path}")
        if sha256_path(path) != checkpoint["sha256"]:
            raise ValueError(f"custom checkpoint content changed after submission: {path}")
        _check_owner(str(path), checkpoint["owner"], task["dataset"]["animal"])
