"""Dependency-light experiment artifact and reporting primitives."""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from wildmatch.utils.fingerprints import sha256_file
from wildmatch.utils.io import atomic_write, file_lock, write_csv_atomically

ARTIFACT_SCHEMA_VERSION = 1
RUN_INDEX_COLUMNS = [
    "run_id",
    "status",
    "workflow",
    "run_utc",
    "dataset",
    "animal",
    "split_protocol",
    "model",
    "method",
    "variant",
    "checkpoint_source",
    "checkpoint_variant",
    "checkpoint_component",
    "checkpoint_owner",
    "evaluation_animal",
    "checkpoint_component_files",
    "checkpoint_file_hashes",
    "checkpoint_protocol",
    "checkpoint_default_components",
    "checkpoint_applied_prefixes",
    "checkpoint_ignored_prefixes",
    "checkpoint_validation",
    "config_hash",
    "git_commit",
    "num_query",
    "num_database",
    "top_1",
    "top_5",
    "top_10",
    "mAP",
    "classifier_open_set_policy",
    "classification_num_query_images",
    "classification_num_seen_query_images",
    "classification_num_unseen_query_images",
    "classification_num_query_identities",
    "classification_num_seen_query_identities",
    "classification_num_unseen_query_identities",
    "classification_query_seen_coverage",
    "classification_top_1",
    "classification_top_5",
    "classification_top_10",
    "classification_balanced_top_1",
    "classification_seen_top_1",
    "classification_seen_top_5",
    "classification_seen_top_10",
    "classification_seen_balanced_top_1",
    "classification_open_top_1",
    "classification_open_top_5",
    "classification_open_top_10",
    "classification_open_balanced_top_1",
    "classification_embedding_retrieval_enabled",
    "primary_compute_runtime_sec",
    "matcher_runtime_sec",
    "method_compute_runtime_sec",
    "total_runtime_sec",
    "feature_extraction_sec",
    "feature_extraction_compute_sec",
    "feature_cache_lookup_sec",
    "model_setup_sec",
    "calibration_sec",
    "matching_sec",
    "run_dir",
    "manifest_path",
    "metrics_path",
    "visualization_dir",
]


def safe_component(value: Any, default: str = "unknown") -> str:
    """Convert a config value into a readable, filesystem-safe path component."""

    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or default


def _resolved_container(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, Mapping)):
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return cfg


def configuration_hash(cfg: Any) -> str:
    """Return a stable SHA-256 hash for the resolved configuration."""

    payload = json.dumps(
        _resolved_container(cfg),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.select(cfg, key, default=default)
        return default if value is None and default is not None else value
    current = cfg
    for part in key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            current = getattr(current, part, default)
    return default if current is None and default is not None else current


# Paths whose uncommitted changes can change a run: the package (code and configs) and the
# locked environment. Edits elsewhere (docs, notes, AGENTS.md) do not mark a run dirty.
CODE_PATHS = ("src", "pyproject.toml", "uv.lock")
CODE_DIFF_NAME = "code.diff"


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), *args], stderr=subprocess.DEVNULL, text=True)


def _code_identity(package_dir: Path) -> Dict[str, Any]:
    """Commit and uncommitted changes of the git checkout that ``package_dir`` belongs to.

    Uses the checkout the code was imported from, not the working directory, so a run started
    from one checkout with another's code on PYTHONPATH records the code it actually ran. The
    ``diff`` entry (text of tracked changes plus untracked files) is written to the run folder
    as ``code.diff`` and left out of the manifest.
    """
    try:
        checkout = Path(_git(package_dir, "rev-parse", "--show-toplevel").strip())
        commit = _git(checkout, "rev-parse", "HEAD").strip()
    except Exception:
        return {"commit": "unknown", "dirty": None, "checkout": None, "changed_paths": [], "diff_sha256": None}
    status = _git(checkout, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    changed = sorted(line[3:] for line in status.splitlines() if line.strip())
    diff = _git(checkout, "diff", "HEAD", "--binary", "--", *CODE_PATHS)
    for line in status.splitlines():
        if line.startswith("?? "):
            path = checkout / line[3:]
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                content = f"<unreadable or binary; sha256 {sha256_file(path) if path.is_file() else 'missing'}>\n"
            diff += f"--- untracked: {line[3:]}\n{content}"
    return {
        "commit": commit,
        "dirty": bool(changed),
        "checkout": checkout.as_posix(),
        "changed_paths": changed,
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if changed else None,
        "diff": diff if changed else "",
    }


@functools.lru_cache(maxsize=None)
def code_identity() -> Dict[str, Any]:
    """The code identity of this process (computed once, when the first run records it)."""
    return _code_identity(Path(__file__).resolve().parents[1])


def _git_commit() -> str:
    return str(code_identity()["commit"])


def file_identity(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Return a compact identity for a generated or checkpoint file."""

    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return {"path": candidate.as_posix(), "exists": False}
    stat = candidate.stat()
    return {
        "path": candidate.as_posix(),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(candidate),
    }


def environment_info() -> Dict[str, Any]:
    """Collect lightweight environment details without requiring ML packages."""

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore

        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        info["torch"] = None
        info["cuda_available"] = None
        info["cuda_version"] = None
    return info


@dataclass
class RunContext:
    """Paths and identity shared by one probe or finetune run."""

    workflow: str
    run_id: str
    run_utc: str
    config_hash: str
    run_dir: Path
    config_snapshot_path: Path
    manifest_path: Path
    metrics_path: Path
    timings_path: Path
    visualization_dir: Path

    def relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.run_dir).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def write_config(self, cfg: Any) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(cfg, (DictConfig, Mapping)):
            OmegaConf.save(cfg, self.config_snapshot_path, resolve=True)
        else:
            self.config_snapshot_path.write_text(
                json.dumps(cfg, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )

    def write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        # Atomic: a full disk on 2026-10-04 left one run with an empty run_manifest.json.
        text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        atomic_write(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))

    def write_metrics(self, metrics: Mapping[str, Any]) -> None:
        self.write_json(self.metrics_path, dict(metrics))

    def write_timings(self, timings: Mapping[str, Any]) -> None:
        self.write_json(self.timings_path, dict(timings))

    def write_manifest(self, payload: Mapping[str, Any], status: str = "running") -> Dict[str, Any]:
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": status,
            "run_id": self.run_id,
            "run_utc": self.run_utc,
            "workflow": self.workflow,
            "config_hash": self.config_hash,
            "git_commit": _git_commit(),
            "code": {key: value for key, value in code_identity().items() if key != "diff"},
            "environment": environment_info(),
            "artifacts": {
                "config": self.relative(self.config_snapshot_path),
                "manifest": self.relative(self.manifest_path),
                "metrics": self.relative(self.metrics_path),
                "timings": self.relative(self.timings_path),
                "visualizations": self.relative(self.visualization_dir),
            },
        }
        manifest.update(dict(payload))
        self.write_json(self.manifest_path, manifest)
        identity = code_identity()
        diff_path = self.run_dir / CODE_DIFF_NAME
        if identity.get("dirty") and not diff_path.exists():
            diff_path.write_text(str(identity["diff"]), encoding="utf-8")
        return manifest


def build_run_context(
    cfg: Any,
    workflow: str,
    *,
    run_started: Optional[datetime] = None,
) -> RunContext:
    """Build a readable, hash-addressed run directory for a resolved config."""

    started = run_started or datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    started = started.astimezone(timezone.utc)
    run_utc = started.isoformat().replace("+00:00", "Z")
    config_hash = configuration_hash(cfg)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{config_hash[:8]}"

    root = Path(str(_select(cfg, "output.experiment_root", "experiments")))
    dataset = safe_component(_select(cfg, "dataset.name", "dataset"))
    animal = safe_component(_select(cfg, "dataset.animal", "animal"))
    split = safe_component(_select(cfg, "dataset.split_col", "split"))
    model = safe_component(_select(cfg, "model.type", "model"))
    workflow_name = safe_component(workflow)

    if workflow == "probe":
        method = safe_component(_select(cfg, "benchmark.method", "method"))
        variant = "default"
        if method == "vismatch":
            variant = safe_component(_select(cfg, "benchmark.methods.vismatch.matcher", "default"))
        relative_dir = Path(workflow_name, dataset, animal, split, model, method, variant, run_id)
    elif workflow == "finetune":
        relative_dir = Path(workflow_name, dataset, animal, split, model, run_id)
    else:
        raise ValueError(f"Unsupported reporting workflow: {workflow}")

    run_dir = root / relative_dir
    collision = 1
    while run_dir.exists():
        run_id = f"{timestamp}_{config_hash[:8]}_{collision:02d}"
        run_dir = root / relative_dir.parent / run_id
        collision += 1
    visualization_dir = run_dir / "visualizations"
    return RunContext(
        workflow=workflow_name,
        run_id=run_id,
        run_utc=run_utc,
        config_hash=config_hash,
        run_dir=run_dir,
        config_snapshot_path=run_dir / "config.snapshot.yaml",
        manifest_path=run_dir / "run_manifest.json",
        metrics_path=run_dir / "metrics.json",
        timings_path=run_dir / "timings.json",
        visualization_dir=visualization_dir,
    )


def upsert_run_index(index_path: Path, row: Mapping[str, Any]) -> None:
    """Insert or replace one run in the central CSV index (locked: parallel tasks share it)."""

    with file_lock(index_path):
        _upsert_run_index_locked(index_path, row)


def _upsert_run_index_locked(index_path: Path, row: Mapping[str, Any]) -> None:
    rows: list[Dict[str, Any]] = []
    fieldnames = list(RUN_INDEX_COLUMNS)
    if index_path.is_file():
        with index_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or fieldnames)
            rows = [dict(existing) for existing in reader]
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    new_row = {key: row.get(key, "") for key in fieldnames}
    replaced = False
    for index, existing in enumerate(rows):
        if str(existing.get("run_id", "")) == str(row.get("run_id", "")):
            rows[index] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    write_csv_atomically(index_path, fieldnames, rows)


def run_index_row(context: RunContext, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a central-index row with stable artifact links."""

    row = {
        "run_id": context.run_id,
        "status": payload.get("status", "completed"),
        "workflow": context.workflow,
        "run_utc": context.run_utc,
        "config_hash": context.config_hash,
        "run_dir": context.run_dir.as_posix(),
        "manifest_path": context.manifest_path.as_posix(),
        "metrics_path": context.metrics_path.as_posix(),
        "visualization_dir": context.visualization_dir.as_posix(),
        # The column existed since the index was created but was never filled (found 2026-10-06).
        "git_commit": _git_commit(),
    }
    row.update({key: value for key, value in payload.items() if key not in row})
    return row
