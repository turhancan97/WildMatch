"""Provenance of training and mining launches: the code, the inputs and the environment that ran.

Evaluation runs record their code identity in `run_manifest.json` (`wildmatch.reporting.artifacts`).
Matcher fine-tuning and pair mining write their outputs into folders the trainers and miners own, so
the launchers record the same identity in a separate file next to those outputs instead of changing
the formats the checkpoint loader and the trainers read.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PACKAGES = ("wildmatch", "torch", "torchvision", "accelerate", "lomatch", "vismatch", "numpy", "opencv-python")


def _version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Path, existence and SHA-256 of an input file (None when no path was given)."""
    if path is None:
        return None
    from wildmatch.utils.fingerprints import sha256_file

    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "exists": False}
    return {"path": str(p), "exists": True, "size": p.stat().st_size, "sha256": sha256_file(p)}


def launch_record(command: Sequence[str], inputs: Mapping[str, Optional[Path]], **extra: Any) -> dict[str, Any]:
    """One launch: when, where, which code (commit, uncommitted changes), command, inputs, packages."""
    from wildmatch.reporting.artifacts import code_identity

    code = dict(code_identity())
    code.pop("diff", None)
    return {
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "code": code,
        "command": list(command),
        "inputs": {name: file_sha256(path) for name, path in inputs.items()},
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in PACKAGES},
        **extra,
    }


def append_launch(path: Path, record: Mapping[str, Any]) -> Path:
    """Append a launch to `path` (a JSON object with a `launches` list); save the code diff if dirty.

    Resumed runs append rather than overwrite, so the file keeps every launch that wrote into the
    folder. A dirty checkout's diff goes to `code-<n>.diff` next to the file (n = launch number).
    """
    from wildmatch.reporting.artifacts import code_identity

    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"launches": []}
    record = dict(record)
    diff = code_identity().get("diff") or ""
    if diff:
        diff_path = path.parent / f"code-{len(data['launches'])}.diff"
        diff_path.write_text(diff, encoding="utf-8")
        record["code_diff_file"] = diff_path.name
    data["launches"].append(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
