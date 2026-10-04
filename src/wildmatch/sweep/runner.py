"""Run sweeps: list, freeze, submit to Slurm, or run tasks locally.

``wildmatch sweep <spec> --list-tasks | --dry-run | --submit | --local`` builds the task table
(:mod:`.spec`); every mode except ``--list-tasks`` freezes it into an immutable submission
(:mod:`.manifest`). ``--submit`` hands the submission to ``sbatch`` as an array job whose
elements call ``wildmatch sweep-task --manifest <m> --index <i>``; ``--local`` runs the
same per-task code one task after another on this machine.

A task re-validates its frozen config and checkpoint, writes its log record, runs the
evaluation (``python -m wildmatch evaluate`` with Hydra overrides read from the manifest)
while mirroring its output into the task's ``.out``/``.err``/``.combined.log``, and rebuilds
``logs/index.csv``. A validation failure records the reason, rebuilds the index and, inside
Slurm, cancels only its own array element.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from wildmatch.paths import active_profile
from wildmatch.sweep import logs
from wildmatch.sweep.manifest import create_submission, load_task, validate_task
from wildmatch.sweep.spec import SweepError, build_tasks, describe_task, load_spec, resolve_spec_path

DEFAULT_LOGS_ROOT = Path("logs/parallel_run")
DEFAULT_SBATCH_SCRIPT = Path("slurm/sweep_task.sbatch")
PROBE_METHODS = ("linear_probe", "efficient_probe")
# Manifests written on the refactor branch by the bash launchers carry the config groups but
# no paths_profile; those launchers always passed paths=gmum.
LAUNCHER_PATHS_PROFILE = "gmum"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value or "unknown")) or "unknown"
    return text[:96]


def task_from_record(record: Mapping[str, Any]) -> Dict[str, str]:
    """The flat task fields of one manifest task record."""
    dataset, benchmark, checkpoint = record["dataset"], record["benchmark"], record["checkpoint"]
    return {
        "profile_id": record["profile_id"],
        "dataset_name": dataset["name"],
        "animal": dataset["animal"],
        "evaluation_animal": dataset.get("evaluation_animal", dataset["animal"]),
        "root": dataset["root"],
        "metadata_file": dataset["metadata_file"],
        "label_col": dataset["label_col"],
        "mask_col": dataset["mask_col"],
        "no_background": str(dataset["no_background"]).lower(),
        "image_variant": dataset["image_variant"],
        "split_col": dataset["split_col"],
        "database_split_value": dataset["database_split_value"],
        "query_split_value": dataset["query_split_value"],
        "calibration_size": str(dataset["calibration_size"]),
        "method": benchmark["method"],
        "matcher": benchmark["matcher"],
        "candidate_k": str(benchmark["candidate_k"]),
        "checkpoint_label": benchmark["checkpoint_label"],
        "checkpoint_components": benchmark["checkpoint_components"],
        "loma_arch": benchmark["loma_arch"],
        "train_mode": benchmark["train_mode"],
        "class_weighting": benchmark.get("class_weighting", "-"),
        "checkpoint_source": checkpoint["source"],
        "checkpoint_path": checkpoint["path"] or "-",
        "checkpoint_owner": checkpoint["owner"] or "-",
        "checkpoint_sha256": checkpoint["sha256"] or "",
    }


def probe_arguments(payload: Mapping[str, Any], task: Mapping[str, str]) -> List[str]:
    """Hydra arguments of one task: the frozen config plus the task's explicit settings.

    The same overrides, in the same order, as the bash launchers passed, so run configs and
    cache identities do not change.
    """
    arguments = ["--config-path", str(Path(payload["config_snapshot"]).parent), "--config-name", "probe"]
    if payload.get("config_tree_sha256") is not None:  # snapshots with the paths/ and dataset/ groups
        arguments += [f"paths={payload.get('paths_profile') or LAUNCHER_PATHS_PROFILE}", f"dataset={task['profile_id']}"]
    arguments += [
        f"dataset.name={task['dataset_name']}", f"dataset.animal={task['animal']}",
        f"dataset.root={task['root']}", f"dataset.metadata_file={task['metadata_file']}",
        f"dataset.label_col={task['label_col']}", f"dataset.mask_col={task['mask_col']}",
        f"dataset.no_background={task['no_background']}", f"dataset.image_variant={task['image_variant']}",
        f"dataset.split_col={task['split_col']}", f"dataset.database_split_value={task['database_split_value']}",
        f"dataset.query_split_value={task['query_split_value']}", f"dataset.calibration_size={task['calibration_size']}",
        f"benchmark.method={task['method']}", f"benchmark.candidate_k={task['candidate_k']}",
    ]
    method = task["method"]
    if method in PROBE_METHODS:
        arguments.append(f"benchmark.methods.{method}.train_mode={task['train_mode']}")
        weighting = {"weighted": "inverse_frequency", "unweighted": "none"}.get(task["class_weighting"])
        if weighting:
            arguments.append(f"benchmark.methods.{method}.class_weighting={weighting}")
    if method == "vismatch":
        arguments.append(f"benchmark.methods.vismatch.matcher={task['matcher']}")
        if task["matcher"] == "loma":
            arguments.append(f"benchmark.methods.vismatch.loma_arch={task['loma_arch']}")
        if task["checkpoint_source"] == "custom":
            arguments += [
                "benchmark.methods.vismatch.checkpoint_source=custom",
                f"benchmark.methods.vismatch.checkpoint_path={task['checkpoint_path']}",
                f"benchmark.methods.vismatch.checkpoint_components={task['checkpoint_components']}",
                f"benchmark.methods.vismatch.checkpoint_owner={task['checkpoint_owner']}",
                f"benchmark.methods.vismatch.evaluation_animal={task['evaluation_animal']}",
            ]
        else:
            arguments.append("benchmark.methods.vismatch.checkpoint_source=default")
    return arguments


def evaluate_command(arguments: Sequence[str]) -> List[str]:
    return [sys.executable, "-m", "wildmatch", "evaluate", *arguments]


def task_log_paths(logs_root: Path, task: Mapping[str, str], index: int, job_id: str) -> Dict[str, Path]:
    """``<logs>/<dataset>/<animal>/<split>/job-<id>/task-NNN__<split>__<slug>__<checkpoint>__k<k>``."""
    split = _sanitize(task["split_col"])
    slug = task["method"] if task["matcher"] == "-" else f"{task['method']}-{task['matcher']}"
    if task["method"] in PROBE_METHODS:
        slug = f"{slug}-{task['train_mode']}-{task['class_weighting']}"
    folder = logs_root / _sanitize(task["dataset_name"]) / _sanitize(task["animal"]) / split / f"job-{job_id}"
    stem = f"task-{index:03d}__{split}__{_sanitize(slug)}__{_sanitize(task['checkpoint_label'])}__k{task['candidate_k']}"
    return {suffix: folder / f"{stem}.{suffix}" for suffix in ("out", "err", "combined.log", "json")}


def _record_fields(task: Mapping[str, str], index: int, job_id: str, payload: Mapping[str, Any],
                   manifest_path: Path, paths: Mapping[str, Path]) -> Dict[str, Any]:
    return dict(
        job_id=job_id, task_id=index, dataset=_sanitize(task["dataset_name"]), animal=_sanitize(task["animal"]),
        split_protocol=task["split_col"], method=task["method"], matcher=task["matcher"],
        train_mode=task["train_mode"], class_weighting=task["class_weighting"],
        checkpoint=task["checkpoint_label"], checkpoint_path=task["checkpoint_path"],
        candidate_k=int(task["candidate_k"]), stdout_path=str(paths["out"]), stderr_path=str(paths["err"]),
        combined_path=str(paths["combined.log"]), submission_id=payload["submission_id"],
        manifest_path=str(manifest_path), profile_id=task["profile_id"],
        checkpoint_owner=task["checkpoint_owner"], checkpoint_sha256=task["checkpoint_sha256"],
    )


def _cancel_own_array_element() -> None:
    array_job, array_task, job = (os.environ.get(k) for k in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID"))
    target = f"{array_job}_{array_task}" if array_job and array_task else job
    if target and shutil.which("scancel"):
        subprocess.run(["scancel", target], check=False)


def _pump(stream, targets, lock) -> None:
    """Copy bytes as they arrive (like ``tee``), so progress-bar carriage returns stay intact."""
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            break
        with lock:
            for handle in targets:
                handle.write(chunk)
                handle.flush()
    stream.close()


class _TextSink:
    """Byte writes to a text-only console stream (for example a captured ``StringIO``)."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, chunk: bytes) -> None:
        self.stream.write(chunk.decode("utf-8", errors="replace"))

    def flush(self) -> None:
        self.stream.flush()


def _console(stream):
    stream.flush()
    return getattr(stream, "buffer", None) or _TextSink(stream)


def _run_teed(command: Sequence[str], paths: Mapping[str, Path]) -> int:
    """Run ``command``, mirroring stdout/stderr to the console and the task's log files."""
    lock = threading.Lock()
    environment = dict(os.environ, PYTHONUNBUFFERED="1")  # live logs; no effect on results
    with paths["out"].open("ab") as out, paths["err"].open("ab") as err, paths["combined.log"].open("ab") as combined:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        threads = [threading.Thread(target=_pump, args=(process.stdout, (_console(sys.stdout), out, combined), lock)),
                   threading.Thread(target=_pump, args=(process.stderr, (_console(sys.stderr), err, combined), lock))]
        for thread in threads:
            thread.start()
        code = process.wait()
        for thread in threads:
            thread.join()
    return code


def _run_directory(stdout_path: Path) -> str:
    if not stdout_path.is_file():
        return ""
    saved = [line[len("Saved JSON: "):].strip() for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("Saved JSON: ")]
    return str(Path(saved[-1]).parent) if saved and Path(saved[-1]).is_file() else ""


def run_task(manifest_path: Path, index: int, logs_root: Path = DEFAULT_LOGS_ROOT, job_id: Optional[str] = None,
             dry_run: bool = False) -> int:
    """Validate and run task ``index`` of a submission; returns its exit code."""
    manifest_path = Path(manifest_path).resolve()
    payload, record = load_task(manifest_path, index)
    task = task_from_record(record)
    job_id = job_id or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    paths = task_log_paths(logs_root, task, index, job_id)
    index_path = logs_root.parent / "index.csv"
    fields = _record_fields(task, index, job_id, payload, manifest_path, paths)
    try:
        validate_task(payload, record)
    except ValueError as exc:
        if dry_run:
            raise
        paths["json"].parent.mkdir(parents=True, exist_ok=True)
        message = f"[sweep] immutable task validation failed: {exc}\n"
        sys.stderr.write(message)
        for name in ("err", "combined.log"):
            with paths[name].open("a", encoding="utf-8") as handle:
                handle.write(message)
        logs.init_record(paths["json"], status="failed", command="validation-only: no probe execution",
                         start_time=_now(), validation_status="failed", **fields)
        logs.update_record(paths["json"], status="failed", end_time=_now(), validation_status="failed",
                           validation_error=str(exc), error_file=paths["err"])
        logs.write_index(logs_root, index_path)
        _cancel_own_array_element()
        return 1

    command = evaluate_command(probe_arguments(payload, task))
    command_text = shlex.join(command)
    header = (f"Starting sweep task {index}\n{describe_task(index, task)}\n"
              f"Submission ID: {payload['submission_id']}\nManifest: {manifest_path}\n"
              f"Config snapshot: {payload['config_snapshot']}\nCheckpoint owner: {task['checkpoint_owner']}\n"
              f"Checkpoint SHA256: {task['checkpoint_sha256']}\n{command_text}\n")
    sys.stdout.write(header)
    sys.stdout.flush()
    if dry_run:
        return 0
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    for name in ("out", "combined.log"):
        with paths[name].open("a", encoding="utf-8") as handle:
            handle.write(header)
    logs.init_record(paths["json"], status="running", command=command_text, start_time=_now(),
                     validation_status="validated", **fields)
    code = 1
    try:
        code = _run_teed(command, paths)
    finally:
        logs.update_record(paths["json"], status="completed" if code == 0 else "failed", end_time=_now(),
                           experiment_run_directory=_run_directory(paths["out"]), error_file=paths["err"],
                           validation_status="validated")
        logs.write_index(logs_root, index_path)
    return code


def _submission_id(submissions_root: Path) -> str:
    """``<UTC time>-<pid>`` as the launchers wrote it, with ``-2``, ``-3``, ... on a collision."""
    base = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    candidate, number = base, 1
    while (submissions_root / candidate).exists():
        number += 1
        candidate = f"{base}-{number}"
    return candidate


def sweep_main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Build, freeze and run an evaluation sweep.")
    parser.add_argument("spec", help="packaged sweep name (conf/sweep/<name>.yaml) or a YAML file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-tasks", action="store_true", help="print the task table; writes nothing")
    mode.add_argument("--dry-run", action="store_true", help="freeze the submission and print the sbatch command")
    mode.add_argument("--submit", action="store_true", help="freeze the submission and submit a Slurm array")
    mode.add_argument("--local", action="store_true", help="freeze the submission and run every task here, in order")
    parser.add_argument("--paths", help="path profile (default: WILDMATCH_PATHS, wildmatch.local.yaml, default)")
    parser.add_argument("--config", type=Path, help="probe.yaml to freeze instead of the packaged one "
                        "(its paths/ and dataset/ groups come from the package unless they sit next to it)")
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--max-concurrent", type=int, help="Slurm array throttle; overrides the spec")
    parser.add_argument("--sbatch-script", type=Path, default=DEFAULT_SBATCH_SCRIPT)
    parser.add_argument("--sbatch-arg", action="append", default=[], metavar="ARG",
                        help="extra sbatch argument, e.g. --sbatch-arg=--partition=gpu (repeatable)")
    args = parser.parse_args(argv)
    try:
        spec_path = resolve_spec_path(args.spec)
        spec = load_spec(spec_path)
        profile = active_profile(args.paths)
        tasks = build_tasks(spec, profile)
    except SweepError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    if args.list_tasks:
        for index, task in enumerate(tasks):
            print(describe_task(index, task))
        return 0

    max_concurrent = args.max_concurrent or int(spec["max_concurrent"])
    if max_concurrent <= 0:
        parser.exit(2, f"{parser.prog}: --max-concurrent must be positive\n")
    if args.submit and not args.sbatch_script.is_file():
        parser.exit(2, f"{parser.prog}: sbatch script not found: {args.sbatch_script} (run from the repository root)\n")
    submission_id = _submission_id(args.logs_root / "submissions")
    try:
        manifest = create_submission(tasks, args.logs_root / "submissions" / submission_id, submission_id,
                                     spec_path, profile, config_file=args.config)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"Immutable submission manifest: {manifest}")
    array = f"0-{len(tasks) - 1}%{max_concurrent}"
    sbatch = ["sbatch", f"--array={array}", f"--export=ALL,WILDMATCH_SWEEP_MANIFEST={manifest}",
              *args.sbatch_arg, str(args.sbatch_script)]
    if args.dry_run:
        print("Dry run: no Slurm array submitted.")
        print(shlex.join(sbatch))
        for index, task in enumerate(tasks):
            print(describe_task(index, task))
        return 0
    if args.submit:
        print(f"Submitting {len(tasks)} sweep tasks with array throttle {max_concurrent}.")
        return subprocess.run(sbatch, check=False).returncode
    failed = [index for index in range(len(tasks))
              if run_task(manifest, index, args.logs_root, job_id=f"local-{submission_id}") != 0]
    print(f"Local sweep finished: {len(tasks) - len(failed)} of {len(tasks)} tasks succeeded"
          + (f"; failed: {', '.join(map(str, failed))}" if failed else "."))
    return 1 if failed else 0


def sweep_task_main(argv: Optional[Sequence[str]] = None, prog: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Run one task of a frozen sweep submission.")
    parser.add_argument("--manifest", type=Path, default=os.environ.get("WILDMATCH_SWEEP_MANIFEST"))
    parser.add_argument("--index", type=int, default=os.environ.get("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--job-id")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the command without running it")
    args = parser.parse_args(argv)
    if args.manifest is None or args.index is None:
        parser.exit(2, f"{parser.prog}: --manifest and --index are required outside a Slurm array\n")
    try:
        return run_task(args.manifest, int(args.index), args.logs_root, args.job_id, args.dry_run)
    except ValueError as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
