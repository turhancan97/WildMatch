"""`wildmatch mine`: build views and feature caches, mine positive/negative pairs, aggregate the index.

A port of the rdd-parallel-benchmark Slurm wrappers (`slurm/mining/`) and of the feature-cache
wrappers of lynx-finetuning (`slurm/matcher_finetune/build_*_cache.sh`): the same views, caches,
index prefixes, module arguments and Slurm array layout, with every location taken from the path
profile and the dataset registry instead of environment variables and absolute paths. The mining
modules themselves are unchanged; each step runs one of them as `python -m wildmatch.mining.<module>`
(or `wildmatch.matcher_finetune.build_keypoint_cache` for the RDD cache, as the wrappers did).

Steps:
  plan       print the layout and every command (writes nothing)
  view       build the symlink view the miner and the trainers read
  cache      build the backend's per-frame feature cache (GPU)
  check      validate the view and the cache before mining (the spawn scripts' check)
  task       mine one query collection (one Slurm array element: --split and --index)
  aggregate  combine the per-query files into strong-matches_<split>_combined.json
  submit     check, then submit one Slurm array per split and the dependent aggregation job

Two layouts exist, as in the wrappers: CzechLynx entries (`dataset.name == "CzechLynx_v2"`) and the
WildlifeReID-10k/Salamander layout keyed by `dataset.animal`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

PACKAGE_CONFIGS = Path(__file__).resolve().parent / "configs" / "wildlife"
CZECHLYNX_SPLITS = ("split-time_closed", "split-time_open")
STEPS = ("plan", "view", "cache", "check", "task", "aggregate", "submit")


@dataclass
class MinePlan:
    layout: str
    key: str
    dataset_id: str
    backend: str
    protocol: str
    view: Path
    cache: Path
    weights: Path
    rdd_weights: Path
    report: Path
    splits: tuple[str, ...]
    view_command: list[str]
    cache_command: list[str]
    aggregate_command: list[str]
    variant: str = "loma-b"
    split_column: Optional[str] = None

    def task_command(self, split: str, index: int, frames: int = 20, top_k: int = 5, top_m: int = 10) -> list[str]:
        common = ["--variant", self.variant, "--split", split, "--query_id", str(index),
                  "--frames_per_collection", str(frames), "--top_k_frames", str(top_k), "--top_m", str(top_m),
                  "--dump_report", str(self.report)]  # fmt: skip
        if self.layout == "czechlynx":
            return [sys.executable, "-m", "wildmatch.mining.czechlynx_mine", "--dataset_root", str(self.view),
                    "--cache_dir", str(self.cache), "--rdd_weights", str(self.rdd_weights),
                    "--lg_weights", str(self.weights), "--weights", str(self.weights),
                    "--backend", self.backend, *common]  # fmt: skip
        return [sys.executable, "-m", "wildmatch.mining.wildlife_mine", "--dataset_id", self.dataset_id,
                "--dataset_root", str(self.view), "--cache_dir", str(self.cache), "--lg_weights", str(self.weights),
                "--backend", self.backend, *common]  # fmt: skip


def _required(value, what: str) -> Path:
    if value in (None, ""):
        raise SystemExit(f"{what} is not set in the path profile")
    return Path(str(value))


def plan_mining(key: str, backend: str, protocol: str, profile: Optional[str] = None) -> MinePlan:
    from wildmatch.data.registry import load_dataset
    from wildmatch.paths import load_paths

    if backend not in ("loma", "rdd"):
        raise SystemExit(f"--backend must be loma or rdd, got {backend!r}")
    if protocol not in ("legacy", "strict"):
        raise SystemExit(f"--protocol must be legacy or strict, got {protocol!r}")
    paths = load_paths(profile)
    entry = load_dataset(key, profile)
    data_root = _required(paths.data_root, "data_root")
    checkpoint_root = _required(paths.checkpoint_root, "checkpoint_root")
    mining_outputs = _required(paths.external.mining_outputs, "external.mining_outputs")
    rdd_dir = paths.external.get("rdd_weights_dir")
    rdd_weights = Path(f"{rdd_dir}/RDD-v2.pth") if rdd_dir else None
    lg_weights = Path(f"{rdd_dir}/RDD_lg-v2.pth") if rdd_dir else None
    loma_weights = paths.external.get("loma_weights")
    weights = Path(str(loma_weights)) if backend == "loma" and loma_weights else lg_weights
    weights = _required(weights, "external.loma_weights" if backend == "loma" else "external.rdd_weights_dir")
    py = sys.executable

    if entry.name == "CzechLynx_v2":
        split_column = str(entry.split_col)
        if split_column not in CZECHLYNX_SPLITS:
            raise SystemExit(f"CzechLynx mining needs split-time_closed or split-time_open, got {split_column!r}")
        suffix = split_column.removeprefix("split-")
        experiment = "czechlynx-" + suffix.replace("_", "-")
        view = data_root / f"CzechLynx_processed_{suffix}"
        base = checkpoint_root / experiment
        cache = base / ("loma-b-cache" if backend == "loma" else "rdd-cache")
        report = mining_outputs / experiment / protocol / backend / "strong-matches"
        splits = ("train", "val", "test")
        view_command = [py, "-m", "wildmatch.mining.czechlynx_dataset", "--source_root", str(data_root / "CzechLynx_v2"),
                        "--output_root", str(view), "--validation_fraction", "0.2", "--seed", "0",
                        "--split_column", split_column]  # fmt: skip
        aggregate = [py, "-m", "wildmatch.mining.czechlynx_aggregate", "--dump_report", str(report),
                     "--dataset_root", str(view), "--splits", "train", "val", "test"]  # fmt: skip
        layout, dataset_id, cache_batch = "czechlynx", experiment, "4"
    else:
        dataset_id = str(entry.animal)
        config = PACKAGE_CONFIGS / f"{dataset_id}.json"
        if not config.is_file():
            raise SystemExit(f"no mining config for {dataset_id} ({config})")
        view = data_root / "wildlife_processed" / dataset_id / protocol
        base = checkpoint_root / "wildlife-reid-10k" / dataset_id
        cache = base / f"{backend}-cache"
        report = mining_outputs / "wildlife-reid-10k" / dataset_id / "indices" / backend / "strong-matches"
        splits = ("train", "val", "test") if protocol == "strict" else ("train", "test")
        view_command = [py, "-m", "wildmatch.mining.wildlife_dataset", "--config", str(config),
                        "--output_root", str(view), "--protocol", protocol]  # fmt: skip
        aggregate = [py, "-m", "wildmatch.mining.wildlife_aggregate", "--dataset_id", dataset_id,
                     "--protocol", protocol, "--dump_report", str(report), "--dataset_root", str(view),
                     "--backend", backend, "--variant", "loma-b", "--weights", str(weights),
                     "--cache_dir", str(cache)]  # fmt: skip
        layout, split_column, cache_batch = "wildlife", None, "8"

    if backend == "loma":
        cache_command = [py, "-m", "wildmatch.mining.lynx_build_loma_cache", "--dataset_root", str(view),
                         "--cache_dir", str(cache), "--weights", str(weights), "--variant", "loma-b",
                         "--all_frames", "--splits", *splits, "--resize_max", "512", "--num_keypoints", "512",
                         "--batch_size", "4", "--num_workers", "16", "--resume"]  # fmt: skip
    else:
        cache_command = [py, "-m", "wildmatch.matcher_finetune.build_keypoint_cache", "--data_root", str(view),
                         "--cache_root", str(cache), "--rdd_weights", str(rdd_weights), "--splits", *splits,
                         "--resize", "512", "--top_k", "512", "--batch_size", cache_batch,
                         "--num_workers", "16", "--resume"]  # fmt: skip
    return MinePlan(
        layout=layout, key=key, dataset_id=dataset_id, backend=backend, protocol=protocol, view=view,
        cache=cache, weights=weights,
        rdd_weights=(weights if backend == "loma" else _required(rdd_weights, "external.rdd_weights_dir")),
        report=report, splits=splits, view_command=view_command, cache_command=cache_command,
        aggregate_command=aggregate, split_column=split_column,
    )  # fmt: skip


def count_collections(plan: MinePlan, split: str) -> int:
    root = plan.view / split
    if plan.layout == "czechlynx":
        return sum(1 for identity in root.iterdir() if identity.is_dir() for c in identity.iterdir() if c.is_dir())
    from wildmatch.mining.wildlife_dataset import list_collections

    return len(list_collections(plan.view, split))


def check_inputs(plan: MinePlan) -> list[str]:
    """The spawn scripts' pre-submission check: view, weights, and a complete, well-formed cache."""
    import numpy as np

    if not plan.view.is_dir():
        return [f"missing view: {plan.view} (run `wildmatch mine view`)"]
    if not plan.weights.exists():
        return [f"missing {plan.backend} weights: {plan.weights}"]
    frames = []
    for split in plan.splits:
        if not (plan.view / split).is_dir():
            return [f"missing split {split} under {plan.view}"]
        frames.extend(sorted((plan.view / split).rglob("frame_*.jpg")))
    if not frames:
        return [f"no frames under {plan.view}"]
    problems = []
    if plan.backend == "loma":
        manifest = plan.cache / "manifest.json"
        if not manifest.is_file():
            return [f"missing LoMa cache manifest: {manifest} (run `wildmatch mine cache`)"]
        meta = json.loads(manifest.read_text())
        expected = {"backend": "loma", "variant": plan.variant, "resize": 512, "num_keypoints": 512, "patch_size": 14}
        problems += [f"LoMa cache {k}={meta.get(k)!r}, expected {v!r}" for k, v in expected.items() if meta.get(k) != v]
    full, compact = {"keypoints", "descriptors", "scores", "image_size"}, {"k", "d", "hw"}
    missing = malformed = 0
    for frame in frames:
        path = plan.cache / frame.relative_to(plan.view).with_suffix(".npz")
        if not path.is_file():
            missing += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                fields = set(data.files)
        except Exception:
            malformed += 1
            continue
        if not full <= fields and not (plan.backend == "rdd" and compact <= fields):
            malformed += 1
    if missing or malformed:
        problems.append(f"cache {plan.cache}: {missing} missing, {malformed} malformed of {len(frames)} frames")
    return problems


def _print_plan(plan: MinePlan) -> None:
    print(
        f"layout={plan.layout} dataset={plan.key} ({plan.dataset_id}) backend={plan.backend} protocol={plan.protocol}"
    )
    print(f"view={plan.view}\ncache={plan.cache}\nweights={plan.weights}\nreport={plan.report}")
    print(f"splits={' '.join(plan.splits)}")
    print("view: " + shlex.join(plan.view_command))
    print("cache: " + shlex.join(plan.cache_command))
    print("task: " + shlex.join(plan.task_command(plan.splits[0], 0)))
    print("aggregate: " + shlex.join(plan.aggregate_command))


def _submit(plan: MinePlan, args: argparse.Namespace) -> int:
    combined = [Path(f"{plan.report}_{s}_combined.json") for s in plan.splits]
    if any(p.exists() for p in combined) and not args.overwrite:
        print(f"refusing: {plan.report}_*_combined.json exists (pass --overwrite to mine again)", file=sys.stderr)
        return 1
    problems = check_inputs(plan)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    script = Path(args.slurm_script)
    common = ["--dataset", plan.key, "--backend", plan.backend, "--protocol", plan.protocol]
    if args.paths:
        common += ["--paths", args.paths]
    if args.report is not None:
        common += ["--report", str(args.report)]
    if args.overwrite:
        common += ["--overwrite"]
    log_root = Path("logs") / "mining" / plan.dataset_id / plan.protocol / plan.backend
    jobs = []
    for split in plan.splits:
        count = count_collections(plan, split)
        if count < 1:
            print(f"no {split} collections under {plan.view}", file=sys.stderr)
            return 1
        (log_root / split).mkdir(parents=True, exist_ok=True)
        command = ["sbatch", "--parsable", f"--array=0-{count - 1}%{args.max_concurrent}",
                   f"--output={log_root / split}/%A_%a.out", f"--error={log_root / split}/%A_%a.err",
                   str(script), "task", *common, "--split", split]  # fmt: skip
        print(shlex.join(command))
        if not args.dry_run:
            jobs.append(subprocess.check_output(command, text=True).strip().split(";")[0])
            print(f"{plan.backend} {split} mining: {jobs[-1]} ({count} collections)")
    command = ["sbatch", "--parsable", f"--dependency=afterok:{':'.join(jobs) or '<jobs>'}",
               f"--output={log_root}/aggregate-%j.out", str(script), "aggregate", *common]  # fmt: skip
    print(shlex.join(command))
    if not args.dry_run:
        print("aggregation: " + subprocess.check_output(command, text=True).strip())
    return 0


def main(argv: Optional[Sequence[str]] = None, prog: str = "wildmatch mine") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--dataset", required=True, help="registry key, e.g. salamander or czechlynx_closed")
    parser.add_argument("--backend", choices=["loma", "rdd"], required=True)
    parser.add_argument("--protocol", choices=["legacy", "strict"], default="legacy")
    parser.add_argument("--paths", default=None, help="path profile (default: WILDMATCH_PATHS / wildmatch.local.yaml)")
    parser.add_argument("--split", default=None, help="task: the query split")
    parser.add_argument("--index", type=int, default=None, help="task: query collection (default SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--max-concurrent", type=int, default=30, help="submit: Slurm array %% throttle")
    parser.add_argument("--slurm-script", default="slurm/mine.sbatch", help="submit: script running `wildmatch mine`")
    parser.add_argument("--dry-run", action="store_true", help="submit: print the sbatch commands only")
    parser.add_argument("--overwrite", action="store_true", help="replace existing views or mined files")
    parser.add_argument(
        "--report", type=Path, default=None, help="index prefix (default: under external.mining_outputs)"
    )
    args = parser.parse_args(argv)
    plan = plan_mining(args.dataset, args.backend, args.protocol, args.paths)
    if args.report is not None:
        plan.report = args.report
        plan.aggregate_command[plan.aggregate_command.index("--dump_report") + 1] = str(args.report)

    if args.step == "plan":
        _print_plan(plan)
        return 0
    if args.step == "view":
        command = plan.view_command + (["--force"] if args.overwrite else [])
        return subprocess.call(command)
    if args.step == "cache":
        return subprocess.call(plan.cache_command)
    if args.step == "check":
        problems = check_inputs(plan)
        print("\n".join(problems) if problems else f"ready: view, weights and cache for {plan.dataset_id}")
        return 1 if problems else 0
    if args.step == "task":
        import os

        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        if args.split not in plan.splits:
            raise SystemExit(f"--split must be one of {', '.join(plan.splits)}")
        output = Path(f"{plan.report}_{args.split}_{index}.json")
        if output.exists() and not args.overwrite:
            print(f"refusing: {output} exists (pass --overwrite to mine it again)", file=sys.stderr)
            return 1
        return subprocess.call(plan.task_command(args.split, index))
    if args.step == "aggregate":
        return subprocess.call(plan.aggregate_command)
    return _submit(plan, args)
