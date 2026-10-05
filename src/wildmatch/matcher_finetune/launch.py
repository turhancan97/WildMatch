"""`wildmatch finetune-matcher`: plan and launch one matcher fine-tuning run.

A port of the lynx-finetuning Slurm wrappers (`slurm/matcher_finetune/train_{wildlife,czechlynx}_
{loma,rdd}.sh`): the same layout of views, mined indices, feature caches and output folders, the
same trainer arguments and the same protocol JSON, but with every location taken from the path
profile and the dataset registry instead of environment variables and absolute paths. The trainers
themselves are unchanged; this module only builds their command line and runs it under
`accelerate launch`.

Two layouts exist, as in the wrappers. CzechLynx entries (`dataset.name == "CzechLynx_v2"`) use
`CzechLynx_processed_<split>/` views, indices under `<mining_outputs>/czechlynx-<split>/<protocol>/`
and outputs under `<checkpoint_root>/czechlynx-<split>/`; the other entries use the
WildlifeReID-10k layout keyed by `dataset.animal`.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

TRAINERS = {
    "loma": "wildmatch.matcher_finetune.train_loma_matches",
    "rdd": "wildmatch.matcher_finetune.train_by_lg_matches",
}
LOMA_COMPONENTS = ("matcher", "descriptor", "joint")
# RDD wrapper vocabulary: component -> (--trained_model, --rdd_train_component).
RDD_COMPONENTS = {
    "lg": ("lg", "all"),
    "descriptor": ("rdd", "descriptor"),
    "rdd": ("rdd", "all"),
    "all": ("rdd", "all"),
    "lg+rdd": ("lg+rdd", "all"),
    "joint": ("lg+rdd", "descriptor"),
}
CZECHLYNX_SPLITS = ("split-time_closed", "split-time_open")
PROTOCOLS = ("legacy", "strict")
MINERS = ("loma", "rdd")


@dataclass
class Plan:
    """Everything one run needs; `command` is the full `accelerate launch` command line."""

    layout: str
    matcher: str
    component: str
    data_root: Path
    train_index: Path
    val_index: Path
    output_dir: Path
    protocol_file: Path
    protocol_text: str
    trainer_args: list[str]
    command: list[str]
    cache: Optional[Path] = None
    resume: Optional[Path] = None
    notes: list[str] = field(default_factory=list)


def _require(value: Any, what: str) -> Path:
    if value in (None, ""):
        raise ValueError(f"{what} is not set (set it in the path profile or pass it explicitly)")
    return Path(str(value))


def _choice(value: str, allowed, what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def czechlynx_layout(split_column: str, protocol: str, miner: str, mining_outputs: Path) -> dict[str, Any]:
    """`czechlynx_resolve_protocol` of slurm/matcher_finetune/czechlynx_protocol.sh, without overrides."""
    _choice(split_column, CZECHLYNX_SPLITS, "the CzechLynx split column")
    _choice(protocol, PROTOCOLS, "protocol")
    _choice(miner, MINERS, "mined_by")
    suffix = split_column.removeprefix("split-")
    experiment = "czechlynx-" + suffix.replace("_", "-")
    experiment_root = mining_outputs / experiment / protocol
    backend_root = experiment_root / miner
    if (
        miner == "rdd"
        and (experiment_root / "strong-matches_train_combined.json").is_file()
        and not backend_root.exists()
    ):
        index_root = experiment_root
    else:
        index_root = backend_root
    val_name = "test" if protocol == "legacy" else "val"
    return {
        "split_suffix": suffix,
        "experiment": experiment,
        "index_root": index_root,
        "train_index": index_root / "strong-matches_train_combined.json",
        "val_index": index_root / f"strong-matches_{val_name}_combined.json",
        "output_suffix": protocol,
    }


def wildlife_index_root(index_base: Path, miner: str) -> Path:
    """`<base>/<miner>/` when it holds the training index, else the older shared `<base>/`."""
    candidate = index_base / miner
    if (candidate / "strong-matches_train_combined.json").is_file() or not (
        index_base / "strong-matches_train_combined.json"
    ).is_file():
        return candidate
    return index_base


def _resume(cfg_resume: Any, output_dir: Path, marker: str) -> Optional[Path]:
    """The wrappers' resume rule: `auto` takes the newest complete epoch; never overwrite epochs."""
    if cfg_resume in (None, ""):
        if any(output_dir.glob("epoch_*")):
            raise ValueError(
                f"{output_dir} already holds epoch directories; pass resume=auto or resume=<epoch dir>, "
                "or choose another output_dir"
            )
        return None
    if str(cfg_resume) == "auto":
        done = [d for d in output_dir.glob("epoch_*") if (d / marker).is_file()]
        if not done:
            raise ValueError(f"resume=auto: no epoch directory with {marker} under {output_dir}")
        return max(done, key=lambda d: int(d.name.split("_", 1)[1]))
    return Path(str(cfg_resume))


def plan_run(cfg: Mapping[str, Any]) -> Plan:
    """Build the run plan from a resolved `finetune_matcher` config (paths, dataset, matcher_finetune)."""
    paths, dataset, mf = cfg["paths"], cfg["dataset"], cfg["matcher_finetune"]
    matcher = _choice(str(mf["matcher"]), tuple(TRAINERS), "matcher_finetune.matcher")
    component = str(mf["component"])
    protocol = _choice(str(mf["protocol"]), PROTOCOLS, "matcher_finetune.protocol")
    miner = _choice(str(mf["mined_by"]), MINERS, "matcher_finetune.mined_by")
    checkpoint_root = _require(paths["checkpoint_root"], "paths.checkpoint_root")
    data_root = _require(paths["data_root"], "paths.data_root")
    layout = "czechlynx" if dataset["name"] == "CzechLynx_v2" else "wildlife"
    notes: list[str] = []

    if matcher == "loma":
        _choice(component, LOMA_COMPONENTS, "matcher_finetune.component for LoMa")
        allowed = LOMA_COMPONENTS if layout == "czechlynx" else ("matcher", "descriptor")
    else:
        _choice(component, tuple(RDD_COMPONENTS), "matcher_finetune.component for RDD")
        allowed = tuple(RDD_COMPONENTS) if layout == "czechlynx" else tuple(c for c in RDD_COMPONENTS if c != "joint")
    if component not in allowed:
        raise ValueError(f"component {component!r} is only defined for CzechLynx (as in the original wrappers)")

    if layout == "czechlynx":
        mining_outputs = _require(paths["external"]["mining_outputs"], "paths.external.mining_outputs")
        lx = czechlynx_layout(str(dataset["split_col"]), protocol, miner, mining_outputs)
        view = data_root / f"CzechLynx_processed_{lx['split_suffix']}"
        train_index, val_index = lx["train_index"], lx["val_index"]
        base = checkpoint_root / lx["experiment"]
        mined = f"{miner}-mined-{lx['output_suffix']}"
        if matcher == "loma":
            cache = base / ("loma-b-cache" if component == "matcher" else "loma-b-keypoint-cache")
            output = (
                base
                / {
                    "descriptor": f"loma-b-descriptor-finetuned-{lx['output_suffix']}",
                    "joint": f"loma-b-joint-finetuned-{mined}",
                    "matcher": f"loma-b-finetuned-{lx['output_suffix']}",
                }[component]
            )
            run_name = (
                f"czechlynx-{lx['experiment']}-loma"
                + {"descriptor": "-descriptor", "joint": "-joint"}.get(component, "")
                + f"-{protocol}"
            )
            project = f"lynx-{lx['experiment']}-loma"
        else:
            cache = base / "rdd-cache"
            tag = {"descriptor": "rdd-descriptor", "joint": "rdd-joint", "lg": "rdd"}.get(component, "rdd-full")
            output = base / f"{tag}-finetuned-{mined}"
            name_tag = {"descriptor": "rdd-descriptor", "joint": "rdd-joint", "lg": "rdd"}.get(component, component)
            run_name = f"czechlynx-{lx['experiment']}-{name_tag}-{protocol}-relaxed"
            project = f"lynx-{lx['experiment']}-rdd"
        protocol_name = "czechlynx_protocol.json"
        keep_every_default = 50
    else:
        mining_outputs = _require(paths["external"]["mining_outputs"], "paths.external.mining_outputs")
        animal = str(dataset["animal"])
        view = data_root / "wildlife_processed" / animal / protocol
        index_root = wildlife_index_root(mining_outputs / "wildlife-reid-10k" / animal / "indices", miner)
        if index_root.name != miner:
            notes.append(f"using the older shared index folder {index_root} (no {miner}/ subfolder)")
        train_index = index_root / "strong-matches_train_combined.json"
        val_index = index_root / f"strong-matches_{'test' if protocol == 'legacy' else 'val'}_combined.json"
        base = checkpoint_root / "wildlife-reid-10k" / animal
        if matcher == "loma":
            cache = base / ("loma-cache" if component == "matcher" else "loma-keypoint-cache")
            tag = "loma-descriptor" if component == "descriptor" else "loma"
            output = base / f"{tag}-finetuned" / protocol
            run_name = f"{animal}-{tag}-{protocol}-finetuned"
            project = f"wildlife-reid-loma-{animal}-{protocol}"
            keep_every_default = None
        else:
            cache = base / "rdd-cache"
            tag = {"descriptor": "rdd-descriptor", "lg": "rdd"}.get(component, "rdd-full")
            output = base / f"{tag}-finetuned" / protocol
            name_tag = {"descriptor": "rdd-descriptor", "lg": "rdd"}.get(component, component)
            run_name = f"{animal}-{name_tag}-{protocol}-finetuned-relaxed"
            project = f"wildlife-reid-rdd-{animal}-{protocol}"
            keep_every_default = 50
        protocol_name = "wildlife_protocol.json"

    fewshot = mf.get("fewshot") or {}
    if fewshot.get("fraction") is not None:
        # Few-shot view (wildmatch.mining.wildlife_fewshot): the full view's caches stay valid, while the
        # view, its mined indices and the checkpoints live in the few-shot layout.
        from wildmatch.mining.launch import fewshot_dirs
        from wildmatch.mining.wildlife_fewshot import view_name

        fraction, seed = float(fewshot["fraction"]), int(fewshot.get("seed") or 0)
        if layout == "czechlynx" and dataset["split_col"] != "split-time_closed":
            raise ValueError("few-shot CzechLynx views exist for split-time_closed only")
        name = "CzechLynx" if layout == "czechlynx" else str(dataset["animal"])
        dirs = fewshot_dirs(_require(paths.get("fewshot_root"), "paths.fewshot_root"), name, protocol, fraction, seed)
        view = dirs["views"]
        train_index = dirs["indices"] / miner / "strong-matches_train_combined.json"
        val_index = (
            dirs["indices"] / miner / f"strong-matches_{'test' if protocol == 'legacy' else 'val'}_combined.json"
        )
        output = dirs["checkpoints"] / f"{matcher}-finetuned"
        run_name = f"{name}-{matcher}-{protocol}-{view_name(fraction, seed)}"
    view = Path(str(mf["data_root"])) if mf.get("data_root") else view
    train_index = Path(str(mf["train_index"])) if mf.get("train_index") else train_index
    val_index = Path(str(mf["val_index"])) if mf.get("val_index") else val_index
    output = Path(str(mf["output_dir"])) if mf.get("output_dir") else output
    cache = Path(str(mf["cache"])) if mf.get("cache") else cache
    run_name = mf["wandb"].get("run_name") or run_name
    project = mf["wandb"].get("project") or project
    keep_every = mf.get("keep_every")
    keep_every = keep_every_default if keep_every is None else int(keep_every)
    num_processes = int(mf["num_processes"])
    # RDD descriptor and joint training run one image per GPU with 8 accumulation steps (wrappers).
    small = matcher == "rdd" and component in ("descriptor", "joint")
    batch_size = int(mf["batch_size"]) if mf.get("batch_size") is not None else (1 if small else 8)
    resume = _resume(mf.get("resume"), output, "metadata.json" if matcher == "loma" else "train_state.json")

    if matcher == "loma":
        weights = _require(mf.get("weights"), "matcher_finetune.weights (paths.external.loma_weights)")
        args = [
            "--trained_model", "loma", "--loma_train_component", component,
            "--train_index", str(train_index), "--val_index", str(val_index),
            "--data_root", str(view), "--loma_weights", str(weights),
            "--output_dir", str(output), "--project", project,
            "--run_name", run_name, "--split_protocol", protocol, "--wandb_mode", str(mf["wandb"]["mode"]),
            "--loma_variant", str(mf["variant"]), "--epochs", str(mf["epochs"]),
            "--batch_size", str(batch_size), "--lr", str(mf["lr"]), "--weight_decay", str(mf["weight_decay"]),
            "--margin", str(mf["margin"]), "--random_negative_prob", str(mf["random_negative_prob"]),
            "--num_workers", str(mf["num_workers"]), "--eval_every_epochs", str(mf["eval_every_epochs"]),
            "--seed", str(mf["seed"]), "--resize", str(mf["resize"]), "--num_keypoints", str(mf["num_keypoints"]),
            "--descriptor_microbatch_size", str(mf["descriptor_microbatch_size"]),
        ]  # fmt: skip
        if keep_every is not None:
            args += ["--keep_every", str(keep_every)]
        args += ["--loma_cache" if component == "matcher" else "--loma_keypoint_cache", str(cache)]
        uses_cache = True
        if layout == "czechlynx":
            protocol_text = (
                "{\n"
                f'  "split_column": "{dataset["split_col"]}",\n'
                f'  "backend": "{miner}",\n'
                f'  "protocol": "{protocol}",\n'
                f'  "train_index": "{train_index}",\n'
                f'  "validation_index": "{val_index}",\n'
                '  "final_evaluation_split": "test",\n'
                f'  "loma_train_component": "{component}",\n'
                '  "training_score": "relaxed_v1",\n'
                '  "optimizer": "adamw"\n'
                "}\n"
            )
        else:
            protocol_text = (
                f'{{"dataset": "{dataset["animal"]}", "protocol": "{protocol}", "train_component": "{component}", '
                f'"train_index": "{train_index}", "validation_index": "{val_index}", '
                f'"final_evaluation_split": "test"}}\n'
            )
    else:
        trained_model, rdd_component = RDD_COMPONENTS[component]
        grad_accum = int(mf["grad_accum_steps"]) if mf.get("grad_accum_steps") is not None else (8 if small else 1)
        rdd_dir = paths["external"].get("rdd_weights_dir")
        rdd_weights = mf.get("rdd_weights") or (rdd_dir and f"{rdd_dir}/RDD-v2.pth")
        lg_weights = mf.get("lg_weights") or (rdd_dir and f"{rdd_dir}/RDD_lg-v2.pth")
        args = [
            "--train_index", str(train_index), "--val_index", str(val_index),
            "--data_root", str(view), "--rdd_weights", str(_require(rdd_weights, "paths.external.rdd_weights_dir")),
            "--lg_weights", str(_require(lg_weights, "paths.external.rdd_weights_dir")), "--output_dir", str(output),
            "--project", project, "--run_name", run_name,
            "--split_protocol", protocol, "--trained_model", trained_model,
            "--rdd_train_component", rdd_component,
            "--epochs", str(mf["epochs"]), "--batch_size", str(batch_size),
            "--grad_accum_steps", str(grad_accum), "--keep_every", str(keep_every),
            "--lr", str(mf["lr"]), "--weight_decay", str(mf["weight_decay"]),
            "--num_workers", str(mf["num_workers"]), "--lg_margin", str(mf["margin"]),
            "--random_negative_prob", str(mf["random_negative_prob"]),
            "--resize", str(mf["resize"]), "--top_k", str(mf["num_keypoints"]),
        ]  # fmt: skip
        if layout == "wildlife":
            args += ["--eval_every_epochs", str(mf["eval_every_epochs"])]
        args += ["--seed", str(mf["seed"])]
        uses_cache = trained_model == "lg"
        if uses_cache:
            args += ["--keypoint_cache", str(cache)]
        common = {
            "protocol": protocol,
            "backend": miner,
            "training_score": "relaxed_v1",
            "optimizer": "adamw",
            "batch_size_per_gpu": batch_size,
            "grad_accum_steps": grad_accum,
            "num_gpus": num_processes,
        }
        if layout == "czechlynx":
            protocol_text = (
                "{\n"
                f'  "split_column": "{dataset["split_col"]}",\n'
                f'  "backend": "{miner}",\n'
                f'  "protocol": "{protocol}",\n'
                f'  "train_index": "{train_index}",\n'
                f'  "validation_index": "{val_index}",\n'
                '  "final_evaluation_split": "test",\n'
                f'  "rdd_train_component": "{component}",\n'
                f'  "trained_model": "{trained_model}",\n'
                f'  "rdd_component": "{rdd_component}",\n'
                '  "training_score": "relaxed_v1",\n'
                '  "optimizer": "adamw",\n'
                f'  "batch_size_per_gpu": {batch_size},\n'
                f'  "grad_accum_steps": {grad_accum},\n'
                f'  "num_gpus": {num_processes}\n'
                "}\n"
            )
        else:
            protocol_text = (
                f'{{"dataset": "{dataset["animal"]}", "protocol": "{protocol}", "backend": "{miner}", '
                f'"train_component": "{component}", "train_index": "{train_index}", '
                f'"validation_index": "{val_index}", "final_evaluation_split": "test", '
                f'"training_score": "{common["training_score"]}", "optimizer": "{common["optimizer"]}", '
                f'"batch_size_per_gpu": {batch_size}, "grad_accum_steps": {grad_accum}, '
                f'"num_gpus": {num_processes}}}\n'
            )
        json.loads(protocol_text)  # the f-strings above must stay valid JSON
    if resume is not None:
        args += ["--resume", str(resume)]

    command = [
        "accelerate", "launch", "--num_processes", str(num_processes), "--num_machines", "1",
        "--mixed_precision", "no", "--dynamo_backend", "no", "-m", TRAINERS[matcher], *args,
    ]  # fmt: skip
    return Plan(
        layout=layout,
        matcher=matcher,
        component=component,
        data_root=view,
        train_index=train_index,
        val_index=val_index,
        output_dir=output,
        protocol_file=output / protocol_name,
        protocol_text=protocol_text,
        trainer_args=args,
        command=command,
        cache=cache if uses_cache else None,
        resume=resume,
        notes=notes,
    )


def check_inputs(plan: Plan) -> list[str]:
    """Missing inputs, as messages (the run is refused when any is missing)."""
    problems = [f"missing {what}: {p}" for what, p in (("train index", plan.train_index), ("validation index",
                plan.val_index), ("view", plan.data_root)) if not p.exists()]  # fmt: skip
    if plan.cache is not None and not (plan.cache / "manifest.json").is_file():
        problems.append(
            f"missing feature cache {plan.cache} (manifest.json); build it first "
            "(RDD: python -m wildmatch.matcher_finetune.build_keypoint_cache; LoMa: the mining cache builder)"
        )
    return problems


PROVENANCE_FILE = "wildmatch_provenance.json"


def _flag(args: list[str], flag: str) -> Optional[Path]:
    return Path(args[args.index(flag) + 1]) if flag in args else None


def record_launch(plan: Plan, cfg: Mapping[str, Any]) -> Path:
    """Append this launch to `<output_dir>/wildmatch_provenance.json` (code, inputs, seed, packages).

    A separate file: the checkpoint loader reads only the protocol JSON, whose contents are part of
    the checkpoint fingerprint, so provenance is kept out of it.
    """
    from wildmatch.utils.provenance import append_launch, launch_record

    args = plan.trainer_args
    inputs = {
        "train_index": plan.train_index,
        "validation_index": plan.val_index,
        "pretrained_loma": _flag(args, "--loma_weights"),
        "pretrained_rdd": _flag(args, "--rdd_weights"),
        "pretrained_lightglue": _flag(args, "--lg_weights"),
        "cache_manifest": plan.cache / "manifest.json" if plan.cache is not None else None,
        "resume_from": plan.resume / "model.safetensors" if plan.resume is not None else None,
    }
    mf = cfg["matcher_finetune"]
    record = launch_record(
        plan.command,
        {k: v for k, v in inputs.items() if v is not None},
        matcher=plan.matcher,
        component=plan.component,
        layout=plan.layout,
        view=str(plan.data_root),
        seed=int(mf["seed"]),
        protocol_file=plan.protocol_file.name,
    )
    return append_launch(plan.output_dir / PROVENANCE_FILE, record)


def run(cfg: Mapping[str, Any]) -> int:
    plan = plan_run(cfg)
    print(f"layout={plan.layout} matcher={plan.matcher} component={plan.component}")
    print(f"view={plan.data_root}\ntrain index={plan.train_index}\nvalidation index={plan.val_index}")
    print(f"output directory={plan.output_dir}")
    for note in plan.notes:
        print(f"note: {note}")
    print("command: " + shlex.join(plan.command))
    if cfg["matcher_finetune"].get("dry_run"):
        return 0
    problems = check_inputs(plan)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    plan.protocol_file.write_text(plan.protocol_text, encoding="utf-8")
    record_launch(plan, cfg)
    # The wrappers exported WANDB_MODE; the RDD trainer reads it from the environment.
    env = {**os.environ, "WANDB_MODE": str(cfg["matcher_finetune"]["wandb"]["mode"])}
    return subprocess.call(plan.command, env=env)
