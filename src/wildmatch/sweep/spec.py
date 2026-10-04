"""Sweep specs and the task table.

A spec is a YAML mapping::

    datasets: [salamander]          # registry keys (conf/dataset/<key>.yaml)
    candidate_k: [50, 250]          # classifier probes run once, at the first value
    max_concurrent: 12              # Slurm array throttle (%N)
    variants:
      - {method: cosine}
      - {method: vismatch, matcher: loma}                       # default weights
      - {method: vismatch, matcher: loma, checkpoint: custom}   # fine-tuned, matcher only
      - {method: linear_probe, train_mode: classifier, class_weighting: weighted}
    dataset_overrides:              # optional, per registry key
      salamander:
        checkpoints: {custom: {loma: /path/to/epoch_299/model.safetensors}}
        checkpoint_owner: SalamanderID2025     # descriptor-fine-tuned rows only
        evaluation_animal: SalamanderID2025

Checkpoint labels follow the run manifests: ``default`` (Vismatch weights), ``custom``
(matcher only), ``descriptor-fine-tuned`` (descriptor only) and ``joint-fine-tuned``
(descriptor + matcher, loaded as a complete model). Fine-tuned paths come from the registry
entry's ``registry.checkpoints.<label>.<matcher>`` unless the row or ``dataset_overrides``
names one. The rules are the bash launchers' rules; every violation fails before anything
is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from importlib import resources

from omegaconf import DictConfig, OmegaConf

from wildmatch.data.registry import load_dataset

TASK_FIELDS = (
    "profile_id",
    "dataset_name",
    "animal",
    "root",
    "metadata_file",
    "label_col",
    "mask_col",
    "no_background",
    "image_variant",
    "split_col",
    "database_split_value",
    "query_split_value",
    "calibration_size",
    "method",
    "matcher",
    "checkpoint_label",
    "checkpoint_path",
    "checkpoint_owner",
    "checkpoint_components",
    "loma_arch",
    "train_mode",
    "class_weighting",
    "candidate_k",
    "evaluation_animal",
)

METHODS = ("cosine", "wildfusion", "local_lightglue", "linear_probe", "efficient_probe", "vismatch")
PROBE_METHODS = ("linear_probe", "efficient_probe")
MATCHERS = ("loma", "rdd-lightglue")
TRAIN_MODES = ("classifier", "partial", "all")
CLASS_WEIGHTINGS = ("weighted", "unweighted")
# Each fine-tuned checkpoint label loads exactly one component mode.
COMPONENTS = {"custom": "matcher_only", "descriptor-fine-tuned": "descriptor_only", "joint-fine-tuned": "full"}
CHECKPOINT_LABELS = ("default", *COMPONENTS)
LOMA_ARCH = "LoMa-B"
UNSEEN_EVAL_KEY = "czechlynx_unseen_eval"
_VARIANT_KEYS = {"method", "matcher", "checkpoint", "checkpoint_path", "components", "train_mode", "class_weighting"}
_SPEC_KEYS = {"datasets", "candidate_k", "max_concurrent", "variants", "dataset_overrides", "description"}
_OVERRIDE_KEYS = {"checkpoints", "checkpoint_owner", "evaluation_animal"}


class SweepError(ValueError):
    """A sweep spec or task violates the launcher rules."""


def packaged_specs() -> List[str]:
    folder = resources.files("wildmatch") / "conf" / "sweep"
    return sorted(entry.name[:-5] for entry in folder.iterdir() if entry.name.endswith(".yaml"))


def resolve_spec_path(name_or_path: str) -> Path:
    """A path to an existing YAML file, or the name of a packaged spec (``conf/sweep``)."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        if not candidate.is_file():
            raise SweepError(f"sweep spec does not exist: {candidate}")
        return candidate
    packaged = resources.files("wildmatch") / "conf" / "sweep" / f"{name_or_path}.yaml"
    if not packaged.is_file():
        raise SweepError(f"unknown sweep {name_or_path!r}; packaged sweeps: {', '.join(packaged_specs())}")
    return Path(str(packaged))


def load_spec(path: Path) -> Dict[str, Any]:
    raw = OmegaConf.load(path)
    if not isinstance(raw, DictConfig):
        raise SweepError(f"sweep spec must be a mapping: {path}")
    spec = OmegaConf.to_container(raw, resolve=True)
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise SweepError(f"unknown sweep keys: {', '.join(sorted(unknown))}")
    for key in ("datasets", "candidate_k", "variants"):
        if not spec.get(key):
            raise SweepError(f"sweep spec needs a non-empty '{key}' list")
    for value in spec["candidate_k"]:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SweepError(f"candidate_k must be a positive integer; got {value!r}")
    max_concurrent = spec.setdefault("max_concurrent", 12)
    if not isinstance(max_concurrent, int) or max_concurrent <= 0:
        raise SweepError(f"max_concurrent must be a positive integer; got {max_concurrent!r}")
    for override in (spec.get("dataset_overrides") or {}).values():
        unknown = set(override or {}) - _OVERRIDE_KEYS
        if unknown:
            raise SweepError(f"unknown dataset_overrides keys: {', '.join(sorted(unknown))}")
    return spec


def _text(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _variant(row: Mapping[str, Any]) -> Dict[str, str]:
    unknown = set(row) - _VARIANT_KEYS
    if unknown:
        raise SweepError(f"unknown variant keys: {', '.join(sorted(unknown))}")
    variant = {key: _text(row.get(key)) for key in _VARIANT_KEYS}
    if variant["checkpoint"] == "-":
        variant["checkpoint"] = "default"
    method = variant["method"]
    if method not in METHODS:
        raise SweepError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")
    if method == "vismatch" and variant["matcher"] not in MATCHERS:
        raise SweepError(f"vismatch rows need matcher {' or '.join(MATCHERS)}; got {variant['matcher']!r}")
    if method != "vismatch" and variant["matcher"] != "-":
        raise SweepError(f"only vismatch rows take a matcher; got {variant['matcher']!r} for {method}")
    if method in PROBE_METHODS:
        if variant["train_mode"] not in TRAIN_MODES:
            raise SweepError(f"{method} variant must use train_mode classifier, partial, or all")
        if variant["class_weighting"] not in CLASS_WEIGHTINGS:
            raise SweepError(f"{method} variant must specify class_weighting weighted or unweighted")
    else:
        if variant["train_mode"] != "-":
            raise SweepError(
                f"only classifier-probe variants may specify train_mode; got '{variant['train_mode']}' for {method}"
            )
        if variant["class_weighting"] != "-":
            raise SweepError(
                f"only classifier-probe variants may specify class_weighting; got '{variant['class_weighting']}' for {method}"
            )
    label = variant["checkpoint"]
    if label not in CHECKPOINT_LABELS:
        raise SweepError(f"unknown checkpoint label {label!r}; expected one of {', '.join(CHECKPOINT_LABELS)}")
    if label == "default":
        if variant["components"] != "-" or variant["checkpoint_path"] != "-":
            raise SweepError("default-checkpoint rows take no components or checkpoint_path")
    else:
        if method != "vismatch":
            raise SweepError("custom checkpoint variants are only valid for vismatch")
        expected = COMPONENTS[label]
        if variant["components"] == "-":
            variant["components"] = expected
        if variant["components"] != expected:
            raise SweepError(f"{label} checkpoints must use checkpoint_components={expected}")
    return variant


def _checkpoint_path(entry: DictConfig, overrides: Mapping[str, Any], label: str, matcher: str, explicit: str) -> str:
    if explicit != "-":
        return explicit
    override = ((overrides.get("checkpoints") or {}).get(label) or {}).get(matcher)
    if override:
        return str(override)
    value = OmegaConf.select(entry, f"registry.checkpoints.{label}.{matcher}")
    if value in (None, ""):
        raise SweepError(
            f"{entry.registry.key} has no {label} {matcher} checkpoint in the registry; "
            "name one with checkpoint_path or dataset_overrides"
        )
    return str(value)


def _check_unseen_eval(entry: DictConfig) -> None:
    metadata = Path(str(entry.metadata_file))
    if not metadata.is_absolute():
        metadata = Path(str(entry.root)) / metadata
    if not metadata.is_file():
        raise SweepError(
            f"{UNSEEN_EVAL_KEY} metadata file does not exist: {metadata} "
            "(build it with `wildmatch build-unseen-split`; set CZECHLYNX_UNSEEN_EVAL_METADATA_FILE)"
        )
    if (str(entry.split_col), str(entry.database_split_value), str(entry.query_split_value)) != (
        "unseen_eval_split",
        "database",
        "query",
    ):
        raise SweepError(f"{UNSEEN_EVAL_KEY} must use unseen_eval_split with database/query values")


def build_tasks(spec: Mapping[str, Any], profile: Optional[str] = None) -> List[Dict[str, str]]:
    """The task table: datasets x candidate budgets x variants, in that nesting order."""
    variants = [_variant(row) for row in spec["variants"]]
    overrides_by_key = spec.get("dataset_overrides") or {}
    unknown = set(overrides_by_key) - set(spec["datasets"])
    if unknown:
        raise SweepError(f"dataset_overrides for datasets not in the sweep: {', '.join(sorted(unknown))}")
    budgets = [int(value) for value in spec["candidate_k"]]
    tasks: List[Dict[str, str]] = []
    for key in spec["datasets"]:
        try:
            entry = load_dataset(str(key), profile)
        except KeyError as exc:
            raise SweepError(str(exc.args[0])) from exc
        if str(key) == UNSEEN_EVAL_KEY:
            _check_unseen_eval(entry)
        overrides = overrides_by_key.get(key) or {}
        animal = str(entry.animal)
        evaluation_animal = str(overrides.get("evaluation_animal") or animal)
        descriptor_owner = str(overrides.get("checkpoint_owner") or animal)
        for candidate_k in budgets:
            for variant in variants:
                method = variant["method"]
                if method in PROBE_METHODS and candidate_k != budgets[0]:
                    continue
                label, matcher = variant["checkpoint"], variant["matcher"]
                path, owner = "-", "-"
                if label != "default":
                    path = _checkpoint_path(entry, overrides, label, matcher, variant["checkpoint_path"])
                    owner = descriptor_owner if label == "descriptor-fine-tuned" else animal
                tasks.append(
                    {
                        "profile_id": str(key),
                        "dataset_name": str(entry.name),
                        "animal": animal,
                        "root": str(entry.root),
                        "metadata_file": str(entry.metadata_file),
                        "label_col": str(entry.label_col),
                        "mask_col": str(entry.mask_col),
                        "no_background": _text(bool(entry.no_background)),
                        "image_variant": str(entry.image_variant),
                        "split_col": str(entry.split_col),
                        "database_split_value": str(entry.database_split_value),
                        "query_split_value": str(entry.query_split_value),
                        "calibration_size": str(int(entry.calibration_size)),
                        "method": method,
                        "matcher": matcher,
                        "checkpoint_label": label,
                        "checkpoint_path": path,
                        "checkpoint_owner": owner,
                        "checkpoint_components": variant["components"],
                        "loma_arch": LOMA_ARCH if matcher == "loma" else "-",
                        "train_mode": variant["train_mode"],
                        "class_weighting": variant["class_weighting"],
                        "candidate_k": str(candidate_k),
                        "evaluation_animal": evaluation_animal,
                    }
                )
    identities = [
        tuple(
            task[field]
            for field in (
                "profile_id",
                "candidate_k",
                "method",
                "matcher",
                "checkpoint_label",
                "train_mode",
                "class_weighting",
            )
        )
        for task in tasks
    ]
    if len(set(identities)) != len(identities):
        raise SweepError("the sweep repeats a task (same dataset, budget, method, matcher, checkpoint and probe mode)")
    return tasks


def task_line(task: Mapping[str, str]) -> str:
    """One ``tasks.tsv`` row (the launchers' pipe-separated format)."""
    return "|".join(task[field] for field in TASK_FIELDS)


def describe_task(index: int, task: Mapping[str, str]) -> str:
    """The ``--list-tasks`` line (the CzechLynx launcher's format, which names the split)."""
    return (
        f"index={index} profile={task['profile_id']} split_protocol={task['split_col']} "
        f"dataset={task['dataset_name']} animal={task['animal']} evaluation_animal={task['evaluation_animal']} "
        f"candidate_k={task['candidate_k']} method={task['method']} matcher={task['matcher']} "
        f"train_mode={task['train_mode']} class_weighting={task['class_weighting']} "
        f"checkpoint={task['checkpoint_label']} components={task['checkpoint_components']} "
        f"checkpoint_owner={task['checkpoint_owner']} path={task['checkpoint_path']}"
    )
