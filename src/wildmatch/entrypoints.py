"""Hydra entry points for single evaluation (probe) and backbone fine-tuning runs.

Configs ship inside the package (``wildmatch/conf``). The job names are pinned in the
configs (``hydra.job.name``), so Hydra's job log stays ``logs/hydra/probe.log`` and
``logs/hydra/finetune.log`` although the decorated functions live in this module.
"""

from __future__ import annotations

import sys
from typing import Iterable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

_REMOVED_PROBE_BUDGET_KEYS = {
    "benchmark.map_at_k",
    "benchmark.methods.local_lightglue.B",
    "benchmark.methods.vismatch.candidate_k",
    "benchmark.methods.wildfusion.B",
}


def reject_removed_probe_budget_overrides(argv: Optional[Iterable[str]] = None) -> None:
    for argument in sys.argv[1:] if argv is None else argv:
        key = argument.split("=", 1)[0]
        if key in _REMOVED_PROBE_BUDGET_KEYS:
            raise ValueError(f"'{key}' is no longer configurable; use benchmark.candidate_k instead.")


@hydra.main(version_base="1.3", config_path="conf", config_name="probe")
def probe_main(cfg: DictConfig) -> None:
    """Run a single retrieval benchmark with Hydra configuration overrides."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    from wildmatch.evaluate.probe_runner import run_probe

    run_probe(cfg)


@hydra.main(version_base="1.3", config_path="conf", config_name="finetune")
def finetune_main(cfg: DictConfig) -> None:
    """Run one backbone fine-tuning experiment with Hydra configuration overrides."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    from wildmatch.train.finetune_runner import run_finetune

    run_finetune(cfg)


@hydra.main(version_base="1.3", config_path="conf", config_name="finetune_matcher")
def finetune_matcher_main(cfg: DictConfig) -> None:
    """Fine-tune LoMa or RDD-LightGlue on a mined index (wildmatch.matcher_finetune.launch)."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    from wildmatch.matcher_finetune.launch import run

    code = run(cfg)
    if code:
        raise SystemExit(code)


def apply_paths_profile(argv: list) -> Optional[str]:
    """Add ``paths=<profile>`` from WILDMATCH_PATHS or wildmatch.local.yaml when not given.

    Skipped when the command already selects a profile or loads its own config with
    ``--config-path``/``-cp`` (submission snapshots and copied run configs, which may predate
    the ``paths`` group). Returns the profile that was added, if any.
    """
    from wildmatch.paths import DEFAULT_PROFILE, active_profile

    arguments = argv[1:]
    if any(a.split("=", 1)[0] in ("paths", "+paths", "++paths") for a in arguments):
        return None
    if any(a in ("--config-path", "-cp") or a.startswith("--config-path=") for a in arguments):
        return None
    profile = active_profile()
    if profile == DEFAULT_PROFILE:
        return None
    argv.append(f"paths={profile}")
    return profile


def probe() -> None:
    reject_removed_probe_budget_overrides()
    apply_paths_profile(sys.argv)
    probe_main()


def finetune() -> None:
    apply_paths_profile(sys.argv)
    finetune_main()


def finetune_matcher() -> None:
    apply_paths_profile(sys.argv)
    finetune_matcher_main()
