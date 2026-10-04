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
            raise ValueError(
                f"'{key}' is no longer configurable; use benchmark.candidate_k instead."
            )


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


def probe() -> None:
    reject_removed_probe_budget_overrides()
    probe_main()


def finetune() -> None:
    finetune_main()
