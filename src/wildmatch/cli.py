"""The ``wildmatch`` command: one entry point with a subcommand per workflow.

``evaluate`` and ``finetune-backbone`` are Hydra applications: everything after the
subcommand is passed to Hydra unchanged (``benchmark.method=vismatch``, ``dataset=salamander``,
``--config-path``, ``--help``, ...). The other subcommands take ordinary options; run
``wildmatch <subcommand> --help``.

Subcommands are imported only when used, so ``wildmatch --help`` and the light tools start
without importing PyTorch.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence


class Command(NamedTuple):
    target: str  # "module:function"
    summary: str
    hydra: bool = False


COMMANDS: Dict[str, Command] = {
    "evaluate": Command("wildmatch.entrypoints:probe", "run one evaluation (Hydra overrides)", hydra=True),
    "finetune-backbone": Command(
        "wildmatch.entrypoints:finetune", "fine-tune the ArcFace backbone (Hydra overrides)", hydra=True
    ),
    "finetune-matcher": Command(
        "wildmatch.entrypoints:finetune_matcher",
        "fine-tune LoMa or RDD-LightGlue on a mined index (Hydra overrides)",
        hydra=True,
    ),
    "mine": Command(
        "wildmatch.mining.launch:main", "build views and caches, mine pairs, aggregate the index (plan|view|cache|...)"
    ),
    "sweep": Command(
        "wildmatch.sweep.runner:sweep_main", "build, freeze and run a method x budget grid (Slurm or local)"
    ),
    "sweep-task": Command("wildmatch.sweep.runner:sweep_task_main", "run one task of a frozen sweep submission"),
    "summarize-runs": Command("wildmatch.reporting.summarize_runs:main", "list completed runs from reports/runs.csv"),
    "summarize-logs": Command("wildmatch.sweep.logs:main", "list sweep task logs; rebuild logs/index.csv"),
    "check-index": Command(
        "wildmatch.reporting.check_index:main", "find runs.csv rows whose run directory moved or is gone"
    ),
    "audit-runs": Command(
        "wildmatch.reporting.audit_runs:main", "report run manifests and index rows that predate provenance fields"
    ),
    "tables": Command("wildmatch.reporting.export_tables:main", "export LaTeX/CSV result tables from experiments/"),
    "figures": Command("wildmatch.reporting.export_figures:main", "plot accuracy versus candidate budget"),
    "build-unseen-split": Command("wildmatch.data.unseen_split:main", "build the unseen-identity evaluation split"),
    "class-balance": Command(
        "wildmatch.reporting.class_balance:main", "per-identity image counts of every dataset split"
    ),
    "audit": Command("wildmatch.data.image_quality:main", "flag candidate low-quality images for review"),
    "prepare": Command(
        "wildmatch.data.prepare:main",
        "check, download or rebuild datasets (status, download, build, retry-empty, finish, compare-masks, jaguar, unseen-split)",
    ),
    "weights": Command("wildmatch.weights:main", "list, verify, download or stage the paper checkpoints"),
    "demo": Command("wildmatch.demo:main", "run the pipeline on 24 bundled synthetic lynx renders (CPU)"),
}


def _resolve(target: str) -> Callable:
    module, function = target.split(":")
    return getattr(import_module(module), function)


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: wildmatch <command> [arguments]", "", "commands:"]
    lines += [f"  {name:<{width}}  {command.summary}" for name, command in COMMANDS.items()]
    lines += ["", "Run `wildmatch <command> --help` for the options of one command."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments: List[str] = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    if arguments[0] == "--version":
        from importlib.metadata import version

        print(f"wildmatch {version('wildmatch')}")
        return 0
    name, rest = arguments[0], arguments[1:]
    command = COMMANDS.get(name)
    if command is None:
        print(f"wildmatch: unknown command {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    function = _resolve(command.target)
    prog = f"wildmatch {name}"
    if command.hydra:
        # Hydra parses sys.argv itself.
        sys.argv = [prog, *rest]
        function()
        return 0
    result = function(rest, prog=prog)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
