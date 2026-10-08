from pathlib import Path

SBATCH = Path(__file__).resolve().parents[2] / "slurm" / "finetune_matcher.sbatch"


def test_finetune_sbatch_sets_expandable_segments_by_default():
    lines = SBATCH.read_text().splitlines()
    export = 'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
    assert export in lines
    # set before the trainer is launched, so accelerate's child processes inherit it
    assert lines.index(export) < next(
        i for i, line in enumerate(lines) if line.startswith("wildmatch finetune-matcher")
    )
