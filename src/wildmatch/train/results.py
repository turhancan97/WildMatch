"""Dependency-light serialization helpers for training model selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping

SELECTION_MODES = ("final", "best_on_test")


def resolve_selection(value: Any) -> str:
    """Validate ``output.selection``: which checkpoint a backbone fine-tuning run reports."""
    selection = str(value)
    if selection not in SELECTION_MODES:
        raise ValueError(f"output.selection must be one of {SELECTION_MODES}, got {selection!r}")
    return selection


def build_final_training_metrics(
    selected_metrics: Mapping[str, Any],
    final_epoch_metrics: Mapping[str, Any],
    *,
    best_epoch: int,
    best_metric: str,
    selected_checkpoint: str,
    selection: str = "best_on_test",
    best_metric_value: Any = None,
) -> Dict[str, Any]:
    """Primary metrics of the reported checkpoint, with the final-epoch metrics kept alongside.

    ``selection="final"`` reports the final-epoch model. ``"best_on_test"`` reports the epoch
    with the best ``best_metric`` on the evaluation split, which is the test split; the metrics
    record that selection so it is never mistaken for an unbiased estimate. ``best_epoch`` and
    ``best_metric_value`` always describe the best-on-test epoch seen during training
    (``checkpoint-best.pth``).
    """
    result = dict(selected_metrics)
    result.update(
        {
            "selection": str(selection),
            "selected_on": "test" if selection == "best_on_test" else "final_epoch",
            "best_epoch": int(best_epoch),
            "best_metric": str(best_metric),
            "best_metric_value": (
                selected_metrics.get(best_metric) if best_metric_value is None else best_metric_value
            ),
            "selected_checkpoint": str(selected_checkpoint),
            "final_epoch_metrics": dict(final_epoch_metrics),
        }
    )
    if selection == "best_on_test":
        result["best_checkpoint_metrics"] = dict(selected_metrics)
    return result
