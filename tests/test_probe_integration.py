"""End-to-end probe runs on the bundled demo images with a tiny fake backbone (CPU, no download).

The backbone factory is patched, so ``wildmatch evaluate`` runs its real pipeline (Hydra config,
dataset views, feature cache, method, metrics, run directory, manifest, run index) on the 24
synthetic demo renders without any network access. Numbers are meaningless; the tests check the
contract of a completed run.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional
from unittest import mock

import numpy as np

try:
    import torch
    from torch import nn

    from wildmatch.demo import overrides as demo_overrides
    from wildmatch.entrypoints import probe_main

    HAS_PROBE_DEPS = True
except Exception:  # pragma: no cover - optional heavy dependencies
    HAS_PROBE_DEPS = False

REQUIRED_RUN_FILES = ("config.snapshot.yaml", "run_manifest.json", "metrics.json", "timings.json", "scores.npz")


def fake_get_model(model_type):
    """Mimic ``wildmatch.models.model.get_model``: a deterministic 16-d CNN at 64 px."""
    torch.manual_seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 16),
    )
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return backbone, 16, mean, std, 64, "swin", None, None


def run_probe(output: Path, extra: List[str]) -> Path:
    """Run ``wildmatch evaluate`` on the demo data under ``output`` and return the run directory."""
    arguments = [
        "wildmatch evaluate",
        "paths=default",
        *demo_overrides("cosine", "megadescriptor-t", output),
        f"hydra.run.dir={output}/hydra",
        "visualization.enabled=false",
        *extra,
    ]
    with (
        mock.patch("wildmatch.evaluate.probe_runner.get_model", fake_get_model),
        mock.patch.object(sys, "argv", arguments),
    ):
        probe_main()
    manifests = sorted((output / "experiments").rglob("run_manifest.json"))
    if len(manifests) != 1:
        raise AssertionError(f"expected one run under {output}, found {len(manifests)}")
    return manifests[0].parent


def read_index(output: Path) -> List[dict]:
    with (output / "runs.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


LINEAR_PROBE = [
    "benchmark.method=linear_probe",
    "benchmark.methods.linear_probe.epochs=2",
    "benchmark.methods.linear_probe.num_workers=0",
    "benchmark.methods.linear_probe.eval_num_workers=0",
    "benchmark.methods.linear_probe.batch_size=4",
    "benchmark.methods.linear_probe.accumulation_steps=1",
    "benchmark.top_k=[1,5]",  # identity-level metrics: the demo has six identities
]


@unittest.skipUnless(HAS_PROBE_DEPS, "probe dependencies are not installed")
class ProbeIntegrationTest(unittest.TestCase):
    def assert_completed_run(self, output: Path, run_dir: Path, method: str) -> dict:
        for name in REQUIRED_RUN_FILES:
            self.assertTrue((run_dir / name).is_file(), f"{name} missing in {run_dir}")
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        for key in ("top_1", "top_5"):
            self.assertGreaterEqual(float(metrics[key]), 0.0)
            self.assertLessEqual(float(metrics[key]), 1.0)
        rows = read_index(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["method"], method)
        self.assertEqual((rows[0]["num_query"], rows[0]["num_database"]), ("6", "18"))
        self.assertTrue(rows[0]["git_commit"], "run index row without git_commit")
        return metrics

    def test_cosine_run_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_dir = run_probe(output, ["benchmark.method=cosine"])
            self.assert_completed_run(output, run_dir, "cosine")

    def test_cosine_run_is_deterministic(self):
        scores: List[np.ndarray] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = run_probe(Path(tmp), ["benchmark.method=cosine"])
                with np.load(run_dir / "scores.npz") as data:
                    scores.append({key: data[key].copy() for key in data.files})
        self.assertEqual(scores[0].keys(), scores[1].keys())
        for key in scores[0]:
            np.testing.assert_array_equal(scores[0][key], scores[1][key])

    def test_per_epoch_test_pass_does_not_change_results(self):
        results = []
        for flag in ("false", "true"):
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = run_probe(
                    Path(tmp), [*LINEAR_PROBE, f"benchmark.methods.linear_probe.log_test_each_epoch={flag}"]
                )
                with np.load(run_dir / "scores.npz") as data:
                    scores = {key: data[key].copy() for key in data.files}
                metrics = json.loads((run_dir / "metrics.json").read_text())
                results.append((scores, metrics))
        (scores_off, metrics_off), (scores_on, metrics_on) = results
        for key in scores_off:
            np.testing.assert_array_equal(scores_off[key], scores_on[key])
        for key in ("top_1", "top_5", "balanced_top_1", "classification_top_1"):
            self.assertEqual(metrics_off[key], metrics_on[key], key)

    def test_linear_probe_run_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            run_dir = run_probe(output, LINEAR_PROBE)
            self.assert_completed_run(output, run_dir, "linear_probe")


def probe_metrics(extra: List[str], output: Optional[Path] = None) -> dict:
    """Metrics of one demo run (helper for tests that compare configurations)."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = run_probe(Path(tmp) if output is None else output, extra)
        return json.loads((run_dir / "metrics.json").read_text())


if __name__ == "__main__":
    unittest.main()
