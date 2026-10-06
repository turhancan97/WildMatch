"""End-to-end backbone fine-tuning on the bundled demo images with a tiny fake backbone (CPU).

``wildmatch finetune-backbone`` runs its real loop (ArcFace training, per-epoch evaluation,
checkpoints, manifest, run index) for two epochs on the 24 synthetic demo renders, with the
backbone factory patched so nothing is downloaded. The demo's gallery is the training split and
its queries the evaluation split.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest import mock

try:
    from test_probe_integration import fake_get_model  # pytest puts tests/ on sys.path

    from wildmatch.demo import data_root
    from wildmatch.entrypoints import finetune_main

    HAS_FINETUNE_DEPS = True
except Exception:  # pragma: no cover - optional heavy dependencies
    HAS_FINETUNE_DEPS = False


def run_finetune(output: Path, extra: List[str]) -> Path:
    arguments = [
        "wildmatch finetune-backbone",
        "paths=default",
        "dataset.name=WildMatchDemo",
        "dataset.animal=SyntheticLynx",
        f"dataset.root={data_root()}",
        "dataset.metadata_file=metadata.csv",
        "dataset.label_col=identity",
        "dataset.split_col=split",
        "dataset.train_split_value=database",
        "dataset.val_split_value=query",
        "dataset.no_background=false",
        "dataset.image_variant=no_background",
        "model.type=megadescriptor-t",
        "train.epochs=2",
        "train.batch_size=6",
        "train.num_workers=0",
        "train.accumulation_steps=1",
        "benchmark.top_k=[1,5]",
        "benchmark.val_num_workers=0",
        "output.save_every=1",
        f"output.experiment_root={output}/experiments",
        f"output.run_dir={output}/results",
        f"output.csv_path={output}/results/train_metrics.csv",
        f"reporting.index_path={output}/runs.csv",
        f"hydra.run.dir={output}/hydra",
        "wandb.enabled=false",
        *extra,
    ]
    with (
        mock.patch("wildmatch.train.finetune_runner.get_model", fake_get_model),
        mock.patch.object(sys, "argv", arguments),
    ):
        finetune_main()
    manifests = sorted((output / "experiments").rglob("run_manifest.json"))
    if len(manifests) != 1:
        raise AssertionError(f"expected one run under {output}, found {len(manifests)}")
    return manifests[0].parent


@unittest.skipUnless(HAS_FINETUNE_DEPS, "fine-tuning dependencies are not installed")
class FinetuneIntegrationTest(unittest.TestCase):
    def test_default_reports_the_final_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_finetune(Path(tmp), [])
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["selection"], "final")
            self.assertEqual(metrics["selected_on"], "final_epoch")
            self.assertTrue(metrics["selected_checkpoint"].endswith("checkpoint-final.pth"))
            self.assertEqual(metrics["top_1"], metrics["final_epoch_metrics"]["top_1"])
            self.assertNotIn("best_checkpoint_metrics", metrics)
            for name in ("checkpoint-final.pth", "checkpoint-final-full.pth", "checkpoint-epoch-2.pth"):
                self.assertTrue((run_dir / name).is_file(), name)

    def test_best_on_test_is_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_finetune(Path(tmp), ["output.selection=best_on_test"])
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertEqual(metrics["selection"], "best_on_test")
            self.assertEqual(metrics["selected_on"], "test")
            self.assertTrue(metrics["selected_checkpoint"].endswith("checkpoint-best.pth"))
            self.assertIn("best_checkpoint_metrics", metrics)

    def test_unknown_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(Exception):
            run_finetune(Path(tmp), ["output.selection=best"])


if __name__ == "__main__":
    unittest.main()
