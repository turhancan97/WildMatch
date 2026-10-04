"""wildmatch demo: bundled data and Hydra overrides (no model download)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_module

from wildmatch import demo


class DemoTests(unittest.TestCase):
    def test_bundled_data_is_complete(self):
        root = demo.data_root()
        metadata = pd.read_csv(root / "metadata.csv")
        self.assertEqual(len(metadata), 24)
        self.assertEqual(metadata["identity"].nunique(), 6)
        self.assertEqual(metadata.groupby("split").size().to_dict(), {"database": 18, "query": 6})
        for path in metadata["path"]:
            self.assertTrue((root / path).is_file(), path)
        self.assertIn("CC BY 4.0", (root / "ATTRIBUTION.md").read_text())

    def test_overrides_compose_for_every_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method in demo.METHODS:
                with self.subTest(method), initialize_config_module(version_base="1.3", config_module="wildmatch.conf"):
                    cfg = compose(
                        config_name="probe",
                        overrides=["paths=default", *demo.overrides(method, "megadescriptor-t", Path(tmp))],
                    )
                self.assertEqual(cfg.model.device, "cpu")
                self.assertEqual(cfg.dataset.root, str(demo.data_root()))
                self.assertTrue(str(cfg.output.experiment_root).startswith(str(Path(tmp).resolve())))
                self.assertTrue(str(cfg.reporting.index_path).startswith(str(Path(tmp).resolve())))
                self.assertLessEqual(max(cfg.benchmark.top_k), cfg.benchmark.candidate_k)
                expected = "cosine" if method == "cosine" else "vismatch"
                self.assertEqual(cfg.benchmark.method, expected)


if __name__ == "__main__":
    unittest.main()
