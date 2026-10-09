"""Static checks of every packaged YAML configuration (no data, no GPU, no network).

Composition of `probe` with every dataset entry, every packaged sweep spec and the weights manifest
against the registry are covered in test_paths_registry, test_sweep and test_weights; this file adds
the checks that apply to all files: each YAML parses, the weights manifest is well formed, and the
matcher fine-tuning config composes with both recipes.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path, PurePosixPath

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from wildmatch.data.registry import dataset_keys

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "src" / "wildmatch" / "conf"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConfFilesTest(unittest.TestCase):
    def test_every_yaml_parses(self):
        files = sorted(CONF_DIR.rglob("*.yaml"))
        self.assertGreater(len(files), 30)
        for path in files:
            with self.subTest(path=str(path.relative_to(CONF_DIR))):
                config = OmegaConf.load(path)
                self.assertIsNotNone(config)

    def test_weights_manifest_is_well_formed(self):
        manifest = OmegaConf.to_container(OmegaConf.load(CONF_DIR / "weights.yaml"), resolve=False)
        self.assertEqual(set(manifest), {"repo_id", "revision", "entries"})
        known = set(dataset_keys())
        hub_names, pairs = [], []
        for entry in manifest["entries"]:
            with self.subTest(entry=f"{entry.get('dataset')}/{entry.get('matcher')}"):
                self.assertIn(set(entry), ({"dataset", "matcher", "files"}, {"dataset", "matcher", "set", "files"}))
                self.assertIn(entry.get("set", "default"), {"default", "paper"})
                self.assertIn(entry["dataset"], known)
                self.assertIn(entry["matcher"], {"loma", "rdd-lightglue"})
                pairs.append((entry["dataset"], entry["matcher"], entry.get("set", "default")))
                self.assertTrue(entry["files"])
                for item in entry["files"]:
                    self.assertEqual(set(item), {"hub", "local", "sha256"})
                    self.assertRegex(item["sha256"], SHA256)
                    for key in ("hub", "local"):
                        path = PurePosixPath(item[key])
                        self.assertFalse(path.is_absolute(), item[key])
                        self.assertNotIn("..", path.parts, item[key])
                    hub_names.append(item["hub"])
        self.assertEqual(len(hub_names), len(set(hub_names)), "duplicate Hub file names")
        self.assertEqual(len(pairs), len(set(pairs)), "duplicate dataset/matcher/set entries")

    def test_finetune_matcher_composes_with_both_recipes(self):
        for recipe in ("loma", "rdd"):
            for dataset in ("czechlynx_closed", "salamander"):
                with self.subTest(recipe=recipe, dataset=dataset):
                    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
                        cfg = compose(
                            config_name="finetune_matcher",
                            overrides=["paths=default", f"dataset={dataset}", f"matcher_finetune={recipe}"],
                        )
                    self.assertEqual(cfg.matcher_finetune.matcher, recipe)
                    self.assertEqual(cfg.dataset.registry.key, dataset)


if __name__ == "__main__":
    unittest.main()
