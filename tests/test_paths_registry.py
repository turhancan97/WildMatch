"""Phase 2 of the package refactor: path profiles, the dataset registry and their consumers."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from wildmatch import paths as P
from wildmatch.data import registry as R
from wildmatch.entrypoints import apply_paths_profile

ROOT = Path(__file__).resolve().parents[1]
GMUM_DATA = "/shared/sets/datasets/vision/czechlynx"
GMUM_CACHE = "/shared/results/common/kargin/lynx/results"

# PAPER_PROFILES / BENCHMARK_ONLY_PROFILES exactly as the hard-coded module defined them
# before the registry existed (captured 2026-10-04); under paths=gmum they must not change.
PROFILES_BEFORE_REGISTRY = {'paper': [{'key': 'nyala',
            'label': 'Nyala',
            'source': 'NyalaData',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_no_background/metadata_NyalaData.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'NyalaData',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'beluga',
            'label': 'Beluga',
            'source': 'BelugaID',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_no_background/metadata_BelugaID.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'BelugaID',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'hyena',
            'label': 'Hyena',
            'source': 'HyenaID2022',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_mdsplit_no_background/metadata_HyenaID2022.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'HyenaID2022',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'leopard',
            'label': 'Leopard',
            'source': 'LeopardID2022',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_mdsplit_no_background/metadata_LeopardID2022.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'LeopardID2022',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'sea_star',
            'label': 'Sea Star',
            'source': 'SeaStarReID2023',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_mdsplit_no_background/metadata_SeaStarReID2023.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'SeaStarReID2023',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'whale_shark',
            'label': 'Whale Shark',
            'source': 'WhaleSharkID',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_no_background/metadata_WhaleSharkID.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'WhaleSharkID',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'turtle',
            'label': 'Turtle',
            'source': 'ZindiTurtleRecall',
            'metadata': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k/metadata_no_background/metadata_ZindiTurtleRecall.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'WildlifeReID-10k',
            'animal': 'ZindiTurtleRecall',
            'root': '/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k',
            'mask_col': None},
           {'key': 'salamander',
            'label': 'Salamander',
            'source': 'SalamanderID2025',
            'metadata': '/shared/sets/datasets/vision/czechlynx/SalamanderID2025/split_time_closed_no_background.csv',
            'identity_col': 'identity',
            'split_col': 'split',
            'database_value': 'database',
            'query_value': 'query',
            'dataset_name': 'SalamanderID2025',
            'animal': 'SalamanderID2025',
            'root': '/shared/sets/datasets/vision/czechlynx/SalamanderID2025',
            'mask_col': None},
           {'key': 'lynx_closed',
            'label': 'Lynx (closed)',
            'source': 'CzechLynx v2',
            'metadata': '/shared/sets/datasets/vision/czechlynx/CzechLynx_v2/CzechLynxDataset-Metadata-Real.csv',
            'identity_col': 'unique_name',
            'split_col': 'split-time_closed',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'CzechLynx_v2',
            'animal': 'CzechLynx',
            'root': '/shared/sets/datasets/vision/czechlynx/CzechLynx_v2',
            'mask_col': 'mask'},
           {'key': 'lynx_open',
            'label': 'Lynx (open)',
            'source': 'CzechLynx v2',
            'metadata': '/shared/sets/datasets/vision/czechlynx/CzechLynx_v2/CzechLynxDataset-Metadata-Real.csv',
            'identity_col': 'unique_name',
            'split_col': 'split-time_open',
            'database_value': 'train',
            'query_value': 'test',
            'dataset_name': 'CzechLynx_v2',
            'animal': 'CzechLynx',
            'root': '/shared/sets/datasets/vision/czechlynx/CzechLynx_v2',
            'mask_col': 'mask'}],
 'benchmark_only': [{'key': 'jaguar',
                     'label': 'Jaguar',
                     'source': 'Kaggle Jaguar Re-ID',
                     'metadata': '/shared/sets/datasets/vision/czechlynx/jaguar/jaguar_reid_v2_no_background.csv',
                     'identity_col': 'identity',
                     'split_col': 'split_v2',
                     'database_value': 'database',
                     'query_value': 'query',
                     'dataset_name': 'JaguarReID',
                     'animal': 'JaguarReID',
                     'root': '/shared/sets/datasets/vision/czechlynx/jaguar',
                     'mask_col': None}]}


# The dataset profiles of the bash launchers removed on 2026-10-04 (probe-parallel-wildlife.sh and
# probe-parallel-czechlynx.sh at d295f41): name|animal|root|metadata|label|mask|no_background|
# image_variant|split|database|query|calibration. The registry must keep these values.
LAUNCHER_PROFILES = {
    "atrw": "WildlifeReID-10k|ATRW|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_ATRW.csv|identity|mask|false|no_background|split|train|test|100",
    "beluga": "WildlifeReID-10k|BelugaID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_BelugaID.csv|identity|mask|false|no_background|split|train|test|100",
    "cowdataset": "WildlifeReID-10k|CowDataset|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_CowDataset.csv|identity|mask|false|no_background|split|train|test|100",
    "czechlynx_closed": "CzechLynx_v2|CzechLynx|/shared/sets/datasets/vision/czechlynx/CzechLynx_v2|CzechLynxDataset-Metadata-Real.csv|unique_name|mask|true|no_background|split-time_closed|train|test|100",
    "czechlynx_open": "CzechLynx_v2|CzechLynx|/shared/sets/datasets/vision/czechlynx/CzechLynx_v2|CzechLynxDataset-Metadata-Real.csv|unique_name|mask|true|no_background|split-time_open|train|test|100",
    "czechlynx_unseen_eval": "CzechLynx_v2|CzechLynx|/shared/sets/datasets/vision/czechlynx/CzechLynx_v2|${CZECHLYNX_UNSEEN_EVAL_METADATA_FILE}|unique_name|mask|true|no_background|unseen_eval_split|database|query|100",
    "giraffes": "WildlifeReID-10k|Giraffes|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_Giraffes.csv|identity|mask|false|no_background|split|train|test|100",
    "giraffezebraid": "WildlifeReID-10k|GiraffeZebraID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_GiraffeZebraID.csv|identity|mask|false|no_background|split|train|test|100",
    "hyenaid2022": "WildlifeReID-10k|HyenaID2022|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_HyenaID2022.csv|identity|mask|false|no_background|split|train|test|100",
    "jaguar": "JaguarReID|JaguarReID|/shared/sets/datasets/vision/czechlynx/jaguar|jaguar_reid_v2_no_background.csv|identity|mask|false|no_background|split_v2|database|query|100",
    "leopardid2022": "WildlifeReID-10k|LeopardID2022|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_LeopardID2022.csv|identity|mask|false|no_background|split|train|test|100",
    "nyala": "WildlifeReID-10k|NyalaData|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_NyalaData.csv|identity|mask|false|no_background|split|train|test|100",
    "salamander": "SalamanderID2025|SalamanderID2025|/shared/sets/datasets/vision/czechlynx/SalamanderID2025|split_time_closed_no_background.csv|identity|mask|false|no_background|split|database|query|100",
    "seastarreid2023": "WildlifeReID-10k|SeaStarReID2023|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_SeaStarReID2023.csv|identity|mask|false|no_background|split|train|test|100",
    "stripespotter": "WildlifeReID-10k|StripeSpotter|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_mdsplit_no_background/metadata_StripeSpotter.csv|identity|mask|false|no_background|split|train|test|100",
    "whaleshark": "WildlifeReID-10k|WhaleSharkID|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_WhaleSharkID.csv|identity|mask|false|no_background|split|train|test|100",
    "zindi": "WildlifeReID-10k|ZindiTurtleRecall|/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k|metadata_no_background/metadata_ZindiTurtleRecall.csv|identity|mask|false|no_background|split|train|test|100",
}
# WildlifeReID-10k launcher checkpoint folders (LoMa, RDD) under <animal>/{loma,rdd}-finetuned/.
LAUNCHER_CHECKPOINT_DIRS = {
    "atrw": ('legacy', 'legacy'),
    "beluga": ('legacy', 'legacy'),
    "cowdataset": ('legacy', 'legacy'),
    "giraffes": ('legacy', 'legacy'),
    "giraffezebraid": ('legacy', 'legacy'),
    "hyenaid2022": ('legacy', 'legacy'),
    "jaguar": ('legacy-loma-mined', 'legacy-loma-mined'),
    "leopardid2022": ('legacy', 'legacy'),
    "nyala": ('legacy', 'legacy'),
    "salamander": ('legacy-loma-mined', 'legacy-rdd-mined'),
    "seastarreid2023": ('legacy', 'legacy'),
    "stripespotter": ('legacy', 'legacy'),
    "whaleshark": ('legacy', 'legacy'),
    "zindi": ('legacy', 'legacy'),
}


class PathProfileTests(unittest.TestCase):
    def test_profiles_and_resolution(self):
        self.assertEqual(P.available_profiles(), ["default", "gmum"])
        gmum = P.load_paths("gmum")
        self.assertEqual(gmum.data_root, GMUM_DATA)
        self.assertEqual(gmum.cache_root, GMUM_CACHE)
        with mock.patch.dict(os.environ, {}, clear=False):
            for var in ("WILDMATCH_DATA_ROOT", "WILDMATCH_PAPER_REPO"):
                os.environ.pop(var, None)
            default = P.load_paths("default")
            self.assertEqual(default.data_root, "data")
            self.assertIsNone(default.external.paper_repo)
        with mock.patch.dict(os.environ, {"WILDMATCH_DATA_ROOT": "/elsewhere"}):
            self.assertEqual(P.load_paths("default").data_root, "/elsewhere")
        with self.assertRaises(ValueError):
            P.load_paths("nope")

    def test_profile_choice_order(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WILDMATCH_PATHS", None)
            self.assertEqual(P.active_profile(cwd=Path(tmp)), "default")
            Path(tmp, P.LOCAL_SETTINGS).write_text("paths: gmum\n", encoding="utf-8")
            self.assertEqual(P.active_profile(cwd=Path(tmp)), "gmum")
            os.environ["WILDMATCH_PATHS"] = "default"
            self.assertEqual(P.active_profile(cwd=Path(tmp)), "default")
            self.assertEqual(P.active_profile("gmum", cwd=Path(tmp)), "gmum")

    def test_entry_points_add_the_profile_only_when_free(self):
        with mock.patch.dict(os.environ, {"WILDMATCH_PATHS": "gmum"}):
            argv = ["probe.py", "dataset=salamander"]
            self.assertEqual(apply_paths_profile(argv), "gmum")
            self.assertEqual(argv[-1], "paths=gmum")
            for given in (["probe.py", "paths=default"], ["probe.py", "--config-path", "/x"], ["probe.py", "-cp", "/x"]):
                argv = list(given)
                self.assertIsNone(apply_paths_profile(argv))
                self.assertEqual(argv, given)
        with mock.patch.dict(os.environ, {"WILDMATCH_PATHS": "default"}):
            argv = ["probe.py"]
            self.assertIsNone(apply_paths_profile(argv))


class RegistryTests(unittest.TestCase):
    def test_registry_covers_every_launcher_profile_with_identical_values(self):
        self.assertEqual(sorted(LAUNCHER_PROFILES), R.dataset_keys())
        fields = ("name", "animal", "root", "metadata_file", "label_col", "mask_col", "no_background",
                  "image_variant", "split_col", "database_split_value", "query_split_value", "calibration_size")
        for key, line in LAUNCHER_PROFILES.items():
            entry = R.load_dataset(key, "gmum")
            for field, expected in zip(fields, line.split("|")):
                if key == "czechlynx_unseen_eval" and field == "metadata_file":
                    continue  # the launcher took it from CZECHLYNX_UNSEEN_EVAL_METADATA_FILE
                actual = entry[field]
                if isinstance(actual, bool):
                    expected = expected == "true"
                elif isinstance(actual, int):
                    expected = int(expected)
                self.assertEqual(actual, expected, f"{key}.{field}")
            self.assertEqual(entry.registry.key, key)
            if key in LAUNCHER_CHECKPOINT_DIRS:
                loma_dir, rdd_dir = LAUNCHER_CHECKPOINT_DIRS[key]
                base = f"{GMUM_DATA}/checkpoints/wildlife-reid-10k/{entry.animal}"
                custom = entry.registry.checkpoints.custom
                self.assertEqual(custom.loma, f"{base}/loma-finetuned/{loma_dir}/epoch_299/model.safetensors")
                self.assertEqual(custom["rdd-lightglue"], f"{base}/rdd-finetuned/{rdd_dir}/epoch_299/model.safetensors")

    def test_czechlynx_checkpoints_match_the_launcher_defaults(self):
        closed = f"{GMUM_DATA}/checkpoints/czechlynx-time-closed"
        open_ = f"{GMUM_DATA}/checkpoints/czechlynx-time-open"
        expected = {
            "czechlynx_closed": {
                "custom": {"loma": f"{closed}/loma-b-finetuned-loma-mined-legacy/epoch_299/model.safetensors",
                           "rdd-lightglue": f"{closed}/rdd-finetuned-loma-mined-legacy/epoch_299/model.safetensors"},
                "descriptor-fine-tuned": {
                    "loma": f"{closed}/loma-b-descriptor-finetuned-legacy/epoch_252/model.safetensors",
                    "rdd-lightglue": f"{closed}/rdd-descriptor-finetuned-loma-mined-legacy/epoch_175/model.safetensors"},
                "joint-fine-tuned": {
                    "loma": f"{closed}/loma-b-joint-finetuned-loma-mined-legacy/epoch_299/model.safetensors",
                    "rdd-lightglue": f"{closed}/rdd-joint-finetuned-loma-mined-legacy/epoch_299"},
            },
        }
        for key in ("czechlynx_open", "czechlynx_unseen_eval"):
            expected[key] = {"custom": {"loma": f"{open_}/loma-b-finetuned-loma-mined-legacy/epoch_299/model.safetensors",
                                        "rdd-lightglue": f"{open_}/rdd-finetuned-loma-mined-legacy/epoch_299/model.safetensors"}}
        for key, checkpoints in expected.items():
            self.assertEqual(OmegaConf.to_container(R.load_dataset(key, "gmum").registry.checkpoints), checkpoints, key)

    def test_profiles_under_gmum_are_unchanged(self):
        code = ("import json\nfrom wildmatch.reporting.paper_datasets import PAPER_PROFILES, BENCHMARK_ONLY_PROFILES\n"
                "d = lambda ps: [{k: (str(v) if v is not None else None) for k, v in p._asdict().items()} for p in ps]\n"
                "print(json.dumps({'paper': d(PAPER_PROFILES), 'benchmark_only': d(BENCHMARK_ONLY_PROFILES)}))")
        env = dict(os.environ, WILDMATCH_PATHS="gmum")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout
        self.assertEqual(json.loads(out), PROFILES_BEFORE_REGISTRY)

    def test_hydra_composition_matches_the_registry(self):
        for key in R.dataset_keys():
            with initialize_config_module(version_base="1.3", config_module="wildmatch.conf"):
                cfg = compose(config_name="probe", overrides=["paths=gmum", f"dataset={key}"])
            OmegaConf.resolve(cfg)
            self.assertEqual(OmegaConf.to_container(cfg.dataset), OmegaConf.to_container(R.load_dataset(key, "gmum")), key)
        self.assertEqual(cfg.benchmark.cache.dir, f"{GMUM_CACHE}/{cfg.dataset.name}/{cfg.dataset.animal}/cache/features")
        self.assertEqual(cfg.benchmark.methods.vismatch.cache_dir, f"{GMUM_CACHE}/{cfg.dataset.name}/{cfg.dataset.animal}/cache/vismatch")
        with initialize_config_module(version_base="1.3", config_module="wildmatch.conf"):
            default = compose(config_name="probe", overrides=["paths=default"])
        self.assertEqual(default.dataset.name, "CzechLynx_v2")
        self.assertFalse(str(OmegaConf.to_container(default, resolve=True)["dataset"]["root"]).startswith("/shared"))


class SubmissionSnapshotTests(unittest.TestCase):
    def test_snapshot_freezes_the_config_groups(self):
        from wildmatch.sweep import manifest as M

        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp, "custom")
            custom.mkdir()
            (custom / "probe.yaml").write_text((ROOT / "src/wildmatch/conf/probe.yaml").read_text(encoding="utf-8"))
            snapshot = Path(tmp, "snapshot")
            snapshot.mkdir()
            M._snapshot_config_groups(custom / "probe.yaml", snapshot)  # groups come from the package
            (snapshot / "probe.yaml").write_text((custom / "probe.yaml").read_text())
            self.assertEqual(sorted(p.name for p in (snapshot / "dataset").glob("*.yaml")),
                             [f"{k}.yaml" for k in R.dataset_keys()])
            digest = M.config_tree_sha256(snapshot)
            (snapshot / "dataset" / "salamander.yaml").write_text("name: changed\n")
            self.assertNotEqual(M.config_tree_sha256(snapshot), digest)


class LegacyRunReadingTests(unittest.TestCase):
    def test_pre_refactor_run_manifest_is_still_read(self):
        from wildmatch.reporting.paper_tables import discover_records

        records = discover_records(ROOT / "tests" / "fixtures" / "legacy_experiments")
        self.assertEqual(len(records), 1)
        record = records[0]
        metrics = json.loads(next((ROOT / "tests" / "fixtures").rglob("metrics.json")).read_text())
        self.assertEqual(record["animal"], "SalamanderID2025")
        self.assertAlmostEqual(float(record["top_1"]), metrics["top_1"])


if __name__ == "__main__":
    unittest.main()
