import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

try:
    from wildmatch.evaluate.probe_runner import resolve_candidate_k, resolve_map_at_k

    HAS_PROBE_RUNNER = True
except ModuleNotFoundError:
    HAS_PROBE_RUNNER = False

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "src" / "wildmatch" / "conf"


def compose_config(name, overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


class HydraConfigurationTests(unittest.TestCase):
    def test_probe_and_finetune_defaults_compose(self):
        probe = compose_config("probe")
        finetune = compose_config("finetune")

        self.assertEqual(probe.model.type, "megadescriptor-l")
        self.assertEqual(probe.benchmark.method, "vismatch")
        self.assertEqual(probe.benchmark.methods.vismatch.matcher, "rdd-lightglue")
        self.assertEqual(probe.benchmark.candidate_k, 100)
        self.assertEqual(probe.benchmark.classifier_evaluation.open_set_policy, "open")
        self.assertFalse(probe.benchmark.classifier_evaluation.embedding_retrieval)
        # Disabled on purpose in the shipped probe config; see AGENTS.md.
        self.assertFalse(probe.safety_checks.enabled)
        self.assertNotIn("map_at_k", probe.benchmark)
        self.assertNotIn("B", probe.benchmark.methods.wildfusion)
        self.assertNotIn("candidate_k", probe.benchmark.methods.vismatch)
        self.assertNotIn("B", probe.benchmark.methods.local_lightglue)
        self.assertLessEqual(max(probe.benchmark.top_k), probe.benchmark.candidate_k)
        self.assertEqual(probe.benchmark.methods.vismatch.checkpoint_source, "default")
        self.assertEqual(probe.benchmark.methods.vismatch.checkpoint_components, "auto")
        self.assertEqual(probe.benchmark.methods.linear_probe.class_weighting, "inverse_frequency")
        self.assertTrue(probe.benchmark.methods.linear_probe.class_weight_normalize)
        self.assertEqual(probe.benchmark.methods.linear_probe.class_weight_max, 5.0)
        self.assertEqual(probe.benchmark.methods.efficient_probe.class_weighting, "inverse_frequency")
        self.assertTrue(probe.benchmark.methods.efficient_probe.class_weight_normalize)
        self.assertEqual(probe.benchmark.methods.efficient_probe.class_weight_max, 5.0)
        self.assertEqual(probe.dataset.image_variant, "no_background")
        self.assertEqual(finetune.model.type, "megadescriptor-l")
        self.assertEqual(finetune.dataset.image_variant, "background")
        self.assertEqual(finetune.train.epochs, 30)

    def test_configs_load_as_package_resources(self):
        # The entry points live in wildmatch/entrypoints.py, so Hydra resolves the configs as
        # pkg://wildmatch.conf; this needs conf/ to be an importable package.
        from hydra import initialize_config_module

        for name, job in (("probe", "probe"), ("finetune", "finetune")):
            with initialize_config_module(version_base="1.3", config_module="wildmatch.conf"):
                cfg = compose(config_name=name, return_hydra_config=True)
            self.assertEqual(cfg.model.type, "megadescriptor-l")
            self.assertEqual(cfg.hydra.job.name, job)

    def test_frozen_config_snapshot_wins_with_config_path(self):
        # Sweep tasks pass the submission's frozen probe.yaml with --config-path. With
        # --config-dir Hydra keeps the packaged probe.yaml as primary and ignores the snapshot.
        with TemporaryDirectory() as tmp:
            snapshot = (
                (CONF_DIR / "probe.yaml").read_text(encoding="utf-8").replace("cache/features", "cache/FROM_SNAPSHOT")
            )
            Path(tmp, "probe.yaml").write_text(snapshot, encoding="utf-8")
            for group in ("paths", "dataset"):  # a submission snapshot freezes the config groups too
                shutil.copytree(CONF_DIR / group, Path(tmp, group))
            outputs = {}
            for flag in ("--config-path", "--config-dir"):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "wildmatch",
                        "evaluate",
                        flag,
                        tmp,
                        "--config-name",
                        "probe",
                        "--cfg",
                        "job",
                    ],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                outputs[flag] = result.stdout
        self.assertIn("cache/FROM_SNAPSHOT", outputs["--config-path"])
        self.assertNotIn("cache/FROM_SNAPSHOT", outputs["--config-dir"])  # documents the trap

    def test_hydra_job_log_stays_out_of_repository_root(self):
        for name in ("probe", "finetune"):
            with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
                cfg = compose(config_name=name, return_hydra_config=True)
            hydra_cfg = cfg.hydra
            self.assertEqual(
                OmegaConf.to_container(hydra_cfg.run, resolve=False)["dir"],
                "${hydra:runtime.cwd}/logs/hydra",
                name,
            )
            self.assertFalse(hydra_cfg.job.chdir, name)
            self.assertIsNone(hydra_cfg.output_subdir, name)

    def test_nested_dotlist_overrides_convert_types(self):
        cfg = compose_config(
            "probe",
            [
                "benchmark.method=vismatch",
                "benchmark.methods.vismatch.matcher=loma",
                "benchmark.methods.vismatch.top_k=1024",
                "benchmark.methods.vismatch.oom_backoff=false",
                "benchmark.top_k=[1,10]",
                "dataset.metadata_file=custom.csv",
            ],
        )

        self.assertEqual(cfg.benchmark.method, "vismatch")
        self.assertEqual(cfg.benchmark.methods.vismatch.matcher, "loma")
        self.assertEqual(cfg.benchmark.methods.vismatch.top_k, 1024)
        self.assertFalse(cfg.benchmark.methods.vismatch.oom_backoff)
        self.assertEqual(list(cfg.benchmark.top_k), [1, 10])
        self.assertEqual(cfg.dataset.metadata_file, "custom.csv")

    def test_linear_probe_weighting_override(self):
        cfg = compose_config(
            "probe",
            [
                "benchmark.method=linear_probe",
                "benchmark.methods.linear_probe.class_weighting=none",
                "benchmark.methods.linear_probe.class_weight_max=3.0",
            ],
        )
        self.assertEqual(cfg.benchmark.methods.linear_probe.class_weighting, "none")
        self.assertEqual(cfg.benchmark.methods.linear_probe.class_weight_max, 3.0)

    def test_efficient_probe_weighting_override(self):
        cfg = compose_config(
            "probe",
            [
                "benchmark.method=efficient_probe",
                "benchmark.methods.efficient_probe.class_weighting=none",
                "benchmark.methods.efficient_probe.class_weight_max=3.0",
            ],
        )
        self.assertEqual(cfg.benchmark.methods.efficient_probe.class_weighting, "none")
        self.assertEqual(cfg.benchmark.methods.efficient_probe.class_weight_max, 3.0)

    def test_classifier_evaluation_override_converts_types(self):
        cfg = compose_config(
            "probe",
            [
                "benchmark.classifier_evaluation.open_set_policy=closed",
                "benchmark.classifier_evaluation.embedding_retrieval=true",
            ],
        )
        self.assertEqual(cfg.benchmark.classifier_evaluation.open_set_policy, "closed")
        self.assertTrue(cfg.benchmark.classifier_evaluation.embedding_retrieval)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(ConfigCompositionException):
            compose_config("probe", ["benchmark.methd=vismatch"])

    def test_output_paths_and_interpolations_remain_project_managed(self):
        probe = compose_config("probe")
        finetune = compose_config("finetune")

        self.assertEqual(probe.output.experiment_root, "experiments")
        self.assertEqual(probe.output.run_dir, "benchmark_runs")
        self.assertEqual(probe.output.csv_path, "benchmark_runs/benchmark_results.csv")
        self.assertEqual(finetune.output.experiment_root, "experiments")
        self.assertEqual(finetune.output.run_dir, "results/CzechLynx_v2/CzechLynx/mask_False")
        self.assertTrue(finetune.reporting.enabled)
        self.assertNotIn("${", str(probe.dataset.root))
        self.assertNotIn("${", str(finetune.output.csv_path))

    def test_removed_budget_overrides_are_rejected(self):
        for override in (
            "benchmark.map_at_k=50",
            "benchmark.methods.vismatch.candidate_k=50",
            "benchmark.methods.wildfusion.B=50",
            "benchmark.methods.local_lightglue.B=50",
        ):
            with self.subTest(override=override), self.assertRaises(ConfigCompositionException):
                compose_config("probe", [override])

    @unittest.skipUnless(HAS_PROBE_RUNNER, "probe runtime dependencies not available")
    def test_shared_budget_drives_candidate_and_map_cutoffs(self):
        cfg = compose_config("probe", ["benchmark.candidate_k=200"])
        self.assertEqual(resolve_candidate_k(cfg), 200)
        self.assertEqual(resolve_candidate_k(cfg, database_size=75), 75)
        self.assertEqual(resolve_map_at_k(cfg), 200)

    @unittest.skipUnless(HAS_PROBE_RUNNER, "probe runtime dependencies not available")
    def test_shared_budget_rejects_non_positive_values(self):
        for value in (0, -1):
            with self.subTest(value=value):
                cfg = compose_config("probe", [f"benchmark.candidate_k={value}"])
                with self.assertRaisesRegex(ValueError, "benchmark.candidate_k must be > 0"):
                    resolve_candidate_k(cfg)

    def test_legacy_cli_budget_override_has_migration_error(self):
        from wildmatch.entrypoints import reject_removed_probe_budget_overrides

        with self.assertRaisesRegex(ValueError, "benchmark.candidate_k"):
            reject_removed_probe_budget_overrides(["benchmark.map_at_k=50"])

    def test_resolved_snapshot_contains_overrides(self):
        cfg = compose_config(
            "probe",
            ["benchmark.method=vismatch", "benchmark.methods.vismatch.matcher=loma"],
        )
        with TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "config.snapshot.yaml"
            OmegaConf.save(cfg, snapshot, resolve=True)
            saved = OmegaConf.load(snapshot)
            snapshot_text = snapshot.read_text(encoding="utf-8")

        self.assertEqual(saved.benchmark.method, "vismatch")
        self.assertEqual(saved.benchmark.methods.vismatch.matcher, "loma")
        self.assertNotIn("${", snapshot_text)


if __name__ == "__main__":
    unittest.main()
