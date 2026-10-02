import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "probe-parallel-wildlife.sh"
CZECH_SCRIPT = ROOT / "probe-parallel-czechlynx.sh"


class ParallelProbeLauncherTests(unittest.TestCase):
    def run_script(self, *args, env=None):
        merged_env = os.environ.copy()
        merged_env.pop("SLURM_ARRAY_TASK_ID", None)
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=ROOT,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def checkpoint_env(self, loma, rdd):
        return {
            "LOMA_CUSTOM_CHECKPOINT_PATH": str(loma),
            "RDD_CUSTOM_CHECKPOINT_PATH": str(rdd),
        }

    def run_czech_script(self, *args, env=None):
        merged_env = os.environ.copy()
        merged_env.pop("SLURM_ARRAY_TASK_ID", None)
        if env:
            merged_env.update(env)
        return subprocess.run(
            ["bash", str(CZECH_SCRIPT), *args],
            cwd=ROOT,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    def task_rows(self):
        result = self.run_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        return [dict(item.split("=", 1) for item in line.split()) for line in lines]

    def test_shell_syntax_is_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --output=logs/parallel_run/%x_%A_%a.out", text)
        self.assertIn("#SBATCH --error=logs/parallel_run/%x_%A_%a.err", text)
        self.assertIn('SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"', text)

    def test_new_wildlife_profiles_are_ready_to_activate(self):
        script_text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("|CzechLynx_v2|CzechLynx|", script_text)
        expected_profiles = {
            "ATRW": ("metadata_ATRW.csv", "299", "299"),
            "Giraffes": ("metadata_Giraffes.csv", "299", "299"),
            "LeopardID2022": ("metadata_LeopardID2022.csv", "299", "299"),
            "HyenaID2022": ("metadata_HyenaID2022.csv", "299", "299"),
            "GiraffeZebraID": ("metadata_GiraffeZebraID.csv", "299", "299"),
            "CowDataset": ("metadata_CowDataset.csv", "299", "299"),
            "StripeSpotter": ("metadata_StripeSpotter.csv", "299", "299"),
            "SeaStarReID2023": ("metadata_SeaStarReID2023.csv", "299", "299"),
        }
        for animal, (metadata_name, loma_epoch, rdd_epoch) in expected_profiles.items():
            self.assertIn(f"|WildlifeReID-10k|{animal}|", script_text)
            self.assertIn(
                f"metadata_mdsplit_no_background/{metadata_name}",
                script_text,
            )
            self.assertIn(
                f"|100|legacy|legacy|{loma_epoch}|{rdd_epoch}\"",
                script_text,
            )

    def salamander_launcher(self, temp_root: Path) -> Path:
        """Temporary launcher copy with only the Salamander profile active.

        The editable launcher's active profile, variants and k grid change from
        run to run, so the copy pins a small known grid instead.
        """
        text = SCRIPT.read_text(encoding="utf-8")
        profiles = re.search(r"^DATASET_PROFILES=\(\n.*?^\)", text, flags=re.M | re.S).group(0)
        pinned = re.sub(r'^    "', '    # "', profiles, flags=re.M)
        pinned = pinned.replace('    # "salamander|', '    "salamander|', 1)
        text = text.replace(profiles, pinned, 1)
        text = re.sub(
            r"^VARIANTS=\(\n.*?^\)",
            'VARIANTS=(\n'
            '    "cosine|-|default|-|-|-|-"\n'
            '    "vismatch|loma|custom|${LOMA_CUSTOM_CHECKPOINT_PATH}|matcher_only|-|-"\n'
            '    "vismatch|rdd-lightglue|custom|${RDD_CUSTOM_CHECKPOINT_PATH}|matcher_only|-|-"\n'
            ')',
            text, count=1, flags=re.M | re.S,
        )
        text = re.sub(r"^CANDIDATE_K_VALUES=\(.*\)$", "CANDIDATE_K_VALUES=(10 50)", text, count=1, flags=re.M)
        copy = temp_root / "probe-parallel-wildlife.sh"
        copy.write_text(text, encoding="utf-8")
        return copy

    def test_salamander_profile_template(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"salamander|SalamanderID2025|SalamanderID2025|/shared/sets/datasets/vision/czechlynx/SalamanderID2025|'
            'split_time_closed_no_background.csv|identity|mask|false|no_background|split|database|query|100|'
            'legacy-loma-mined|legacy-rdd-mined|299|299"',
            text,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            launcher = self.salamander_launcher(temp_root)
            loma = temp_root / "loma.safetensors"
            rdd = temp_root / "rdd.safetensors"
            loma.write_bytes(b"loma")
            rdd.write_bytes(b"rdd")
            env = {**os.environ, **self.checkpoint_env(loma, rdd)}
            env.pop("SLURM_ARRAY_TASK_ID", None)
            listed = subprocess.run(
                ["bash", str(launcher), "--list-tasks"], cwd=temp_root, env=env,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(listed.returncode, 0, listed.stderr)
            rows = [dict(item.split("=", 1) for item in line.split())
                    for line in listed.stdout.splitlines() if line.startswith("index=")]
            self.assertEqual(len(rows), 6)  # 3 variants x k in {10, 50}
            self.assertEqual({(row["profile"], row["dataset"], row["evaluation_animal"]) for row in rows},
                             {("salamander", "SalamanderID2025", "SalamanderID2025")})
            self.assertEqual({row["checkpoint_owner"] for row in rows if row["checkpoint"] == "custom"},
                             {"SalamanderID2025"})
            custom = next(row for row in rows if row["matcher"] == "loma")
            dry = subprocess.run(
                ["bash", str(launcher), "--dry-run"], cwd=temp_root,
                env={**env, "SLURM_ARRAY_TASK_ID": custom["index"], "PROBE_PARALLEL_DRY_RUN": "1"},
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(dry.returncode, 0, dry.stderr)
        for override in (
            "dataset.name=SalamanderID2025",
            "dataset.animal=SalamanderID2025",
            "dataset.root=/shared/sets/datasets/vision/czechlynx/SalamanderID2025",
            "dataset.metadata_file=split_time_closed_no_background.csv",
            "dataset.no_background=false",
            "dataset.image_variant=no_background",
            "dataset.split_col=split",
            "dataset.database_split_value=database",
            "dataset.query_split_value=query",
            "checkpoint_components=matcher_only",
        ):
            self.assertIn(override, dry.stdout)

    def test_czechlynx_joint_rows_use_full_components(self):
        text = CZECH_SCRIPT.read_text(encoding="utf-8")
        for matcher, variable in (("loma", "CZECHLYNX_CLOSED_JOINT_LOMA_CHECKPOINT"),
                                  ("rdd-lightglue", "CZECHLYNX_CLOSED_JOINT_RDD_CHECKPOINT")):
            self.assertIn(f'"vismatch|{matcher}|joint-fine-tuned|${{{variable}}}|full|-|-"', text)
        self.assertIn("rdd-joint-finetuned-loma-mined-legacy/epoch_", text)
        self.assertIn("loma-b-joint-finetuned-loma-mined-legacy/epoch_", text)

    def test_slurm_submission_directory_is_used(self):
        with tempfile.TemporaryDirectory() as submit_dir:
            result = self.run_script("--list-tasks", env={"SLURM_SUBMIT_DIR": submit_dir})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(submit_dir) / "logs" / "parallel_run").is_dir())

    def test_task_table_has_expected_grid(self):
        result = self.run_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        self.assertGreater(len(lines), 0)

        parsed = []
        for line in lines:
            fields = dict(item.split("=", 1) for item in line.split())
            parsed.append(fields)

        self.assertEqual([int(row["index"]) for row in parsed], list(range(len(parsed))))
        script_text = SCRIPT.read_text(encoding="utf-8")
        candidate_matches = re.findall(r"(?m)^(?!\s*#)\s*CANDIDATE_K_VALUES=\(([^)]*)\)", script_text)
        self.assertTrue(candidate_matches)
        expected_candidates = sorted(int(value) for value in candidate_matches[-1].split())
        self.assertEqual(sorted({int(row["candidate_k"]) for row in parsed}), expected_candidates)
        methods = {row["method"] for row in parsed}
        self.assertTrue(methods)
        self.assertTrue(methods.issubset({"cosine", "wildfusion", "local_lightglue", "linear_probe", "efficient_probe", "vismatch"}))
        self.assertTrue(all(row["matcher"] in {"-", "loma", "rdd-lightglue"} for row in parsed))
        # Linear-probe rows are opt-in in the editable table; validate any active ones.
        linear_modes = {row["train_mode"] for row in parsed if row["method"] == "linear_probe"}
        self.assertTrue(linear_modes.issubset({"classifier", "partial", "all"}))
        self.assertTrue({row["class_weighting"] for row in parsed}.issubset({"-", "weighted", "unweighted"}))
        variant_text = SCRIPT.read_text(encoding="utf-8")
        for mode in ("classifier", "partial", "all"):
            self.assertIn(f'linear_probe|-|default|-|{mode}', variant_text)
            self.assertIn(f'linear_probe|-|default|-|{mode}|weighted', variant_text)
            self.assertIn(f'linear_probe|-|default|-|{mode}|unweighted', variant_text)
            self.assertIn(f'efficient_probe|-|default|-|{mode}|weighted', variant_text)
            self.assertIn(f'efficient_probe|-|default|-|{mode}|unweighted', variant_text)
        self.assertEqual(
            len({(row["candidate_k"], row["method"], row["matcher"], row["checkpoint"], row["train_mode"], row["class_weighting"]) for row in parsed}),
            len(parsed),
        )

    def test_submission_dry_run_uses_array_range_and_cap(self):
        task_count = len(self.task_rows())
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            result = self.run_script(
                "--dry-run",
                env={**self.checkpoint_env(loma.name, rdd.name), "MAX_CONCURRENT_JOBS": "7"},
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Submitting {task_count} probe tasks with array throttle 7", result.stdout)
        self.assertRegex(
            result.stdout,
            re.compile(rf"sbatch --array=0-{task_count - 1}%7 .*probe-parallel-wildlife\.sh"),
        )

    def test_linear_probe_modes_are_explicit_and_non_redundant(self):
        rows = [row for row in self.task_rows() if row["method"] == "linear_probe"]
        self.assertTrue({row["train_mode"] for row in rows}.issubset({"classifier", "partial", "all"}))
        if not rows:
            self.skipTest("linear-probe rows are opt-in in the launcher")
        self.assertEqual(len({row["candidate_k"] for row in rows}), 1)

        for row in rows:
            result = self.run_script(
                "--dry-run",
                env={"SLURM_ARRAY_TASK_ID": row["index"], "PROBE_PARALLEL_DRY_RUN": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                f"benchmark.methods.linear_probe.train_mode={row['train_mode']}",
                result.stdout,
            )
            expected_weighting = "inverse_frequency" if row["class_weighting"] == "weighted" else "none"
            self.assertIn(
                f"benchmark.methods.linear_probe.class_weighting={expected_weighting}",
                result.stdout,
            )
            self.assertIn(f"train_mode={row['train_mode']}", result.stdout)

    def test_efficient_probe_modes_emit_method_specific_weighting_overrides(self):
        rows = [row for row in self.task_rows() if row["method"] == "efficient_probe"]
        self.assertTrue({row["train_mode"] for row in rows}.issubset({"classifier", "partial", "all"}))
        if not rows:
            self.skipTest("efficient-probe rows are opt-in in the launcher")
        for row in rows:
            result = self.run_script(
                "--dry-run",
                env={"SLURM_ARRAY_TASK_ID": row["index"], "PROBE_PARALLEL_DRY_RUN": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected_weighting = "inverse_frequency" if row["class_weighting"] == "weighted" else "none"
            self.assertIn(
                f"benchmark.methods.efficient_probe.train_mode={row['train_mode']}",
                result.stdout,
            )
            self.assertIn(
                f"benchmark.methods.efficient_probe.class_weighting={expected_weighting}",
                result.stdout,
            )

    def test_czechlynx_launcher_has_all_linear_probe_modes(self):
        result = self.run_czech_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        parsed = [dict(item.split("=", 1) for item in line.split()) for line in lines]
        linear_rows = [row for row in parsed if row["method"] == "linear_probe"]
        self.assertTrue({row["train_mode"] for row in linear_rows}.issubset({"classifier", "partial", "all"}))
        variant_text = CZECH_SCRIPT.read_text(encoding="utf-8")
        for mode in ("classifier", "partial", "all"):
            self.assertIn(f'linear_probe|-|default|-|{mode}', variant_text)
            self.assertIn(f'efficient_probe|-|default|-|{mode}|weighted', variant_text)
            self.assertIn(f'efficient_probe|-|default|-|{mode}|unweighted', variant_text)
        for row in linear_rows:
            task = self.run_czech_script(
                "--dry-run",
                env={"SLURM_ARRAY_TASK_ID": row["index"], "PROBE_PARALLEL_DRY_RUN": "1"},
            )
            self.assertEqual(task.returncode, 0, task.stderr)
            self.assertIn(
                f"benchmark.methods.linear_probe.train_mode={row['train_mode']}",
                task.stdout,
            )

    def test_czechlynx_split_is_explicit_and_open_profile_can_be_enabled(self):
        result = self.run_czech_script("--list-tasks")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"split_protocol=(split-time_(closed|open)|unseen_eval_split)")

        # Exercise the profile table without changing the repository launcher:
        # uncomment the documented open profile in an isolated temporary copy.
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            script_copy = temp_root / "probe-parallel-czechlynx.sh"
            script_text = CZECH_SCRIPT.read_text(encoding="utf-8")
            script_text = script_text.replace(
                '    # "czechlynx_closed|CzechLynx_v2|CzechLynx|',
                '    "czechlynx_closed|CzechLynx_v2|CzechLynx|',
                1,
            )
            script_text = script_text.replace(
                '    # "czechlynx_open|CzechLynx_v2|CzechLynx|',
                '    "czechlynx_open|CzechLynx_v2|CzechLynx|',
                1,
            )
            # The opt-in unseen-eval profile may be active in the editable
            # launcher; disable it in the copy so only closed/open multiply.
            script_text = script_text.replace(
                '    "czechlynx_unseen_eval|CzechLynx_v2|CzechLynx|',
                '    # "czechlynx_unseen_eval|CzechLynx_v2|CzechLynx|',
                1,
            )
            # joint-fine-tuned checkpoints exist for the closed split only, so the
            # launcher fails closed when such a row meets the open profile. Disable
            # any active joint rows in the copy; the launcher guard has its own test.
            script_text = re.sub(
                r'^(\s*)("vismatch\|[^|]+\|joint-fine-tuned\|)',
                r'\1# \2',
                script_text,
                flags=re.MULTILINE,
            )
            # If that left the table empty, re-enable the split-agnostic cosine
            # row so both profiles still have at least one task to multiply.
            script_text = script_text.replace(
                '    # "cosine|-|default|-|-|-|-"',
                '    "cosine|-|default|-|-|-|-"',
                1,
            )
            script_copy.write_text(script_text, encoding="utf-8")
            # The editable launcher may have descriptor rows active. Supply
            # isolated open-split paths so this test exercises profile
            # multiplication rather than intentionally failing closed on the
            # open profile's missing descriptor checkpoints.
            open_loma = temp_root / "open-loma.safetensors"
            open_rdd = temp_root / "open-rdd.safetensors"
            open_loma.write_bytes(b"test-loma")
            open_rdd.write_bytes(b"test-rdd")
            test_env = os.environ.copy()
            test_env.update({
                "CZECHLYNX_OPEN_DESCRIPTOR_LOMA_CHECKPOINT": str(open_loma),
                "CZECHLYNX_OPEN_DESCRIPTOR_RDD_CHECKPOINT": str(open_rdd),
            })
            result = subprocess.run(
                ["bash", str(script_copy), "--list-tasks"],
                cwd=temp_root,
                env=test_env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line for line in result.stdout.splitlines() if line.startswith("index=")]
        parsed = [dict(item.split("=", 1) for item in line.split()) for line in lines]
        self.assertEqual({row["split_protocol"] for row in parsed}, {"split-time_closed", "split-time_open"})
        counts = {split: sum(row["split_protocol"] == split for row in parsed) for split in {"split-time_closed", "split-time_open"}}
        self.assertEqual(counts["split-time_closed"], counts["split-time_open"])
        self.assertEqual(
            len({(row["split_protocol"], row["method"], row["matcher"], row["checkpoint"], row["train_mode"], row["class_weighting"], row["candidate_k"]) for row in parsed}),
            len(parsed),
        )

    def test_default_and_custom_checkpoint_mapping(self):
        rows = self.task_rows()
        custom_rows = [
            row for row in rows
            if row["method"] == "vismatch" and row["checkpoint"] == "custom"
        ]
        if not custom_rows:
            self.skipTest("no custom Vismatch variant is active in the editable launcher table")

        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = self.checkpoint_env(loma.name, rdd.name)
            for row in custom_rows:
                result = self.run_script(
                    "--dry-run",
                    env={
                        **env,
                        "SLURM_ARRAY_TASK_ID": row["index"],
                        "PROBE_PARALLEL_DRY_RUN": "1",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("checkpoint_components=matcher_only", result.stdout)
                expected_path = loma.name if row["matcher"] == "loma" else rdd.name
                self.assertIn(expected_path, result.stdout)

            default_rows = [
                row for row in rows
                if row["method"] == "vismatch" and row["checkpoint"] == "default"
            ]
            for row in default_rows:
                result = self.run_script(
                    "--dry-run",
                    env={
                        **env,
                        "SLURM_ARRAY_TASK_ID": row["index"],
                        "PROBE_PARALLEL_DRY_RUN": "1",
                    },
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("checkpoint_source=default", result.stdout)
                self.assertNotIn("checkpoint_components=matcher_only", result.stdout)

    def test_missing_custom_checkpoint_fails_before_probe(self):
        rows = self.task_rows()
        custom_rows = [
            row for row in rows
            if row["method"] == "vismatch" and row["checkpoint"] == "custom"
        ]
        if not custom_rows:
            self.skipTest("no custom Vismatch variant is active in the editable launcher table")
        custom_index = int(custom_rows[0]["index"])
        result = self.run_script(
            "--dry-run",
            env={
                "SLURM_ARRAY_TASK_ID": str(custom_index),
                "LOMA_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-loma-checkpoint",
                "RDD_CUSTOM_CHECKPOINT_PATH": "/tmp/does-not-exist-rdd-checkpoint",
            },
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, re.compile(r"(LoMa|RDD-LightGlue) custom checkpoint does not exist"))

    def test_array_index_boundaries(self):
        task_count = len(self.task_rows())
        with tempfile.NamedTemporaryFile() as loma, tempfile.NamedTemporaryFile() as rdd:
            env = {**self.checkpoint_env(loma.name, rdd.name), "PROBE_PARALLEL_DRY_RUN": "1"}
            first = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": "0"})
            last = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count - 1)})
            outside = self.run_script("--dry-run", env={**env, "SLURM_ARRAY_TASK_ID": str(task_count)})
        self.assertEqual(first.returncode, 0, first.stderr)
        script_text = SCRIPT.read_text(encoding="utf-8")
        candidate_matches = re.findall(r"(?m)^(?!\s*#)\s*CANDIDATE_K_VALUES=\(([^)]*)\)", script_text)
        self.assertTrue(candidate_matches)
        expected_candidates = sorted(int(value) for value in candidate_matches[-1].split())
        self.assertIn(f"candidate_k={expected_candidates[0]}", first.stdout)
        self.assertEqual(last.returncode, 0, last.stderr)
        self.assertIn(f"candidate_k={expected_candidates[-1]}", last.stdout)
        self.assertNotEqual(outside.returncode, 0)
        self.assertRegex(outside.stderr, re.compile(rf"outside 0\.\.{task_count - 1}"))

    def test_local_lightglue_uses_shared_budget(self):
        text = (ROOT / "conf/probe.yaml").read_text(encoding="utf-8")
        self.assertIn("local_lightglue:", text)
        self.assertNotIn("      B:", text)
        result = self.run_script("--list-tasks")
        local_lines = [line for line in result.stdout.splitlines() if "method=local_lightglue" in line]
        if local_lines:
            self.assertEqual(len(local_lines), 6)


if __name__ == "__main__":
    unittest.main()
