"""Sweeps: spec rules, task table, immutable submissions, task runner, log records and CLI.

Ported on 2026-10-04 from the tests of the removed bash launchers (test_probe_parallel.py),
of scripts/probe_parallel_manifest.py and of the log tools (test_parallel_log_reporting.py).
Nothing here needs data, a GPU or Slurm: checkpoints are temporary files and the evaluation
command is replaced by a stub.
"""

import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from wildmatch import cli
from wildmatch.sweep import logs, runner
from wildmatch.sweep import manifest as M
from wildmatch.sweep import spec as S

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "slurm" / "sweep_task.sbatch"
GMUM_CHECKPOINTS = "/shared/sets/datasets/vision/czechlynx/checkpoints"


def _spec(**changes):
    spec = {"datasets": ["salamander"], "candidate_k": [50], "max_concurrent": 12, "variants": [{"method": "cosine"}]}
    spec.update(changes)
    return spec


def _build(profile="gmum", **changes):
    return S.build_tasks(_spec(**changes), profile)


class _TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_spec(self, **changes):
        path = self.tmp / "sweep.yaml"
        path.write_text(yaml.safe_dump(_spec(**changes)), encoding="utf-8")
        return path

    def checkpoint(self, name="model.safetensors", content=b"weights"):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


class SpecTests(unittest.TestCase):
    def test_packaged_specs_load_and_build(self):
        self.assertIn("parity", S.packaged_specs())
        for name in S.packaged_specs():
            spec = S.load_spec(S.resolve_spec_path(name))
            for profile in ("gmum", "default"):
                with self.subTest(spec=name, profile=profile):
                    try:
                        self.assertTrue(S.build_tasks(spec, profile), f"{name} under {profile}")
                    except S.SweepError as exc:
                        # The unseen-identity entry needs its generated metadata, which exists only
                        # where the split was built (the cluster); every other failure is real.
                        if "metadata file does not exist" not in str(exc):
                            raise
                        self.skipTest(f"{name} under {profile}: unseen-eval metadata not on this machine")

    def test_parity_spec_is_the_reference_set(self):
        tasks = S.build_tasks(S.load_spec(S.resolve_spec_path("parity")), "gmum")
        rows = [
            (
                t["method"],
                t["matcher"],
                t["checkpoint_label"],
                t["checkpoint_components"],
                t["train_mode"],
                t["class_weighting"],
                t["candidate_k"],
                t["profile_id"],
            )
            for t in tasks
        ]
        self.assertEqual(
            rows,
            [
                ("cosine", "-", "default", "-", "-", "-", "50", "salamander"),
                ("wildfusion", "-", "default", "-", "-", "-", "50", "salamander"),
                ("vismatch", "loma", "default", "-", "-", "-", "50", "salamander"),
                ("vismatch", "loma", "custom", "matcher_only", "-", "-", "50", "salamander"),
                ("vismatch", "rdd-lightglue", "default", "-", "-", "-", "50", "salamander"),
                ("vismatch", "rdd-lightglue", "custom", "matcher_only", "-", "-", "50", "salamander"),
                ("linear_probe", "-", "default", "-", "classifier", "weighted", "50", "salamander"),
            ],
        )
        base = f"{GMUM_CHECKPOINTS}/wildlife-reid-10k/SalamanderID2025"
        self.assertEqual(
            tasks[3]["checkpoint_path"], f"{base}/loma-finetuned/legacy-loma-mined/epoch_299/model.safetensors"
        )
        self.assertEqual(
            tasks[5]["checkpoint_path"], f"{base}/rdd-finetuned/legacy-rdd-mined/epoch_299/model.safetensors"
        )
        self.assertEqual({t["checkpoint_owner"] for t in tasks[3:6:2]}, {"SalamanderID2025"})
        self.assertEqual([t["loma_arch"] for t in tasks[2:4]], ["LoMa-B", "LoMa-B"])

    def test_nesting_order_and_probes_only_at_the_first_budget(self):
        tasks = _build(
            datasets=["czechlynx_closed", "czechlynx_open"],
            candidate_k=[250, 50],
            variants=[
                {"method": "cosine"},
                {"method": "efficient_probe", "train_mode": "partial", "class_weighting": "unweighted"},
            ],
        )
        rows = [(t["split_col"], t["candidate_k"], t["method"]) for t in tasks]
        self.assertEqual(
            rows,
            [
                ("split-time_closed", "250", "cosine"),
                ("split-time_closed", "250", "efficient_probe"),
                ("split-time_closed", "50", "cosine"),
                ("split-time_open", "250", "cosine"),
                ("split-time_open", "250", "efficient_probe"),
                ("split-time_open", "50", "cosine"),
            ],
        )
        self.assertEqual({t["no_background"] for t in tasks}, {"true"})
        self.assertEqual({t["image_variant"] for t in tasks}, {"no_background"})

    def test_rules_fail_before_anything_is_written(self):
        bad_variants = {
            "probe without train_mode": {"method": "linear_probe", "class_weighting": "weighted"},
            "probe without weighting": {"method": "linear_probe", "train_mode": "classifier"},
            "train_mode on retrieval": {"method": "cosine", "train_mode": "all"},
            "weighting on retrieval": {"method": "wildfusion", "class_weighting": "weighted"},
            "fine-tuned cosine": {"method": "cosine", "checkpoint": "custom"},
            "vismatch without matcher": {"method": "vismatch"},
            "matcher on cosine": {"method": "cosine", "matcher": "loma"},
            "wrong components": {"method": "vismatch", "matcher": "loma", "checkpoint": "custom", "components": "full"},
            "descriptor as matcher": {
                "method": "vismatch",
                "matcher": "loma",
                "checkpoint": "descriptor-fine-tuned",
                "components": "matcher_only",
            },
            "unknown label": {"method": "vismatch", "matcher": "loma", "checkpoint": "finetuned"},
            "unknown key": {"method": "cosine", "budget": 3},
            "unknown method": {"method": "knn"},
            "unknown matcher": {"method": "vismatch", "matcher": "disk-lightglue"},
            "fine-tuned aliked": {"method": "vismatch", "matcher": "aliked-lightglue", "checkpoint": "custom"},
        }
        for name, variant in bad_variants.items():
            with self.subTest(name), self.assertRaises(S.SweepError):
                _build(variants=[variant])
        with self.assertRaisesRegex(S.SweepError, "unknown dataset"):
            _build(datasets=["nope"])
        with self.assertRaisesRegex(S.SweepError, "repeats a task"):
            _build(variants=[{"method": "cosine"}, {"method": "cosine"}])
        with self.assertRaisesRegex(S.SweepError, "not in the sweep"):
            _build(dataset_overrides={"jaguar": {"evaluation_animal": "x"}})

    def test_ablation_matchers_run_with_default_weights(self):
        rows = [{"method": "vismatch", "matcher": m} for m in ("aliked-lightglue", "superpoint-lightglue")]
        tasks = _build(variants=rows)
        self.assertEqual({t["matcher"] for t in tasks}, {"aliked-lightglue", "superpoint-lightglue"})
        self.assertEqual({t["checkpoint_label"] for t in tasks}, {"default"})
        self.assertEqual({t["loma_arch"] for t in tasks}, {"-"})
        with self.assertRaisesRegex(S.SweepError, "no fine-tuned checkpoints"):
            _build(variants=[{"method": "vismatch", "matcher": "superpoint-lightglue", "checkpoint": "custom"}])

    def test_joint_checkpoints_exist_for_the_closed_split_only(self):
        row = {"method": "vismatch", "matcher": "rdd-lightglue", "checkpoint": "joint-fine-tuned"}
        (task,) = _build(datasets=["czechlynx_closed"], variants=[row])
        self.assertEqual(task["checkpoint_components"], "full")
        self.assertEqual(
            task["checkpoint_path"],
            f"{GMUM_CHECKPOINTS}/czechlynx-time-closed/rdd-joint-finetuned-loma-mined-legacy/epoch_100",
        )
        for key in ("czechlynx_open", "salamander"):
            with self.subTest(key), self.assertRaisesRegex(S.SweepError, "no joint-fine-tuned"):
                _build(datasets=[key], variants=[row])

    def test_descriptor_owner_and_evaluation_animal_overrides(self):
        row = {"method": "vismatch", "matcher": "loma", "checkpoint": "descriptor-fine-tuned"}
        path = f"{GMUM_CHECKPOINTS}/wildlife-reid-10k/GiraffeZebraID/loma-finetuned/x/model.safetensors"
        (task,) = _build(
            datasets=["nyala"],
            variants=[row],
            dataset_overrides={
                "nyala": {
                    "checkpoints": {"descriptor-fine-tuned": {"loma": path}},
                    "checkpoint_owner": "GiraffeZebraID",
                    "evaluation_animal": "NyalaData",
                }
            },
        )
        self.assertEqual(
            (
                task["checkpoint_path"],
                task["checkpoint_owner"],
                task["evaluation_animal"],
                task["checkpoint_components"],
            ),
            (path, "GiraffeZebraID", "NyalaData", "descriptor_only"),
        )
        # Matcher-only checkpoints always belong to the dataset's own animal.
        (custom,) = _build(
            datasets=["nyala"],
            variants=[{"method": "vismatch", "matcher": "loma", "checkpoint": "custom"}],
            dataset_overrides={"nyala": {"checkpoint_owner": "GiraffeZebraID"}},
        )
        self.assertEqual(custom["checkpoint_owner"], "NyalaData")

    def test_unseen_eval_requires_its_generated_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp, "missing.csv"))
            with mock.patch.dict(os.environ, {"CZECHLYNX_UNSEEN_EVAL_METADATA_FILE": missing}):
                with self.assertRaisesRegex(S.SweepError, "metadata file does not exist"):
                    _build(datasets=["czechlynx_unseen_eval"])
            Path(missing).write_text("path\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CZECHLYNX_UNSEEN_EVAL_METADATA_FILE": missing}):
                (task,) = _build(datasets=["czechlynx_unseen_eval"])
        self.assertEqual(
            (task["split_col"], task["database_split_value"], task["query_split_value"]),
            ("unseen_eval_split", "database", "query"),
        )

    def test_inputs_choose_the_current_or_the_paper_tables(self):
        (current,) = _build(datasets=["hyenaid2022"])
        self.assertEqual(current["metadata_file"], "metadata_sam3/metadata_HyenaID2022.csv")
        (paper,) = _build(datasets=["hyenaid2022"], inputs="paper")
        self.assertEqual(paper["metadata_file"], "metadata_mdsplit_no_background/metadata_HyenaID2022.csv")
        both = _build(datasets=["hyenaid2022", "salamander"], dataset_overrides={"hyenaid2022": {"inputs": "paper"}})
        self.assertEqual(
            [t["metadata_file"] for t in both],
            ["metadata_mdsplit_no_background/metadata_HyenaID2022.csv", "split_time_closed_no_background.csv"],
        )
        # entries without paper_inputs keep their table
        (salamander,) = _build(datasets=["salamander"], inputs="paper")
        self.assertEqual(salamander["metadata_file"], "split_time_closed_no_background.csv")

    def test_spec_file_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "s.yaml")
            for content, message in (
                (_spec(candidate_k=[0]), "positive integer"),
                (_spec(variants=[]), "non-empty"),
                (dict(_spec(), extra=1), "unknown sweep keys"),
                (_spec(max_concurrent=0), "max_concurrent"),
                (_spec(inputs="old"), "inputs must be one of"),
                (_spec(experiment_root=""), "experiment_root must be a non-empty path"),
                (_spec(experiment_root="experiments"), "must differ from the default"),
            ):
                path.write_text(yaml.safe_dump(content), encoding="utf-8")
                with self.subTest(message), self.assertRaisesRegex(S.SweepError, message):
                    S.load_spec(path)
        with self.assertRaisesRegex(S.SweepError, "unknown sweep"):
            S.resolve_spec_path("no-such-sweep")

    def test_local_lightglue_uses_the_shared_budget(self):
        text = (ROOT / "src/wildmatch/conf/probe.yaml").read_text(encoding="utf-8")
        self.assertIn("local_lightglue:", text)
        self.assertNotIn("      B:", text)


class SubmissionTests(_TempDir):
    def tasks(self, checkpoint, owner_override=None, label="custom", matcher="rdd-lightglue"):
        overrides = {"checkpoints": {label: {matcher: str(checkpoint)}}}
        if owner_override:
            overrides["checkpoint_owner"] = owner_override
        return _build(
            datasets=["whaleshark"],
            dataset_overrides={"whaleshark": overrides},
            variants=[{"method": "cosine"}, {"method": "vismatch", "matcher": matcher, "checkpoint": label}],
        )

    def create(self, tasks, name="submission"):
        return M.create_submission(tasks, self.tmp / name, "s1", self.write_spec(), "gmum")

    def test_snapshot_is_complete_and_immutable(self):
        checkpoint = self.checkpoint()
        manifest = self.create(self.tasks(checkpoint))
        folder = manifest.parent
        for name in ("probe.yaml", "sweep.yaml", "tasks.tsv", "paths/gmum.yaml", "dataset/salamander.yaml"):
            self.assertTrue((folder / name).is_file(), name)
        payload = json.loads(manifest.read_text())
        self.assertEqual((payload["schema_version"], payload["task_count"], payload["paths_profile"]), (1, 2, "gmum"))
        self.assertEqual(payload["config_tree_sha256"], M.config_tree_sha256(folder))
        record = payload["tasks"][1]
        self.assertEqual(record["dataset"]["split_protocol"], "split")
        self.assertEqual(
            record["checkpoint"],
            {
                "source": "custom",
                "path": str(checkpoint.resolve()),
                "owner": "WhaleSharkID",
                "sha256": M.sha256_path(checkpoint),
            },
        )
        self.assertEqual(
            (folder / "tasks.tsv").read_text().splitlines()[1].split("|")[15:20],
            ["custom", str(checkpoint), "WhaleSharkID", "matcher_only", "-"],
        )
        M.validate_task(payload, record)
        checkpoint.write_bytes(b"weights-v2")
        with self.assertRaisesRegex(ValueError, "content changed"):
            M.validate_task(payload, record)
        checkpoint.write_bytes(b"weights")
        (folder / "dataset" / "salamander.yaml").write_text("name: changed\n")
        with self.assertRaisesRegex(ValueError, "config groups"):
            M.validate_task(payload, record)
        (folder / "probe.yaml").write_text("changed: true\n")
        with self.assertRaisesRegex(ValueError, "snapshot is missing or changed"):
            M.validate_task(payload, record)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create(self.tasks(checkpoint))

    def test_checkpoints_fail_closed_at_submission(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.create(self.tasks(self.tmp / "missing.safetensors"))
        self.assertFalse((self.tmp / "submission").exists())
        foreign = self.checkpoint("wildlife-reid-10k/NyalaData/rdd-finetuned/legacy/epoch_299/model.safetensors")
        with self.assertRaisesRegex(ValueError, "owner mismatch"):
            self.create(self.tasks(foreign))
        # A cross-species descriptor checkpoint is accepted when its owner is declared and the
        # path names the same owner; an unidentifiable path must belong to the dataset itself.
        giraffe = self.checkpoint("wildlife-reid-10k/GiraffeZebraID/loma-finetuned/e/model.safetensors")
        tasks = self.tasks(giraffe, owner_override="GiraffeZebraID", label="descriptor-fine-tuned", matcher="loma")
        payload = json.loads(self.create(tasks, "cross").read_text())
        M.validate_task(payload, payload["tasks"][1])
        plain = self.checkpoint("plain.safetensors")
        with self.assertRaisesRegex(ValueError, "cross-species"):
            self.create(
                self.tasks(plain, owner_override="GiraffeZebraID", label="descriptor-fine-tuned", matcher="loma"),
                "cross-plain",
            )

    def test_component_rules_are_rechecked_in_the_manifest(self):
        checkpoint = self.checkpoint()
        (task,) = _build(
            datasets=["czechlynx_closed"],
            variants=[{"method": "vismatch", "matcher": "loma", "checkpoint": "joint-fine-tuned"}],
            dataset_overrides={"czechlynx_closed": {"checkpoints": {"joint-fine-tuned": {"loma": str(checkpoint)}}}},
        )
        self.assertEqual(M.validate_checkpoint(task)["source"], "custom")
        for label, components, message in (
            ("joint-fine-tuned", "matcher_only", "must use checkpoint_components=full"),
            ("custom", "full", "must use checkpoint_label=joint-fine-tuned"),
            ("custom", "descriptor_only", "must use checkpoint_label=descriptor-fine-tuned"),
            ("descriptor-fine-tuned", "matcher_only", "must use checkpoint_components=descriptor_only"),
        ):
            with self.subTest(label=label, components=components), self.assertRaisesRegex(ValueError, message):
                M.validate_checkpoint(dict(task, checkpoint_label=label, checkpoint_components=components))

    def test_task_index_outside_the_manifest(self):
        manifest = self.create(_build())
        with self.assertRaisesRegex(ValueError, r"outside 0\.\.0"):
            M.load_task(manifest, 1)


class ProbeArgumentTests(_TempDir):
    def payload_and_tasks(self, variants, **changes):
        manifest = M.create_submission(
            _build(variants=variants, **changes), self.tmp / "s", "s", self.write_spec(), "gmum"
        )
        payload = json.loads(manifest.read_text())
        return payload, [runner.task_from_record(record) for record in payload["tasks"]]

    def test_arguments_follow_the_launcher_contract(self):
        checkpoint = self.checkpoint()
        payload, tasks = self.payload_and_tasks(
            [
                {"method": "vismatch", "matcher": "loma"},
                {"method": "vismatch", "matcher": "loma", "checkpoint": "custom"},
                {"method": "linear_probe", "train_mode": "classifier", "class_weighting": "weighted"},
                {"method": "efficient_probe", "train_mode": "all", "class_weighting": "unweighted"},
            ],
            dataset_overrides={"salamander": {"checkpoints": {"custom": {"loma": str(checkpoint)}}}},
        )
        default, custom, linear, efficient = (runner.probe_arguments(payload, task) for task in tasks)
        snapshot = str(Path(payload["config_snapshot"]).parent)
        self.assertEqual(
            default[:8],
            [
                "--config-path",
                snapshot,
                "--config-name",
                "probe",
                "paths=gmum",
                "dataset=salamander",
                "dataset.name=SalamanderID2025",
                "dataset.animal=SalamanderID2025",
            ],
        )
        self.assertEqual(
            default[-3:],
            [
                "benchmark.methods.vismatch.matcher=loma",
                "benchmark.methods.vismatch.loma_arch=LoMa-B",
                "benchmark.methods.vismatch.checkpoint_source=default",
            ],
        )
        self.assertEqual(
            custom[-5:],
            [
                "benchmark.methods.vismatch.checkpoint_source=custom",
                f"benchmark.methods.vismatch.checkpoint_path={checkpoint.resolve()}",
                "benchmark.methods.vismatch.checkpoint_components=matcher_only",
                "benchmark.methods.vismatch.checkpoint_owner=SalamanderID2025",
                "benchmark.methods.vismatch.evaluation_animal=SalamanderID2025",
            ],
        )
        self.assertEqual(
            linear[-2:],
            [
                "benchmark.methods.linear_probe.train_mode=classifier",
                "benchmark.methods.linear_probe.class_weighting=inverse_frequency",
            ],
        )
        self.assertEqual(
            efficient[-2:],
            [
                "benchmark.methods.efficient_probe.train_mode=all",
                "benchmark.methods.efficient_probe.class_weighting=none",
            ],
        )
        self.assertIn("benchmark.candidate_k=50", default)
        self.assertEqual(runner.evaluate_command(["x=1"])[1:], ["-m", "wildmatch", "evaluate", "x=1"])

    def test_experiment_root_reaches_every_task_only_when_set(self):
        payload, tasks = self.payload_and_tasks([{"method": "cosine"}])
        self.assertNotIn("experiment_root", payload)
        plain = runner.probe_arguments(payload, tasks[0])
        self.assertFalse([a for a in plain if a.startswith("output.experiment_root=")])
        manifest = M.create_submission(
            _build(variants=[{"method": "cosine"}]),
            self.tmp / "rooted",
            "s",
            self.write_spec(),
            "gmum",
            experiment_root="experiments/rdd-relaxed",
        )
        rooted = json.loads(manifest.read_text())
        self.assertEqual(rooted["experiment_root"], "experiments/rdd-relaxed")
        arguments = runner.probe_arguments(rooted, runner.task_from_record(rooted["tasks"][0]))
        self.assertEqual(arguments[2:-1], plain[2:])  # [0:2] is --config-path <snapshot folder>
        self.assertEqual(arguments[-1], "output.experiment_root=experiments/rdd-relaxed")

    def test_older_manifests_keep_their_own_configuration(self):
        payload, (task,) = self.payload_and_tasks([{"method": "cosine"}])
        self.assertIn("paths=gmum", runner.probe_arguments(dict(payload, paths_profile=None), task))
        legacy = {key: value for key, value in payload.items() if key not in {"config_tree_sha256", "paths_profile"}}
        arguments = runner.probe_arguments(legacy, task)
        self.assertFalse([a for a in arguments if a.startswith(("paths=", "dataset="))])
        self.assertIn("dataset.root=/shared/sets/datasets/vision/czechlynx/SalamanderID2025", arguments)


class RunTaskTests(_TempDir):
    STUB = (
        "import pathlib, sys\n"
        "run = pathlib.Path(sys.argv[1]); run.mkdir(parents=True, exist_ok=True)\n"
        "(run / 'metrics.json').write_text('{}')\n"
        "print('working'); print('Saved JSON: ' + str(run / 'metrics.json'))\n"
        "print('RuntimeError: synthetic failure', file=sys.stderr)\n"
        "sys.exit(int(sys.argv[2]))\n"
    )

    def setUp(self):
        super().setUp()
        self.checkpoint_path = self.checkpoint()
        tasks = _build(
            variants=[{"method": "cosine"}, {"method": "vismatch", "matcher": "loma", "checkpoint": "custom"}],
            dataset_overrides={"salamander": {"checkpoints": {"custom": {"loma": str(self.checkpoint_path)}}}},
        )
        self.logs_root = self.tmp / "logs" / "parallel_run"
        self.manifest = M.create_submission(
            tasks, self.logs_root / "submissions" / "s1", "s1", self.write_spec(), "gmum"
        )
        (self.tmp / "stub.py").write_text(self.STUB, encoding="utf-8")

    def run_task(self, index, exit_code=0, **kwargs):
        stub = [sys.executable, str(self.tmp / "stub.py"), str(self.tmp / "run"), str(exit_code)]
        with (
            mock.patch.object(runner, "evaluate_command", return_value=stub),
            contextlib.redirect_stdout(io.StringIO()) as out,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            code = runner.run_task(self.manifest, index, self.logs_root, job_id="77", **kwargs)
        return code, out.getvalue()

    def records(self):
        return {record["task_id"]: record for record in logs.load_records(self.logs_root)}

    def test_completed_task_writes_logs_record_and_index(self):
        code, _ = self.run_task(0)
        self.assertEqual(code, 0)
        folder = self.logs_root / "SalamanderID2025" / "SalamanderID2025" / "split" / "job-77"
        stem = "task-000__split__cosine__default__k50"
        self.assertEqual(
            sorted(p.name for p in folder.iterdir()),
            sorted(f"{stem}.{suffix}" for suffix in ("out", "err", "combined.log", "json")),
        )
        self.assertIn("Saved JSON: ", (folder / f"{stem}.out").read_text())
        self.assertIn("synthetic failure", (folder / f"{stem}.combined.log").read_text())
        record = self.records()[0]
        self.assertEqual(
            (
                record["status"],
                record["validation_status"],
                record["experiment_run_directory"],
                record["split_protocol"],
                record["submission_id"],
            ),
            ("completed", "validated", str(self.tmp / "run"), "split", "s1"),
        )
        self.assertEqual(record["error_summary"], "")  # stderr had an error-looking line, but the task succeeded
        with (self.tmp / "logs" / "index.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([(row["task_id"], row["status"]) for row in rows], [("0", "completed")])

    def test_failed_task_records_the_error(self):
        code, _ = self.run_task(1, exit_code=3)
        self.assertEqual(code, 3)
        record = self.records()[1]
        self.assertEqual(
            (record["status"], record["error_summary"], record["checkpoint_owner"]),
            ("failed", "RuntimeError: synthetic failure", "SalamanderID2025"),
        )
        self.assertEqual(record["checkpoint_sha256"], M.sha256_path(self.checkpoint_path))

    def test_validation_failure_runs_nothing_and_cancels_only_itself(self):
        self.checkpoint_path.write_bytes(b"changed")
        with mock.patch.object(runner, "_cancel_own_array_element") as cancel:
            code, _ = self.run_task(1)
        self.assertEqual(code, 1)
        cancel.assert_called_once()
        self.assertFalse((self.tmp / "run").exists())
        record = self.records()[1]
        self.assertEqual(
            (record["status"], record["validation_status"], record["command"]),
            ("failed", "failed", "validation-only: no probe execution"),
        )
        self.assertIn("content changed", record["validation_error"])

    def test_dry_run_prints_the_command_only(self):
        code, out = self.run_task(0, dry_run=True)
        self.assertEqual(code, 0)
        self.assertIn("stub.py", out)
        self.assertEqual(logs.load_records(self.logs_root), [])


class CommandLineTests(_TempDir):
    def call(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                code = cli.main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_dispatch_and_help(self):
        code, out, _ = self.call()
        self.assertEqual(code, 0)
        for name in cli.COMMANDS:
            self.assertIn(name, out)
        self.assertEqual(self.call("nope")[0], 2)

    def test_list_tasks_and_dry_run(self):
        spec = self.write_spec(candidate_k=[10, 50], variants=[{"method": "cosine"}, {"method": "wildfusion"}])
        code, out, _ = self.call("sweep", str(spec), "--list-tasks", "--paths", "gmum")
        self.assertEqual(code, 0)
        rows = [dict(item.split("=", 1) for item in line.split()) for line in out.splitlines()]
        self.assertEqual(
            [(r["index"], r["candidate_k"], r["method"]) for r in rows],
            [("0", "10", "cosine"), ("1", "10", "wildfusion"), ("2", "50", "cosine"), ("3", "50", "wildfusion")],
        )
        logs_root = self.tmp / "logs" / "parallel_run"
        code, out, _ = self.call(
            "sweep",
            str(spec),
            "--dry-run",
            "--paths",
            "gmum",
            "--logs-root",
            str(logs_root),
            "--max-concurrent",
            "7",
            "--sbatch-arg=--partition=x",
        )
        self.assertEqual(code, 0, out)
        self.assertRegex(
            out,
            r"sbatch --array=0-3%7 --export=ALL,WILDMATCH_SWEEP_MANIFEST=\S+manifest\.json "
            r"--partition=x slurm/sweep_task\.sbatch",
        )
        self.assertEqual(len(list((logs_root / "submissions").glob("*/manifest.json"))), 1)
        custom = self.tmp / "custom" / "probe.yaml"
        custom.parent.mkdir()
        custom.write_text(
            (ROOT / "src/wildmatch/conf/probe.yaml")
            .read_text()
            .replace('experiment_root: "experiments"', 'experiment_root: "experiments/elsewhere"')
        )
        code, out, _ = self.call(
            "sweep", str(spec), "--dry-run", "--paths", "gmum", "--logs-root", str(logs_root), "--config", str(custom)
        )
        self.assertEqual(code, 0, out)
        newest = max((logs_root / "submissions").glob("*/probe.yaml"), key=lambda p: p.parent.name)
        self.assertIn("experiments/elsewhere", newest.read_text())
        self.assertTrue((newest.parent / "dataset" / "salamander.yaml").is_file())

    def test_submit_needs_the_sbatch_script(self):
        code, _, err = self.call("sweep", "parity", "--submit", "--sbatch-script", str(self.tmp / "none.sbatch"))
        self.assertEqual(code, 2)
        self.assertIn("sbatch script not found", err)

    def test_sbatch_script(self):
        result = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --output=logs/parallel_run/%x_%A_%a.out", text)
        self.assertIn("#SBATCH --error=logs/parallel_run/%x_%A_%a.err", text)
        self.assertIn('cd "${SLURM_SUBMIT_DIR:?', text)
        self.assertIn(
            'exec wildmatch sweep-task --manifest "${WILDMATCH_SWEEP_MANIFEST}" --index "${SLURM_ARRAY_TASK_ID}"', text
        )


class LogRecordTests(_TempDir):
    def test_metadata_lifecycle_and_index(self):
        logs_root = self.tmp / "logs" / "parallel_run"
        metadata = logs_root / "WildlifeReID-10k" / "WhaleSharkID" / "split" / "job-123" / "task-000__x.json"
        stderr = metadata.with_suffix(".err")
        logs.init_record(
            metadata,
            job_id="123",
            task_id=0,
            dataset="WildlifeReID-10k",
            animal="WhaleSharkID",
            method="vismatch",
            matcher="rdd-lightglue",
            class_weighting="weighted",
            checkpoint="custom",
            checkpoint_path="/tmp/checkpoint.safetensors",
            candidate_k=100,
            command="wildmatch evaluate",
            start_time="2026-08-22T10:00:00Z",
            stdout_path="o",
            stderr_path=str(stderr),
            combined_path="c",
        )
        self.assertEqual(json.loads(metadata.read_text())["status"], "running")
        stderr.write_text("ordinary warning\nRuntimeError: synthetic failure\n", encoding="utf-8")
        logs.update_record(
            metadata,
            status="failed",
            end_time="2026-08-22T10:01:00Z",
            experiment_run_directory="experiments/probe/example",
            error_file=stderr,
        )
        payload = json.loads(metadata.read_text())
        self.assertEqual((payload["status"], payload["error_summary"]), ("failed", "RuntimeError: synthetic failure"))
        index = self.tmp / "logs" / "index.csv"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            logs.main(
                [
                    "--logs-root",
                    str(logs_root),
                    "--write-index",
                    "--method",
                    "vismatch",
                    "--status",
                    "failed",
                    "--format",
                    "csv",
                ]
            )
        self.assertIn("synthetic failure", out.getvalue())
        with index.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            [(r["candidate_k"], r["class_weighting"], r["experiment_run_directory"]) for r in rows],
            [("100", "weighted", "experiments/probe/example")],
        )
        with self.assertRaises(ValueError):
            logs.update_record(metadata.with_name("missing.json"), status="failed", end_time="t")

    def test_summary_filters(self):
        folder = self.tmp / "logs" / "dataset" / "animal" / "job-1"
        folder.mkdir(parents=True)
        for task_id, method in ((0, "cosine"), (1, "vismatch")):
            (folder / f"task-00{task_id}__{method}.json").write_text(
                json.dumps(
                    {
                        "job_id": "1",
                        "task_id": task_id,
                        "dataset": "dataset",
                        "animal": "animal",
                        "method": method,
                        "matcher": "-" if method == "cosine" else "loma",
                        "checkpoint": "default",
                        "candidate_k": 100,
                        "status": "completed",
                        "start_time": f"2026-08-22T10:0{task_id}:00Z",
                    }
                ),
                encoding="utf-8",
            )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            logs.main(["--logs-root", str(self.tmp / "logs"), "--method", "vismatch", "--format", "csv"])
        rows = list(csv.DictReader(out.getvalue().splitlines()))
        self.assertEqual([row["matcher"] for row in rows], ["loma"])


if __name__ == "__main__":
    unittest.main()
