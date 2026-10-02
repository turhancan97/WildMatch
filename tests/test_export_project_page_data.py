"""Tests for scripts/export_project_page_data.py on a synthetic paper results directory."""

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "export_project_page_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_project_page_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E = _load()

COLUMNS = ["animal", "split_protocol", "method", "matcher", "backbone", "checkpoint", "checkpoint_component",
           "candidate_k", "class_weighting", "top_1", "top_5", "top_10", "balanced_top_1", "mAP", "mAP_at_k",
           "runtime_min", "run_id", "manifest_path"]


def _row(method, matcher="-", backbone="megadescriptor-l", checkpoint="default", component="", k="",
         weighting="", base=0.5, run="20260101T000000Z_deadbeef"):
    scale = 1.0 if k == "" else 1.0 + int(k) / 5000.0
    return {"animal": "X", "split_protocol": "split", "method": method, "matcher": matcher, "backbone": backbone,
            "checkpoint": checkpoint, "checkpoint_component": component, "candidate_k": k,
            "class_weighting": weighting, "top_1": f"{base * 0.8 * scale:.4f}", "top_5": f"{base * scale:.4f}",
            "top_10": f"{min(0.99, base * 1.1 * scale):.4f}", "balanced_top_1": f"{base * 0.6 * scale:.4f}",
            "mAP": "" if k else "0.3", "mAP_at_k": "0.4" if k else "", "runtime_min": "1.5", "run_id": run,
            "manifest_path": "experiments/probe/X/run_manifest.json"}


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_files(results, stem, ks, with_families=False):
    ablation = []
    for k in ks:
        ablation.append(_row("WildFusion", k=str(k), base=0.40))
        ablation.append(_row("Vismatch", "loma", checkpoint="default", component="auto", k=str(k), base=0.50))
        ablation.append(_row("Vismatch", "loma", checkpoint="fine-tuned", component="matcher_only", k=str(k), base=0.55))
        ablation.append(_row("Vismatch", "rdd-lightglue", checkpoint="default", component="auto", k=str(k), base=0.48))
        ablation.append(_row("Vismatch", "rdd-lightglue", checkpoint="fine-tuned", component="matcher_only", k=str(k), base=0.52))
    ablation.append(_row("Linear Probe", checkpoint="frozen (weighted)", k=str(ks[0]), weighting="weighted", base=0.20))
    ablation.append(_row("Linear Probe", checkpoint="full fine-tuned (weighted)", k=str(ks[0]), weighting="weighted", base=0.45))
    ablation.append(_row("Linear Probe", checkpoint="full fine-tuned (unweighted)", k=str(ks[0]), weighting="unweighted", base=0.46))
    ablation.append(_row("Linear Probe", checkpoint="frozen (weighting unknown)", k=str(ks[0]), weighting="unknown", base=0.21))
    main = [_row("Cosine", backbone="megadescriptor-l", base=0.30), _row("Cosine", backbone="dinov3-l", base=0.32)]
    _write(results / f"{stem}_ablation.csv", ablation)
    _write(results / f"{stem}_main.csv", main)
    if with_families:
        for short, matcher in (("loma", "loma"), ("rdd", "rdd-lightglue")):
            desc = [_row("Vismatch", matcher, checkpoint="default", component="auto", k="250", base=0.50),
                    _row("Vismatch", matcher, checkpoint="descriptor fine-tuned", component="descriptor", k="250", base=0.47)]
            joint = [_row("Vismatch", matcher, checkpoint="default", component="auto", k="250", base=0.50),
                     _row("Vismatch", matcher, checkpoint="joint fine-tuned", component="full", k="250", base=0.57)]
            _write(results / f"{stem}_descriptor_{short}_ablation.csv", desc)
            _write(results / f"{stem}_joint_{short}_ablation.csv", joint)


def _training_cost(results):
    path = results / E.TRAINING_COST_CSV
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "train_mode", "class_weighting", "epoch", "gpu_hours",
                                                    "top_1", "top_5", "balanced_top_1", "run"])
        writer.writeheader()
        for epoch in range(1, 4):
            writer.writerow({"method": "linear_probe", "train_mode": "all", "class_weighting": "weighted",
                             "epoch": epoch, "gpu_hours": epoch * 0.5, "top_1": 0.1 * epoch, "top_5": 0.2 * epoch,
                             "balanced_top_1": 0.05 * epoch, "run": "r1"})
        for epoch, hours in ((0, 0.02), (100, 1.7), (299, 5.1)):
            writer.writerow({"method": "vismatch_loma_finetuned", "train_mode": "matcher", "class_weighting": "",
                             "epoch": epoch, "gpu_hours": hours, "top_1": 0.47, "top_5": 0.57, "balanced_top_1": 0.34,
                             "run": "r2"})
        writer.writerow({"method": "LoMa default (no training)", "train_mode": "", "class_weighting": "", "epoch": "",
                         "gpu_hours": 0, "top_1": 0.46, "top_5": 0.55, "balanced_top_1": 0.31, "run": ""})
        writer.writerow({"method": "Cosine (no training)", "train_mode": "", "class_weighting": "", "epoch": "",
                         "gpu_hours": 0, "top_1": 0.16, "top_5": 0.32, "balanced_top_1": 0.09, "run": ""})


class ExportProjectPageDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paper = Path(self.tmp.name) / "paper"
        self.results = self.paper / "results"
        self.results.mkdir(parents=True)
        for stem, _, _ in E.DATASETS:
            _dataset_files(self.results, stem, E.KS, with_families=stem.startswith("CzechLynx"))
        _dataset_files(self.results, E.UNSEEN[0], E.UNSEEN_KS)
        _training_cost(self.results)
        self.out = Path(self.tmp.name) / "data"

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_writes_json_following_paper_conventions(self):
        summary = E.export(self.paper, self.results, self.out, None)
        self.assertEqual(sorted(summary["outputs"]),
                         ["adapt.json", "curves.json", "manifest.json", "results.json", "training_cost.json"])
        results = json.loads((self.out / "results.json").read_text(encoding="utf-8"))
        labels = {row["method_label"] for row in results["rows"]}
        self.assertIn("LoMa + WildMatch (ours)", labels)
        self.assertIn("Classifier, full (weighted)", labels)
        self.assertFalse(any("unweighted" in label or "unknown" in label for label in labels))
        self.assertEqual({row["dataset"] for row in results["rows"]},
                         {key for _, key, _ in E.DATASETS} | {E.UNSEEN[1]})
        self.assertNotIn("manifest_path", results["rows"][0])
        text = (self.out / "results.json").read_text(encoding="utf-8")
        self.assertNotIn("experiments/", text)
        self.assertNotIn("/home/", text)

    def test_curves_cover_every_budget_and_flats(self):
        E.export(self.paper, self.results, self.out, None)
        curves = json.loads((self.out / "curves.json").read_text(encoding="utf-8"))
        nyala = curves["datasets"]["nyala"]
        self.assertEqual(sorted(int(k) for k in nyala["series"]["loma_finetuned"]["points"]), E.KS)
        self.assertIn("cosine_megadescriptor", nyala["flats"])
        self.assertIn("classifier_full", nyala["flats"])
        self.assertEqual(sorted(int(k) for k in curves["unseen"]["series"]["rdd_default"]["points"]), E.UNSEEN_KS)

    def test_adapt_rows_carry_delta_and_gpu_hours(self):
        E.export(self.paper, self.results, self.out, None)
        adapt = json.loads((self.out / "adapt.json").read_text(encoding="utf-8"))
        by_key = {(r["matcher"], r["variant"]): r for r in adapt["rows"]}
        self.assertEqual(len(by_key), 8)
        self.assertIsNone(by_key[("loma", "default")]["delta_balanced_top_1"])
        self.assertAlmostEqual(by_key[("loma", "matcher")]["gpu_hours"], 5.1)
        self.assertAlmostEqual(by_key[("rdd-lightglue", "joint")]["gpu_hours"], 103.0)
        matcher = by_key[("loma", "matcher")]
        default = by_key[("loma", "default")]
        self.assertAlmostEqual(matcher["delta_balanced_top_1"],
                               matcher["balanced_top_1"] - default["balanced_top_1"], places=5)

    def test_training_cost_series_are_sorted(self):
        E.export(self.paper, self.results, self.out, None)
        cost = json.loads((self.out / "training_cost.json").read_text(encoding="utf-8"))
        hours = [p["gpu_hours"] for p in cost["series"]["loma_finetuned"]["points"]]
        self.assertEqual(hours, sorted(hours))
        self.assertIn("loma_default", cost["flats"])

    def test_missing_series_fails_closed(self):
        path = self.results / "NyalaData_split_ablation.csv"
        rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["checkpoint"] != "fine-tuned"]
        _write(path, rows)
        with self.assertRaises(ValueError):
            E.export(self.paper, self.results, self.out, None)

    def test_figures_render(self):
        figures = Path(self.tmp.name) / "figures"
        summary = E.export(self.paper, self.results, self.out, figures)
        names = sorted(Path(p).name for p in summary["figures"])
        self.assertIn("gain_loma_k250.svg", names)
        self.assertIn("adapt_cost_k250_dark.svg", names)
        self.assertTrue(all((figures / n).stat().st_size > 1000 for n in names))


class CommittedPageDataTests(unittest.TestCase):
    """The committed docs/data files must stay consistent with the exporter contract."""

    DATA = REPO_ROOT / "docs" / "data"

    def test_committed_data_parses_and_is_clean(self):
        if not (self.DATA / "manifest.json").is_file():
            self.skipTest("docs/data not generated")
        for name in ("results.json", "curves.json", "adapt.json", "training_cost.json", "manifest.json"):
            text = (self.DATA / name).read_text(encoding="utf-8")
            json.loads(text)
            for needle in ("/shared/", "/home/", "experiments/"):
                self.assertNotIn(needle, text, f"{name} contains {needle!r}")
        manifest = json.loads((self.DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["sources"])
        self.assertEqual(manifest["conventions"]["main_k"], E.MAIN_K)


if __name__ == "__main__":
    unittest.main()
