import json
import unittest

from paper.page import export_matcher_ablation as ma


class CommittedExportTests(unittest.TestCase):
    def setUp(self):
        if not ma.DEFAULT_OUT.is_file():
            self.skipTest("matcher_ablation.json not exported")
        self.data = json.loads(ma.DEFAULT_OUT.read_text(encoding="utf-8"))
        self.curves = json.loads(ma.CURVES.read_text(encoding="utf-8"))

    def test_every_dataset_series_and_budget(self):
        self.assertEqual(self.data["ks"], ma.KS)
        self.assertEqual(list(self.data["datasets"]), list(ma.ANIMALS.values()))
        self.assertEqual(self.data["series_order"], [key for key, _ in ma.SERIES])
        for key, ds in self.data["datasets"].items():
            for series in self.data["series_order"]:
                points = ds["series"][series]["points"]
                self.assertEqual(sorted(points, key=int), [str(k) for k in ma.KS], (key, series))
                for k, point in points.items():
                    for metric in ma.METRICS:
                        self.assertTrue(0.0 <= point[metric] <= 1.0, (key, series, k, metric))

    def test_default_matchers_equal_the_paper_curves(self):
        for key, ds in self.data["datasets"].items():
            for series in ("loma_default", "rdd_default"):
                for k, point in ds["series"][series]["points"].items():
                    paper = self.curves["datasets"][key]["series"][series]["points"][k]
                    for metric in ma.METRICS:
                        self.assertEqual(point[metric], paper[metric], (key, series, k, metric))

    def test_ablation_runs_are_recorded_without_paths(self):
        for ds in self.data["datasets"].values():
            for series in ("aliked_default", "superpoint_default"):
                for point in ds["series"][series]["points"].values():
                    self.assertTrue(point["run_id"])
                    self.assertGreater(point["ms_per_pair"], 0)
        ma.check_no_private_paths(self.data)

    def test_mean_is_the_dataset_average(self):
        for series in self.data["series_order"]:
            for k in map(str, ma.KS):
                values = [ds["series"][series]["points"][k]["top_1"] for ds in self.data["datasets"].values()]
                self.assertAlmostEqual(self.data["mean"]["series"][series]["points"][k]["top_1"], sum(values) / 8, 5)


class ExportGuardTests(unittest.TestCase):
    def test_private_paths_fail(self):
        with self.assertRaises(ValueError):
            ma.check_no_private_paths({"x": "/shared/results/run"})


if __name__ == "__main__":
    unittest.main()
