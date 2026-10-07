import json
import tempfile
import unittest
from pathlib import Path

from paper.tools import profile_vismatch as P


def write_run(root: Path, name: str, matcher: str, rerank: float, pairs: float, misses: float, method="vismatch"):
    run = root / name
    run.mkdir(parents=True)
    manifest = {
        "method": method,
        "status": "completed",
        "animal": "SalamanderID2025",
        "variant": matcher,
        "checkpoint_source": "default",
        "vismatch_device": "cuda:NVIDIA GeForce RTX 4090:sm89",
        "num_query": 246,
        "num_database": 1138,
    }
    timings = {
        "vismatch_rerank_sec": rerank,
        "vismatch_model_build_sec": 2.0,
        "feature_extraction_compute_sec": 0.04 * misses,
        "feature_cache_misses": misses,
        "feature_cache_hits": 100.0,
        "feature_cache_lookup_sec": 0.5,
        "vismatch_candidate_k": 50.0,
        "total_run_sec": 2.0 * rerank,
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    (run / "timings.json").write_text(json.dumps(timings))
    (run / "metrics.json").write_text(json.dumps({"num_candidate_pairs": pairs}))


class ProfileVismatchTest(unittest.TestCase):
    def test_costs_per_pair_image_and_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "warm", "loma", rerank=24.6, pairs=12300.0, misses=0.0)
            write_run(root, "cold", "loma", rerank=24.6, pairs=12300.0, misses=1000.0)
            write_run(root, "cosine", "default", rerank=1.0, pairs=1.0, misses=0.0, method="cosine")
            rows = P.collect(root)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertAlmostEqual(row["match_ms_per_pair"], 2.0)
                self.assertAlmostEqual(row["lookup_ms_per_hit"], 5.0)
                self.assertAlmostEqual(row["match_share"], 0.5)
            cold = next(row for row in rows if row["run_dir"].endswith("cold"))
            self.assertAlmostEqual(cold["extract_ms_per_image"], 40.0)
            warm = next(row for row in rows if row["run_dir"].endswith("warm"))
            self.assertIsNone(warm["extract_ms_per_image"])
            (summary,) = P.summarize(rows)
            self.assertEqual((summary["runs"], summary["extracting_runs"]), (2, 1))
            self.assertAlmostEqual(summary["extract_ms_per_image"], 40.0)

    def test_cli_writes_only_its_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            write_run(root, "a", "rdd-lightglue", rerank=10.0, pairs=5000.0, misses=10.0)
            output = Path(tmp) / "profile.csv"
            self.assertEqual(P.main(["--root", str(root), "--output", str(output)]), 0)
            self.assertEqual(len(output.read_text().strip().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
