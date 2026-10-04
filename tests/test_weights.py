"""wildmatch weights (paper checkpoints on the Hub) and wildmatch prepare, without network or data."""

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from omegaconf import OmegaConf

from wildmatch import weights as W
from wildmatch.data import prepare as P
from wildmatch.data.registry import load_dataset
from wildmatch.reporting.paper_datasets import PAPER_PROFILES

GMUM_CHECKPOINTS = "/shared/sets/datasets/vision/czechlynx/checkpoints"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ManifestTests(unittest.TestCase):
    def test_every_paper_dataset_has_both_matchers(self):
        manifest = W.load_manifest()
        published = {(e["dataset"], e["matcher"]) for e in manifest["entries"]}
        keys = {
            key
            for key in (
                "hyenaid2022",
                "leopardid2022",
                "nyala",
                "seastarreid2023",
                "whaleshark",
                "zindi",
                "salamander",
                "czechlynx_closed",
                "czechlynx_open",
            )
        }
        self.assertEqual(published, {(k, m) for k in keys for m in W.MATCHERS})
        # Every paper profile is published except BelugaID, which the paper dropped; czechlynx_open
        # serves the unseen-identity protocol.
        paper_animals = {p.animal for p in PAPER_PROFILES} - {"BelugaID"}
        self.assertEqual(paper_animals, {load_dataset(k, "gmum").animal for k in keys})

    def test_manifest_matches_the_registry(self):
        for entry in W.load_manifest()["entries"]:
            checkpoint = entry["files"][0]
            self.assertTrue(checkpoint["hub"].endswith("model.safetensors"))
            self.assertEqual(len(checkpoint["sha256"]), 64)
            for key in [entry["dataset"]] + (["czechlynx_unseen_eval"] if entry["dataset"] == "czechlynx_open" else []):
                registry = load_dataset(key, "gmum").registry.checkpoints.custom[entry["matcher"]]
                self.assertEqual(registry, f"{GMUM_CHECKPOINTS}/{checkpoint['local']}", key)
            for extra in entry["files"][1:]:
                self.assertEqual(Path(extra["hub"]).parent, Path(checkpoint["hub"]).parent)
                self.assertTrue(Path(checkpoint["local"]).is_relative_to(Path(extra["local"]).parent))

    def test_selection(self):
        entries = W.load_manifest()["entries"]
        self.assertEqual({e["matcher"] for e in W.select(entries, ["nyala"], ["loma"])}, {"loma"})
        with self.assertRaisesRegex(W.WeightsError, "no published checkpoints"):
            W.select(entries, ["atrw"])


class DownloadAndStageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.hub = self.tmp / "hub"
        self.root = self.tmp / "checkpoints"
        self.data = {"a/loma/model.safetensors": b"weights", "a/loma/czechlynx_protocol.json": b"{}"}
        for name, content in self.data.items():
            (self.hub / name).parent.mkdir(parents=True, exist_ok=True)
            (self.hub / name).write_bytes(content)
        self.entries = [
            {
                "dataset": "a",
                "matcher": "loma",
                "files": [
                    {
                        "hub": "a/loma/model.safetensors",
                        "local": "x/run/epoch_299/model.safetensors",
                        "sha256": _sha(b"weights"),
                    },
                    {
                        "hub": "a/loma/czechlynx_protocol.json",
                        "local": "x/run/czechlynx_protocol.json",
                        "sha256": _sha(b"{}"),
                    },
                ],
            }
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def fake_hub(self, repo_id, filename, revision, token):
        return self.hub / filename

    def test_download_places_files_and_skips_valid_ones(self):
        with mock.patch.object(W, "_hub_download", side_effect=self.fake_hub) as hub:
            written = W.download(self.entries, self.root, "org/repo", log=lambda _: None)
            self.assertEqual(len(written), 2)
            self.assertEqual((self.root / "x/run/epoch_299/model.safetensors").read_bytes(), b"weights")
            self.assertEqual(W.download(self.entries, self.root, "org/repo", log=lambda _: None), [])
            self.assertEqual(hub.call_count, 2)

    def test_download_rejects_a_changed_file(self):
        (self.hub / "a/loma/model.safetensors").write_bytes(b"tampered")
        with mock.patch.object(W, "_hub_download", side_effect=self.fake_hub):
            with self.assertRaisesRegex(W.WeightsError, "does not match"):
                W.download(self.entries, self.root, "org/repo", log=lambda _: None)
        self.assertFalse((self.root / "x/run/epoch_299/model.safetensors").exists())

    def test_stage_uses_the_hub_layout_and_checks_hashes(self):
        with self.assertRaisesRegex(W.WeightsError, "missing or changed"):
            W.stage(self.entries, self.root, self.tmp / "stage", log=lambda _: None)
        with mock.patch.object(W, "_hub_download", side_effect=self.fake_hub):
            W.download(self.entries, self.root, "org/repo", log=lambda _: None)
        out = W.stage(self.entries, self.root, self.tmp / "stage", log=lambda _: None)
        self.assertEqual((out / "a/loma/model.safetensors").read_bytes(), b"weights")
        self.assertIn(_sha(b"weights"), (out / "SHA256SUMS.md").read_text())

    def test_cli_verify_and_missing_repository(self):
        with (
            mock.patch.object(
                W, "load_manifest", return_value={"repo_id": None, "revision": "main", "entries": self.entries}
            ),
            mock.patch.object(W, "checkpoint_root", return_value=self.root),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(W.main(["verify"]), 1)
            with self.assertRaises(SystemExit):
                W.main(["download"])
        self.assertIn("no Hub repository configured", err.getvalue())


class PrepareCheckTests(unittest.TestCase):
    def test_check_reports_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "img").mkdir()
            (root / "img" / "a.jpg").write_bytes(b"x")
            pd.DataFrame(
                {"identity": [1, 2], "split": ["database", "query"], "path": ["img/a.jpg", "img/b.jpg"]}
            ).to_csv(root / "meta.csv", index=False)
            entry = OmegaConf.create(
                {
                    "root": str(root),
                    "metadata_file": "meta.csv",
                    "label_col": "identity",
                    "split_col": "split",
                    "database_split_value": "database",
                    "query_split_value": "query",
                    "no_background": True,
                    "mask_col": "mask",
                    "registry": {"download": {"raw": "somewhere", "derived": None}},
                }
            )
            with mock.patch.object(P, "load_dataset", return_value=entry):
                report = P.check("x")
            self.assertFalse(report["ready"])
            self.assertEqual(report["problems"], ["columns missing: mask"])
            entry.no_background = False
            with mock.patch.object(P, "load_dataset", return_value=entry):
                report = P.check("x")
            self.assertEqual(len(report["problems"]), 1)
            self.assertIn("1 of 2 sampled images missing", report["problems"][0])
            self.assertEqual(report["raw"], "somewhere")

    def test_registry_records_every_data_source(self):
        from wildmatch.data.registry import dataset_keys

        for key in dataset_keys():
            download = load_dataset(key, "default").registry.download
            self.assertTrue(download.raw, key)
            self.assertIn(download.reproducible, (True, False), key)
        self.assertEqual(
            load_dataset("czechlynx_unseen_eval", "gmum").metadata_file,
            "metadata/czechlynx-unseen-eval/metadata_unseen_eval.csv",
        )

    def test_unseen_split_parameters_match_the_paper_manifest(self):
        self.assertEqual(P.UNSEEN_SOURCE, "czechlynx_open")
        self.assertEqual(P.UNSEEN_ARGS, ["--group-col", "encounter", "--order-col", "date", "--path-col", "path"])


if __name__ == "__main__":
    unittest.main()
