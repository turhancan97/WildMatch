"""Rebuild rules for the prepared tables (wildmatch.data.prepare.sources) and the prepare steps."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from wildmatch.data import prepare as P
from wildmatch.data.prepare import sam3_masks
from wildmatch.data.prepare import sources as S


def _wildlife_metadata():
    rows = []
    for animal, ids in (("NyalaData", ["2", "10", "3"]), ("HyenaID2022", ["b", "a"])):
        for identity in ids:
            for n in range(5):
                rows.append(
                    {
                        "identity": f"{animal}_{identity}",
                        "path": f"images/{animal}/{identity}_{n}.jpg",
                        "species": "nyala" if animal == "NyalaData" else "hyena",
                        "dataset": animal,
                        "split": "train",
                    }
                )
    rows.append(
        {
            "identity": "BelugaID_w",
            "path": "images/BelugaID/beluga-id-test/x.jpg",
            "species": "whale",
            "dataset": "BelugaID",
            "split": "test",
        }
    )
    for n in range(5):
        rows.append(
            {
                "identity": "BelugaID_w",
                "path": f"images/BelugaID/beluga/{n}.jpg",
                "species": "whale",
                "dataset": "BelugaID",
                "split": "train",
            }
        )
    return pd.DataFrame(rows)


class WildlifeTableTests(unittest.TestCase):
    def test_rule(self):
        metadata = _wildlife_metadata()
        table = S.wildlifereid10k_table(metadata, "NyalaData")
        self.assertEqual(list(table.columns), ["identity", "path", "split", "species", "prompt"])
        self.assertEqual(table["identity"].tolist()[:2], [2, 2])  # numeric identities become integers
        self.assertTrue(table["path"].str.startswith("NyalaData/").all())
        self.assertEqual(set(table["split"]), {"train", "test"})
        # closed set: every identity is on both sides
        for _, group in table.groupby("identity"):
            self.assertEqual(set(group["split"]), {"train", "test"})
        self.assertTrue(table.equals(S.wildlifereid10k_table(metadata, "NyalaData")))  # deterministic
        beluga = S.wildlifereid10k_table(metadata, "BelugaID", "BelugaID/beluga/")
        self.assertEqual(len(beluga), 5)
        self.assertEqual(set(beluga["prompt"]), {"beluga whale"})
        with self.assertRaises(S.SourceError):
            S.wildlifereid10k_table(metadata, "Missing")


class SalamanderTableTests(unittest.TestCase):
    def test_rule(self):
        def row(i, identity, date, view="top", split="database"):
            return {
                "image_id": i,
                "identity": identity,
                "path": f"images/SalamanderID2025/database/images/f{i}.jpg",
                "date": date,
                "orientation": view,
                "species": None,
                "split": split,
                "dataset": "SalamanderID2025",
            }

        metadata = pd.DataFrame(
            [
                row(1, "S_1", "2020-01-01"),
                row(2, "S_1", "2021-01-01", "right"),
                row(3, "S_1", "2021-01-01"),
                row(4, "S_2", "2020-01-01"),
                row(5, "S_2", "2020-01-01"),  # one date: database only
                row(6, "S_3", None),  # undated: dropped
                row(7, "S_4", "2019-01-01", split="query"),  # unlabelled competition query: ignored
                {**row(8, "L_1", "2020-01-01"), "dataset": "LynxID2025"},
            ]
        )
        table = S.salamander_table(metadata)
        self.assertEqual(table["image_id"].tolist(), [1, 4, 5, 2, 3])
        self.assertEqual(table["split"].tolist(), ["database"] * 3 + ["query"] * 2)
        self.assertEqual(table["path"].tolist()[3], "query/images/f2.jpg")
        self.assertEqual(table["cross_view"].tolist(), [False, False, False, True, False])

    def test_copy_keeps_existing_identical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, root = Path(tmp, "src"), Path(tmp, "out")
            (source / "images/SalamanderID2025/database/images").mkdir(parents=True)
            (source / "images/SalamanderID2025/database/images/f1.jpg").write_bytes(b"jpg")
            metadata = pd.DataFrame({"path": ["images/SalamanderID2025/database/images/f1.jpg"]})
            table = pd.DataFrame({"path": ["query/images/f1.jpg"]})
            self.assertEqual(S.copy_salamander_images(table, metadata, source, root), 1)
            self.assertEqual(S.copy_salamander_images(table, metadata, source, root), 0)
            (root / "query/images/f1.jpg").write_bytes(b"different")
            with self.assertRaises(S.SourceError):
                S.copy_salamander_images(table, metadata, source, root)
            with self.assertRaises(S.SourceError):
                S.write_new(table, root / "query/images/f1.jpg")


class PrepareStepTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        metadata = _wildlife_metadata()
        metadata.to_csv(self.root / "metadata.csv", index=False)
        self.entry = OmegaConf.create(
            {
                "name": "WildlifeReID-10k",
                "animal": "HyenaID2022",
                "root": str(self.root),
                "metadata_file": "metadata_mdsplit_no_background/metadata_HyenaID2022.csv",
                "label_col": "identity",
                "split_col": "split",
                "database_split_value": "train",
                "query_split_value": "test",
                "no_background": False,
                "mask_col": "mask",
                "registry": {
                    "prepare": {"builder": "wildlifereid10k", "include": None},
                    "download": {"raw": "x", "derived": "y"},
                },
            }
        )
        for path in metadata[metadata.dataset == "HyenaID2022"].path:
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 6, 3), 200, np.uint8)).save(target)

    def tearDown(self):
        self._tmp.cleanup()

    def run_with_entry(self, function, *args):
        with mock.patch.object(P, "load_dataset", return_value=self.entry), contextlib.redirect_stdout(io.StringIO()):
            return function(*args)

    def test_build_finish_and_compare(self):
        command = self.run_with_entry(P.build, "hyena", None, None, None, False)
        split_table = self.root / "wildmatch_prepare" / "HyenaID2022_split.csv"
        self.assertTrue(split_table.is_file())
        self.assertIn("--prompt-column", command)
        self.assertEqual(command[command.index("--root") + 1], str(self.root / "images"))
        with self.assertRaises(S.SourceError):
            self.run_with_entry(P.build, "hyena", None, None, None, False)
        # Simulate the SAM 3 step: full masks, masked files at masked_images/<path>.
        table = pd.read_csv(split_table)
        full = np.ones((4, 6), bool)
        for path in table.path:
            target = self.root / "masked_images" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 6, 3), 200, np.uint8)).save(target)
        pd.DataFrame(
            {
                "path": table.path,
                "masked_path": "masked_images/" + table.path,
                "mask": sam3_masks.encode_mask(full),
                "best_score": 0.9,
                "fg_fraction": 1.0,
                "n_detections": 1,
                "threshold_used": 0.5,
                "prompt_used": "hyena",
            }
        ).to_csv(self.root / "wildmatch_prepare" / "masks_HyenaID2022.csv", index=False)
        target = self.run_with_entry(P.finish, "hyena", None, None, False)
        metadata = pd.read_csv(target)
        self.assertEqual(target, self.root / self.entry.metadata_file)
        self.assertTrue(metadata.path.str.startswith("masked_images/HyenaID2022/").all())
        self.assertIn("mask", metadata.columns)
        with self.assertRaises(SystemExit):
            self.run_with_entry(P.finish, "hyena", None, None, False)
        # With paper_inputs, the comparison reads the paper's table, not the new metadata_file.
        self.entry.registry.paper_inputs = {"metadata_file": self.entry.metadata_file}
        self.entry.metadata_file = "metadata_sam3/missing.csv"
        scores = self.run_with_entry(
            P.compare_masks, "hyena", None, self.root / "wildmatch_prepare" / "masks_HyenaID2022.csv"
        )
        self.assertEqual(len(scores), len(table))
        self.assertTrue((scores.iou == 1.0).all())

    def test_retry_empty_and_finish_merge(self):
        self.run_with_entry(P.build, "hyena", None, None, None, False)
        table = pd.read_csv(self.root / "wildmatch_prepare" / "HyenaID2022_split.csv")
        full, none = np.ones((4, 6), bool), np.zeros((4, 6), bool)
        for path in table.path:
            target = self.root / "masked_images" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((4, 6, 3), 200, np.uint8)).save(target)

        def masks(rows, mask, detections, **extra):
            return pd.DataFrame(
                {
                    "path": rows,
                    "masked_path": "masked_images/" + rows,
                    "mask": sam3_masks.encode_mask(mask),
                    "best_score": 0.9,
                    "fg_fraction": float(mask.mean()),
                    "n_detections": detections,
                    "threshold_used": 0.5,
                    "prompt_used": "hyena",
                    **extra,
                }
            )

        first = pd.concat([masks(table.path[:1], none, 0), masks(table.path[1:], full, 1)])
        first.to_csv(self.root / "wildmatch_prepare" / "masks_HyenaID2022.csv", index=False)
        self.entry.registry.prepare.retry_prompts = ["dog"]
        command = self.run_with_entry(P.retry_empty, "hyena", None, None, False)
        retry_csv = self.root / "wildmatch_prepare" / "HyenaID2022_split_empty.csv"
        self.assertEqual(pd.read_csv(retry_csv).path.tolist(), [table.path[0]])
        self.assertEqual(command[command.index("--csv") + 1], str(retry_csv))
        self.assertEqual(command[command.index("--masks-csv") + 1], "wildmatch_prepare/masks_HyenaID2022_retry.csv")
        self.assertEqual(
            command[command.index("--fallback-prompts") + 1 : command.index("--fallback-prompts") + 3],
            ["dog", "Animal"],
        )
        self.assertIn("full_frame", command)
        masks(table.path[:1], full, 0, full_frame=True).to_csv(
            self.root / "wildmatch_prepare" / "masks_HyenaID2022_retry.csv", index=False
        )
        target = self.run_with_entry(P.finish, "hyena", None, None, False)
        metadata = pd.read_csv(target)
        self.assertEqual(metadata.path.str.replace("masked_images/", "").tolist(), table.path.tolist())  # order kept
        self.assertEqual(metadata.sam3_full_frame.tolist(), [True] + [False] * (len(table) - 1))
        self.assertTrue((metadata["mask"] == sam3_masks.encode_mask(full)).all())

    def test_builders_without_build_step_refuse(self):
        self.entry.registry.prepare.builder = "official"
        with self.assertRaises(SystemExit):
            self.run_with_entry(P.build, "x", None, None, None, False)


class RegistryPrepareBlockTests(unittest.TestCase):
    def test_wildlife_entries_keep_their_build_settings_together(self):
        from wildmatch.data.registry import dataset_keys, load_dataset

        for key in dataset_keys():
            entry = load_dataset(key, "default")
            block = entry.registry.get("prepare")
            if block is None or block.builder != "wildlifereid10k":
                continue
            self.assertEqual(set(block), {"builder", "include", "merge", "masked_dir", "retry_prompts"}, key)
            self.assertEqual(block.merge, "largest", key)
            self.assertEqual(str(entry.metadata_file), f"metadata_sam3/metadata_{entry.animal}.csv", key)
            self.assertEqual(set(entry.registry.paper_inputs), {"metadata_file"}, key)
        self.assertEqual(load_dataset("beluga", "default").registry.prepare.include, "BelugaID/beluga/")


class Sam3GuardTests(unittest.TestCase):
    def test_new_options_parse(self):
        args = sam3_masks.parse_args(
            [
                "--root",
                "r",
                "--csv",
                "c.csv",
                "--segment",
                "--masks-csv",
                "m.csv",
                "--prompt-column",
                "sam3_prompt",
                "--overwrite",
            ]
        )
        self.assertEqual((args.masks_csv, args.prompt_column, args.overwrite), ("m.csv", "sam3_prompt", True))
        self.assertFalse(sam3_masks.parse_args(["--root", "r", "--csv", "c", "--segment"]).overwrite)


if __name__ == "__main__":
    unittest.main()
