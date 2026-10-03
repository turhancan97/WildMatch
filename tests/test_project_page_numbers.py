"""Every number typed into the project-page Markdown must equal the exported data.

The pages transcribe three tables (what to adapt, unseen-identity protocol, training-cost
final points) and the dataset statistics. These tests parse the Markdown tables and compare
them with ``docs/data/*.json`` (and, when the paper clone is present, with the paper's
``tab_datasets.tex``), so a stale transcription fails the suite instead of reaching readers.
"""

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
DATA = DOCS / "data"
PAPER_TABLE = REPO_ROOT.parent / "ECIR-Animal-ReID-Paper" / "paper" / "tables" / "tab_datasets.tex"


def markdown_tables(text):
    """Yield each pipe table as a list of rows (lists of stripped cell strings)."""
    rows, tables = [], []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            rows.append(cells)
        elif rows:
            tables.append(rows); rows = []
    if rows:
        tables.append(rows)
    return tables


def plain(cell):
    return re.sub(r"[*_`]", "", cell).replace("−", "-").replace("–", "-").strip()


def num(cell):
    return float(plain(cell).replace(",", "").replace("%", "").replace("+", ""))


def pct(fraction):
    return round(fraction * 100, 1)


@unittest.skipUnless((DATA / "manifest.json").is_file(), "docs/data not generated")
class TranscribedNumbersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curves = json.loads((DATA / "curves.json").read_text(encoding="utf-8"))
        cls.adapt = json.loads((DATA / "adapt.json").read_text(encoding="utf-8"))
        cls.cost = json.loads((DATA / "training_cost.json").read_text(encoding="utf-8"))
        cls.results_md = (DOCS / "results" / "index.md").read_text(encoding="utf-8")
        cls.compute_md = (DOCS / "results" / "training-cost.md").read_text(encoding="utf-8")
        cls.datasets_md = (DOCS / "datasets" / "index.md").read_text(encoding="utf-8")

    def _table_with_header(self, text, first_header):
        for table in markdown_tables(text):
            if plain(table[0][0]) == first_header:
                return table
        self.fail(f"no Markdown table starting with {first_header!r}")

    def test_what_to_adapt_table_matches_adapt_json(self):
        table = self._table_with_header(self.results_md, "Matcher")
        rows = {(r["matcher_label"], r["variant"]): r for r in self.adapt["rows"]}
        variant_of = {"none (default)": "default", "descriptor": "descriptor", "matching module (ours)": "matcher", "both": "joint"}
        checked = 0
        for cells in table[1:]:
            matcher, variant = plain(cells[0]), plain(cells[1])
            row = rows[(matcher, variant_of[variant])]
            self.assertEqual(num(cells[2]), pct(row["top_5"]), cells)
            self.assertEqual(num(cells[3]), pct(row["balanced_top_1"]), cells)
            if row["delta_balanced_top_1"] is None:
                self.assertEqual(plain(cells[4]), "-")
                self.assertEqual(plain(cells[5]), "-")
            else:
                self.assertEqual(num(cells[4]), pct(row["delta_balanced_top_1"]), cells)
                self.assertEqual(num(cells[5]), float(row["gpu_hours"]), cells)
            checked += 1
        self.assertEqual(checked, 8)

    def test_unseen_table_matches_curves_json(self):
        table = self._table_with_header(self.results_md, "Method")
        unseen = self.curves["unseen"]
        ks = [int(re.sub(r"\D", "", h)) for h in table[0][1:]]
        self.assertEqual(ks, unseen["ks"])
        series_for = {"LoMa default": "loma_default", "LoMa + WildMatch": "loma_finetuned",
                      "RDD-LightGlue default": "rdd_default", "RDD + WildMatch": "rdd_finetuned",
                      "WildFusion": "wildfusion"}
        flat_for = {"MegaDescriptor-L cosine": "cosine_megadescriptor", "DINOv3-L cosine": "cosine_dinov3"}
        checked = 0
        for cells in table[1:]:
            name = plain(cells[0])
            for k, cell in zip(ks, cells[1:]):
                top5, bal = (num(p) for p in plain(cell).split("/"))
                if name in flat_for:
                    flat = unseen["flats"][flat_for[name]]
                    expected = (pct(flat["top_5"]), pct(flat["balanced_top_1"]))
                else:
                    point = unseen["series"][series_for[name]]["points"][str(k)]
                    expected = (pct(point["top_5"]), pct(point["balanced_top_1"]))
                self.assertEqual((top5, bal), expected, f"{name} k={k}")
                checked += 1
        self.assertEqual(checked, 7 * 4)

    def test_training_cost_final_points_match_exports(self):
        table = self._table_with_header(self.compute_md, "Method")
        cz = self.curves["datasets"]["czechlynx"]
        k250 = lambda key: cz["series"][key]["points"]["250"]
        expected = {
            "Cosine retrieval (MegaDescriptor-L)": (0.0, cz["flats"]["cosine_megadescriptor"]),
            "LoMa default": (0.0, k250("loma_default")),
            "LoMa + WildMatch": (self.cost["series"]["loma_finetuned"]["points"][-1]["gpu_hours"], k250("loma_finetuned")),
            "Classifier, frozen backbone (weighted)": (self.cost["series"]["classifier_frozen"]["points"][-1]["gpu_hours"], cz["flats"]["classifier_frozen"]),
            "Classifier, partially fine-tuned (weighted)": (self.cost["series"]["classifier_partial"]["points"][-1]["gpu_hours"], cz["flats"]["classifier_partial"]),
            "Classifier, fully fine-tuned (weighted)": (self.cost["series"]["classifier_full"]["points"][-1]["gpu_hours"], cz["flats"]["classifier_full"]),
        }
        checked = 0
        for cells in table[1:]:
            hours, metrics = expected[plain(cells[0])]
            self.assertEqual(num(cells[1]), round(hours, 1), cells)
            self.assertEqual(num(cells[2]), pct(metrics["top_1"]), cells)
            self.assertEqual(num(cells[3]), pct(metrics["top_5"]), cells)
            self.assertEqual(num(cells[4]), pct(metrics["balanced_top_1"]), cells)
            checked += 1
        self.assertEqual(checked, 6)

    def test_gain_statements_match_curves(self):
        """Prose on the Results page: smallest k=250 gain on Sea star; Top-5 up everywhere;
        balanced Top-1 down on exactly Leopard and Turtle; 8/8 and 7/8 against the classifier."""
        D = self.curves["datasets"]
        gains = {}
        for key, ds in D.items():
            ft, df = ds["series"]["loma_finetuned"]["points"]["250"], ds["series"]["loma_default"]["points"]["250"]
            gains[key] = (ft["top_5"] - df["top_5"], ft["balanced_top_1"] - df["balanced_top_1"])
        self.assertEqual(min(gains, key=lambda k: gains[k][0]), "sea_star")
        self.assertTrue(all(g[0] > 0 for g in gains.values()))
        self.assertEqual({k for k, g in gains.items() if g[1] < 0}, {"leopard", "turtle"})
        bal_wins = sum(D[k]["series"]["loma_finetuned"]["points"]["250"]["balanced_top_1"] > D[k]["flats"]["classifier_full"]["balanced_top_1"] for k in D)
        top5_wins = sum(D[k]["series"]["loma_finetuned"]["points"]["250"]["top_5"] > D[k]["flats"]["classifier_full"]["top_5"] for k in D)
        self.assertEqual((bal_wins, top5_wins), (8, 7))
        self.assertLess(D["sea_star"]["series"]["loma_finetuned"]["points"]["250"]["top_5"], D["sea_star"]["flats"]["classifier_full"]["top_5"])

    @unittest.skipUnless(PAPER_TABLE.is_file(), "paper clone not present")
    def test_dataset_table_matches_paper_source(self):
        table = self._table_with_header(self.datasets_md, "Dataset")
        tex = PAPER_TABLE.read_text(encoding="utf-8")
        paper = {}
        for line in tex.splitlines():
            m = re.match(r"^([A-Za-z ]+?)~\\cite\{[^}]*\}\s*&(.*)\\\\", line.strip())
            if not m:
                continue
            cells = [c.strip() for c in m.group(2).split("&")]
            numbers = [c.replace("{,}", "").replace("\\%", "") for c in cells if "includegraphics" not in c]
            paper[m.group(1).strip()] = [float(n) for n in numbers]
        self.assertEqual(len(paper), 8)
        for cells in table[1:]:
            name = plain(cells[0])
            page = [num(c) for c in cells[2:]]
            self.assertEqual(page, paper[name], name)


if __name__ == "__main__":
    unittest.main()
