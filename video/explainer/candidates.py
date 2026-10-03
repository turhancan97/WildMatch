#!/usr/bin/env python
"""Real candidate lists for the explainer's shot 6: the top-5 gallery identities of the
explainer's CzechLynx query (first argument: a before_after.json; default: the demo export) under the paper's k=250 default and fine-tuned LoMa
runs, read from their ``scores.npz`` with the shared stable ranking rule. Writes
``candidates.json`` next to this file. Run in the ex-reid environment from the repository
root::

    python video/explainer/candidates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from scripts import export_frequency_bins as fb  # noqa: E402
from scripts import export_score_separation as sep  # noqa: E402


def main() -> None:
    pair_file = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs/assets/demo/before_after/before_after.json"
    pair = json.loads(pair_file.read_text(encoding="utf-8"))
    query_source = next(p for p in pair["pairs"] if p["dataset"] == "lynx_closed")["query"]["source"]
    runs = sep.paper_runs(sep.DEFAULT_PAPER_REPO, "CzechLynx_split-time_closed")
    out = {"query_source": query_source, "k": sep.MAIN_K, "methods": {}}
    for key, run in runs.items():
        run_dir = REPO / Path(run["manifest_path"]).parent
        rows, cols, values, info = sep.load_run(run_dir, run["run_id"])
        q_labels, db_labels = sep.split_labels(run_dir)
        import pandas as pd, yaml
        cfg = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))["dataset"]
        meta = pd.read_csv(Path(cfg["root"]) / cfg["metadata_file"])
        q_paths = meta.loc[meta[cfg["split_col"]] == cfg["query_split_value"], "path"].astype(str).to_numpy()
        q_index = int(np.flatnonzero(q_paths == query_source)[0])
        top = fb.stable_topn(rows, cols, values, len(q_labels), 5)[q_index]
        score = dict(zip(cols[rows == q_index].tolist(), values[rows == q_index].tolist()))
        out["methods"][key] = {
            "run_id": run["run_id"],
            "query_identity": str(q_labels[q_index]),
            "top5": [{"identity": str(db_labels[c]), "score": round(float(score[c]), 4), "correct": bool(db_labels[c] == q_labels[q_index])}
                     for c in top if c >= 0],
        }
    text = json.dumps(out, indent=1)
    assert "/shared/" not in text and "/home/" not in text
    (HERE / "candidates.json").write_text(text + "\n", encoding="utf-8")
    for key, m in out["methods"].items():
        print(key, [(t["identity"], t["score"], t["correct"]) for t in m["top5"]])


if __name__ == "__main__":
    main()
