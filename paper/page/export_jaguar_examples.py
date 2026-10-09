#!/usr/bin/env python3
"""Export the project page's JaguarReID match examples (GPU or CPU; a few pairs).

Takes the fine-tuned LoMa JaguarReID run at k = 250 (sweep ``jaguar_finetuned``), picks ``--n`` queries
by seeded random sampling among those whose stored top-1 gallery photo shows the right jaguar (rank 1 is
read from the run's ``scores.npz`` with the shared lowest-index tie rule, never by score), re-runs the same
fine-tuned matcher on the masked model inputs, and draws the strongest spread-out correspondences on the
original photos (``original_path``; background lightly dimmed with the alpha mask), using the drawing code
of ``paper/figures/plot_match_examples.py``.

Writes ``docs/assets/beyond/jaguar_matches.jpg`` and ``jaguar_matches.json`` (pairs, match counts, stored and
recomputed scores, run id, checkpoint SHA-256; no paths outside the dataset). Recomputed LoMa scores differ
from the stored ones by up to about 1e-2 (bfloat16 autocast), as for the paper's examples.

Research use of the Jaguar data is authorized by the competition's authors (2026-10-07); showing these
photos on the page was approved by the user on 2026-10-09.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # run as a script: make `paper` importable
    sys.path.insert(0, str(REPO_ROOT))
from paper.figures import plot_match_examples as pme  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "assets" / "beyond"
RUNS_ROOT = REPO_ROOT / "experiments" / "probe" / "JaguarReID"


def find_run(root: Path, k: int = 250) -> Path:
    """The newest completed fine-tuned (matcher-only) LoMa JaguarReID run at budget ``k``."""
    best: Optional[Path] = None
    for manifest_path in sorted(root.rglob("vismatch/loma/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = manifest.get("vismatch_checkpoint") or {}
        if (
            manifest.get("status") == "completed"
            and checkpoint.get("source") == "custom"
            and (checkpoint.get("resolved_component_mode") or checkpoint.get("component_mode")) == "matcher_only"
            and int((manifest.get("timings") or {}).get("vismatch_candidate_k") or -1) == k
        ):
            if best is None or manifest_path.parent.name > best.name:
                best = manifest_path.parent
    if best is None:
        raise SystemExit(f"no completed fine-tuned LoMa JaguarReID run at k={k} under {root}")
    return best


def top1(rows: np.ndarray, cols: np.ndarray, values: np.ndarray, n_query: int) -> np.ndarray:
    """Rank-1 gallery index per query: highest stored score, lowest gallery index on ties; -1 if unscored."""
    best = np.full(n_query, -1, dtype=np.int64)
    best_value = np.full(n_query, -np.inf)
    order = np.lexsort((cols, -values))  # by score descending, then gallery index ascending
    for i in order:
        q = rows[i]
        if best[q] < 0:
            best[q], best_value[q] = cols[i], values[i]
    return best


def sample_queries(correct: np.ndarray, n: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(q) for q in rng.choice(np.flatnonzero(correct), size=n, replace=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, help="fine-tuned LoMa run (default: newest at k=250)")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--matches", type=int, default=10, help="correspondences drawn per pair")
    parser.add_argument("--dim", type=float, default=0.35)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    run = args.run or find_run(RUNS_ROOT)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((run / "config.snapshot.yaml").read_text(encoding="utf-8"))
    ds, vm = cfg["dataset"], cfg["benchmark"]["methods"]["vismatch"]
    import pandas as pd

    root = Path(ds["root"])
    frame = pd.read_csv(root / ds["metadata_file"])
    split = frame[ds["split_col"]].astype(str)
    q_rows = np.flatnonzero((split == ds["query_split_value"]).to_numpy())
    g_rows = np.flatnonzero((split == ds["database_split_value"]).to_numpy())
    labels = frame[ds.get("label_col") or "identity"].astype(str).to_numpy()
    with np.load(run / "scores.npz") as z:
        if tuple(int(v) for v in z["shape"]) != (len(q_rows), len(g_rows)):
            raise SystemExit("scores.npz does not match the metadata split")
        rows, cols, values = z["rows"], z["cols"], z["values"]
    best = top1(rows, cols, values, len(q_rows))
    correct = (best >= 0) & (labels[q_rows] == labels[g_rows[np.maximum(best, 0)]])
    chosen = sample_queries(correct, args.n, args.seed)

    from wildmatch.matchers.vismatch import VismatchMatcherBackend, _choose_vismatch_device
    from wildmatch.matchers.vismatch_preprocessing import to_rgb_float_tensor
    from wildmatch.matchers.vismatch_profiles import default_matcher_threshold
    from wildmatch.utils.fingerprints import sha256_file

    component = manifest["vismatch_checkpoint"]["components"][0]
    checkpoint = Path(component["path"])
    if sha256_file(checkpoint) != component["sha256"]:
        raise SystemExit(f"{checkpoint} no longer matches the SHA-256 the run recorded")
    threshold = vm.get("matcher_threshold")
    backend = VismatchMatcherBackend(
        "loma",
        _choose_vismatch_device(args.device),
        int(vm["top_k"]),
        float(threshold) if threshold is not None else default_matcher_threshold("loma"),
        checkpoint_source="custom",
        checkpoint_path=str(checkpoint),
        checkpoint_components="matcher_only",
        loma_arch=str(vm["loma_arch"]),
        resize_max=int(vm["resize_max"]),
    )

    def model_input(row: int) -> Image.Image:
        with Image.open(root / frame.iloc[row]["path"]) as handle:
            return handle.convert("RGB")

    def raw_and_alpha(row: int):
        with Image.open(root / frame.iloc[row]["original_path"]) as handle:
            rgba = handle.convert("RGBA")
        return rgba.convert("RGB"), np.asarray(rgba)[:, :, 3] > 0

    pme._style()
    import matplotlib.pyplot as plt

    columns = 2
    grid_rows = -(-len(chosen) // columns)
    aspect, width = 1.25, 9.0
    cell_w = width / columns
    cell_h = cell_w / (2 * aspect + 0.02) + 0.22
    figure = plt.figure(figsize=(width, grid_rows * cell_h))
    sidecar: List[Dict[str, Any]] = []
    for index, q in enumerate(chosen):
        row_q, row_g = int(q_rows[q]), int(g_rows[best[q]])
        left = backend.extract_frame(to_rgb_float_tensor(model_input(row_q)))
        right = backend.extract_frame(to_rgb_float_tensor(model_input(row_g)))
        result = backend.match_features(left, right)
        (raw_q, fg_q), (raw_g, fg_g) = raw_and_alpha(row_q), raw_and_alpha(row_g)
        for raw, feature in ((raw_q, left), (raw_g, right)):
            if tuple(feature.original_image_size) != (raw.height, raw.width):
                raise SystemExit("original photo and model input differ in size; keypoints cannot be mapped")
        kq = pme.normalized_to_raw_pixels(result.matched_kpts0, (raw_q.height, raw_q.width))
        kg = pme.normalized_to_raw_pixels(result.matched_kpts1, (raw_g.height, raw_g.width))
        r, c = divmod(index, columns)
        axis = figure.add_axes((c / columns, 1 - (r + 1) / grid_rows, 1 / columns - 0.01, 1 / grid_rows - 0.05))
        drawn = pme.draw_paper_pair(
            axis,
            raw_q,
            raw_g,
            (fg_q, fg_g),
            kq,
            kg,
            result.confidences,
            args.matches,
            aspect,
            int(result.match_count),
            args.dim,
            side_tags=index == 0,
            font_size=7.0,
        )
        axis.set_title(f"({chr(ord('a') + index)}) {labels[row_q]}", fontsize=9, pad=2)
        hit = values[(rows == q) & (cols == best[q])]
        sidecar.append(
            {
                "jaguar": labels[row_q],
                "query": str(frame.iloc[row_q]["path"]),
                "top1": str(frame.iloc[row_g]["path"]),
                "match_count": int(result.match_count),
                "drawn_matches": int(drawn),
                "stored_score": round(float(hit[0]), 6),
                "recomputed_score": round(float(result.score), 6),
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out / "jaguar_matches.jpg", dpi=170, facecolor="white", pil_kwargs={"quality": 88})
    plt.close(figure)
    payload = {
        "generated_by": "paper/page/export_jaguar_examples.py",
        "run_id": run.name,
        "candidate_k": 250,
        "checkpoint_sha256": component["sha256"],
        "selection": f"seeded random sample (seed {args.seed}) of queries with a correct stored top-1",
        "pairs": sidecar,
    }
    (args.out / "jaguar_matches.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
