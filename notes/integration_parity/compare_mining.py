"""Per-query comparison of mined positives/negatives: old vs new code, and each vs the paper's files."""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
PAPER = Path("/home/kargin/Projects/repositories/rdd-parallel-benchmark/outputs/wildlife-reid-10k/SalamanderID2025/indices")


def entries(path: Path):
    d = json.loads(path.read_text())
    rows = []
    for sel in d["selected_frames"]:
        rows.append((sel["query_frame"], tuple(p["frame"] for p in sel.get("positives", [])),
                     tuple(n["frame"] if isinstance(n, dict) else n for n in sel.get("negatives", []))))
    scores = [p["score"] for sel in d["all_frames"] for p in sel.get("positives", [])]
    return rows, scores


def compare(a_dir: Path, b_dir: Path, label: str) -> None:
    files = sorted(a_dir.glob("strong-matches_train_*.json"))
    same = 0
    worst = 0.0
    for f in files:
        b = b_dir / f.name
        if not b.is_file():
            continue
        ra, sa = entries(f)
        rb, sb = entries(b)
        same += ra == rb
        if len(sa) == len(sb) and sa:
            worst = max(worst, max(abs(x - y) for x, y in zip(sa, sb)))
    print(f"{label}: {same}/{len(files)} queries with identical selected positives/negatives; "
          f"max |score diff| over all positives {worst:.3e}")


for backend in ("loma", "rdd"):
    old, new = out / "old" / backend, out / "new" / backend
    compare(old, new, f"{backend} old vs new  ")
    compare(old, PAPER / backend, f"{backend} old vs paper")
    compare(new, PAPER / backend, f"{backend} new vs paper")
