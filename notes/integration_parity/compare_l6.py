"""Compare the main-env and integration-env evaluation of one checkpoint: scores and metrics."""

import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])


def run(name):
    found = list((out / f"experiments-{name}").rglob("metrics.json"))
    return found[0].parent if found else None


a, b = run("main"), run("new")
if a is None or b is None:
    print(f"{out.name}: MISSING run (main={a is not None}, new={b is not None})")
    sys.exit(0)
za, zb = np.load(a / "scores.npz"), np.load(b / "scores.npz")
same_entries = all(np.array_equal(za[k], zb[k]) for k in ("shape", "rows", "cols"))
diff = float(np.max(np.abs(za["values"] - zb["values"]))) if same_entries else float("nan")
bit = same_entries and np.array_equal(za["values"], zb["values"])
ma, mb = json.load(open(a / "metrics.json")), json.load(open(b / "metrics.json"))
keys = ("top_1", "top_5", "top_10", "balanced_top_1", "mAP_at_k")
metric_diff = {k: (ma.get(k), mb.get(k)) for k in keys if ma.get(k) != mb.get(k)}
print(f"{out.name}: scores bit-identical={bit} same pairs={same_entries} max|diff|={diff:.2e} "
      f"metrics equal={not metric_diff} {metric_diff or ''}")
