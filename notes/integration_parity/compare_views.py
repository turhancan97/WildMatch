"""Parity L2: views rebuilt by wildmatch.mining equal the views the paper's training read."""

import hashlib
import json
import os
import sys
from pathlib import Path


def tree(root: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(dirnames + filenames):
            p = Path(dirpath) / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                out[rel] = ("link", os.readlink(p))
            elif p.is_dir():
                out[rel] = ("dir", "")
            else:
                out[rel] = ("file", hashlib.sha256(p.read_bytes()).hexdigest())
    return out


new, old = Path(sys.argv[1]), Path(sys.argv[2])
a, b = tree(new), tree(old)
only_new, only_old = sorted(set(a) - set(b)), sorted(set(b) - set(a))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
kinds = {}
for k, (kind, _) in b.items():
    kinds[kind] = kinds.get(kind, 0) + 1
print(json.dumps({"view": str(old), "entries": len(b), "kinds": kinds, "only_new": only_new[:5], "n_only_new": len(only_new),
                  "only_old": only_old[:5], "n_only_old": len(only_old), "differ": diff[:5], "n_differ": len(diff)}))
for k in diff[:3]:
    print("  ", k, a[k], b[k])
