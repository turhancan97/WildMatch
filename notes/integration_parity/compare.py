"""Compare the spike's trained tensors and printed metrics: old vs old (noise floor), old vs new."""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

out = Path(sys.argv[1])


def tensors(run: Path) -> dict[str, dict[str, torch.Tensor]]:
    files = sorted(run.rglob("*.safetensors"))
    return {str(f.relative_to(run)): load_file(str(f)) for f in files}


def metrics(log: Path) -> dict[str, float]:
    """Numbers printed by the trainers: LoMa's JSON blocks and RDD's `key: value` lines (last wins)."""
    text = log.read_text(errors="replace")
    found: dict[str, float] = {}
    for block in re.findall(r"^\{\n.*?^\}", text, flags=re.S | re.M):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        stack = [("", data)]
        while stack:
            prefix, value = stack.pop()
            if isinstance(value, dict):
                stack.extend((f"{prefix}{k}/", v) for k, v in value.items())
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                found[prefix.rstrip("/")] = float(value)
    for key, value in re.findall(r"^([A-Za-z_][\w/.\-]*): (-?[\d.]+(?:e-?\d+)?)$", text, flags=re.M):
        found[key] = float(value)
    return found


def compare(a: str, b: str) -> None:
    ra, rb = out / a, out / b
    print(f"== {a} vs {b}")
    if not ra.is_dir() or not rb.is_dir():
        print("   missing run directory")
        return
    ta, tb = tensors(ra), tensors(rb)
    if set(ta) != set(tb):
        print(f"   different safetensors files: {sorted(set(ta) ^ set(tb))}")
    for name in sorted(set(ta) & set(tb)):
        sa, sb = ta[name], tb[name]
        if set(sa) != set(sb):
            print(f"   {name}: different keys ({len(set(sa) ^ set(sb))})")
        identical = worst = 0
        worst_key, rel = "", 0.0
        for key in sorted(set(sa) & set(sb)):
            x, y = sa[key].float(), sb[key].float()
            if x.shape != y.shape:
                print(f"   {name}:{key}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
                continue
            d = (x - y).abs().max().item() if x.numel() else 0.0
            identical += d == 0.0
            if d > worst:
                worst, worst_key = d, key
                rel = d / max(x.abs().max().item(), 1e-12)
        total = len(set(sa) & set(sb))
        print(f"   {name}: {identical}/{total} tensors bit-identical; max |diff| {worst:.3e} ({worst_key}, rel {rel:.2e})")
    ma, mb = metrics(out / f"{a}.log"), metrics(out / f"{b}.log")
    shared = sorted(set(ma) & set(mb))
    diffs = [(abs(ma[k] - mb[k]), k) for k in shared]
    equal = sum(d == 0 for d, _ in diffs)
    print(f"   printed metrics: {equal}/{len(shared)} equal")
    for d, k in sorted(diffs, reverse=True)[:8]:
        if d:
            print(f"     {k}: {ma[k]:.6g} vs {mb[k]:.6g} (|diff| {d:.3e})")


for prefix in ("loma", "rdd"):
    compare(f"{prefix}_old1", f"{prefix}_old2")
    compare(f"{prefix}_old1", f"{prefix}_new")
