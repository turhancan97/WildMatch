#!/usr/bin/env python
"""Training-cost ablation: test accuracy against training GPU-hours on CzechLynx closed.

Compares fine-tuned LoMa (matcher only, k=50) with the weighted-loss classifier probes
(frozen / partial / full fine-tuned backbone) in three panels: Top-1, Top-5 and
balanced Top-1. Both
arms trained on RTX 4090 GPUs (``rtx4090_batch``), so GPU-hours compare directly.

Sources, all existing artifacts except the LoMa intermediate evaluations:

* Probe curves: the per-epoch ``[linear_probe] epoch N/50 ... val_top1=`` lines of
  Slurm array 508523 (the query split is the test split, and epoch 50 equals the run's
  final ``top_1``/``top_5``, which is checked), and the tqdm training time per epoch in the
  matching ``.err`` file. Per-epoch evaluation time is excluded. Balanced Top-1 is
  logged per epoch (``val_balanced_top1=``) only since 2026-09-30; older logs contribute
  just their final-epoch value to that panel. ``--probe-job-dir`` selects the array.
* LoMa points: the epoch-299 paper run plus the intermediate checkpoints evaluated by
  ``slurm/eval_loma_epoch_curve.sh`` into ``experiments/compute-efficiency/``. The
  training cost of checkpoint ``epoch_E`` is the sum of the per-epoch ``time/train_s``
  of epochs 0..E in the training log (validation time excluded) times the 4 GPUs.
  Pair mining and feature caching are excluded (user decision).
* References: cosine and default LoMa at k=50 (no training) as horizontal lines.

The curves are test-split trajectories; no epoch is selected from them (user
decision, 2026-09-30). Missing LoMa epochs (evaluations still running) are skipped
with a warning.

Writes ``training_cost.{pdf,png}``, ``training_cost.csv`` (every plotted point) and a
``training_cost.json`` sidecar with sources and checkpoint hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

PROBE_JOB_DIR = ROOT_DIR / "logs/parallel_run/CzechLynx_v2/CzechLynx/split-time_closed/job-508523"
from wildmatch.paths import path as _profile_path  # noqa: E402

_FINETUNING_REPO = _profile_path("external.finetuning_repo")  # lynx-finetuning checkout; None when unset
LOMA_TRAIN_LOG = (_FINETUNING_REPO / "logs" / "czechlynx-loma-ft" / "czechlynx-loma-ft-508111.out") if _FINETUNING_REPO else None
LOMA_TRAIN_GPUS = 4  # sacct 508111: gres/gpu=4 on rtx4090_batch node c20
# Whole-job wall time (sacct ElapsedRaw) for --cost job: training, evaluation passes,
# setup and checkpointing. Probe keys are Slurm array task ids of job 508523.
# Probe tasks use their launcher metadata start/end times (within 3 s of sacct).
LOMA_JOB_ELAPSED_SEC = 8026
COST_MODES = {
    "train": ("training_cost", "Training GPU-hours (RTX 4090)",
              "pure training steps only; evaluation, validation, setup, mining and caching excluded"),
    "job": ("training_cost_job", "Whole-job GPU-hours (RTX 4090)",
             "whole Slurm job: training, per-epoch evaluation/validation, setup and checkpointing; "
             "one-off overhead charged at epoch 0; mining and caching excluded"),
}
LOMA_CHECKPOINT_ROOT = _profile_path("checkpoint_root") / "czechlynx-time-closed" / "loma-b-finetuned-loma-mined-legacy"
PROBE_ROOT = ROOT_DIR / "experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/megadescriptor-l"
EPOCH_CURVE_ROOT = ROOT_DIR / "experiments/compute-efficiency/probe"
# Per shortlist budget k: (fine-tuned LoMa epoch-299 run, default LoMa run). k changes only
# inference (how many candidates LoMa re-ranks), never training cost; probes ignore it.
LOMA_RUNS_BY_K = {
    50: ("20260920T122915Z_0015f14a", "20260920T122545Z_0e052185"),  # the paper's main budget
    100: ("20260920T124911Z_d9d962c3", "20260920T124911Z_ffa69d1b"),
    250: ("20260920T131759Z_612b4791", "20260920T131759Z_daf95fda"),
}
COSINE_RUN = PROBE_ROOT / "cosine/default/20260919T223430Z_5463f66e"
TOP1_TOLERANCE = 1e-4

# Okabe-Ito, matching reid/reporting/plot_figures.py (LoMa fine-tuned vermillion, LoMa
# default blue). Validated with the dataviz palette checker on 2026-09-30: the
# green/pink pair is in the colour-blind floor band, so every series also has its own
# marker and a direct label, and the CSV is the table view.
LOMA_STYLE = {"color": "#d55e00", "marker": "^", "label": "LoMa fine-tuned (matcher)"}
PROBE_STYLES = {
    "classifier": {"color": "#009e73", "marker": "o", "label": "Classifier, frozen"},
    "partial": {"color": "#cc79a7", "marker": "D", "label": "Classifier, partial FT"},
    "all": {"color": "#56b4e9", "marker": "s", "label": "Classifier, full FT"},
}
# Reference lines are named in the legend: in-plot labels collided with the probe
# curves, which run along the default-LoMa line.
REFERENCE_STYLES = {
    "Cosine (no training)": {"color": "#7a7a7a", "linestyle": ":"},
    "LoMa default (no training)": {"color": "#0072b2", "linestyle": "--"},
}
LOMA_START = "LoMa default (no training)"  # fine-tuning starts from these weights at 0 GPU-h
# Only the weighted-loss probes are reported (user decision, 2026-09-30).
CLASS_WEIGHTING = "weighted"
PANELS = (("top_1", "Top-1 (%)"), ("top_5", "Top-5 (%)"), ("balanced_top_1", "Balanced Top-1 (%)"))
# Logged per probe epoch. Balanced Top-1 joined the epoch line on 2026-09-30
# (val_balanced_top1=); older logs lack it and then only contribute a final marker.
CURVE_METRICS = ("top_1", "top_5")
OPTIONAL_CURVE_METRICS = ("balanced_top_1",)


# ── pure parsers (unit-tested) ────────────────────────────────────────────────
_PROBE_EPOCH = re.compile(r"\[linear_probe\] epoch (\d+)/(\d+) .*?val_top1=([0-9.]+) val_top5=([0-9.]+) "
                          r"val_top10=([0-9.]+)(?: val_balanced_top1=([0-9.]+|nan))?")
_TQDM_DONE = re.compile(r"\[linear_probe\]\[train\] epoch (\d+)/\d+: 100%\|[^\[]*\[((?:\d+:)?\d+:\d+)<")
_LOMA_EPOCH = re.compile(r'"epoch":\s*(\d+),.*?"time/train_s":\s*([0-9.eE+-]+)', re.S)
_CHECKPOINT_EPOCH = re.compile(r"epoch_(\d+)")


def parse_probe_epochs(text: str) -> Dict[int, Dict[str, float]]:
    """Per-epoch query-split Top-1/5/10 from ``[linear_probe] epoch N/M`` log lines."""
    epochs: Dict[int, Dict[str, float]] = {}
    for match in _PROBE_EPOCH.finditer(text):
        epochs[int(match.group(1))] = {"top_1": float(match.group(3)), "top_5": float(match.group(4)),
                                       "top_10": float(match.group(5))}
        if match.group(6) is not None:
            epochs[int(match.group(1))]["balanced_top_1"] = float(match.group(6))
    return epochs


def clock_seconds(text: str) -> float:
    """tqdm elapsed ``MM:SS`` or ``H:MM:SS`` -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def parse_tqdm_train_seconds(text: str) -> Dict[int, float]:
    """Training seconds per epoch: the elapsed time of each epoch's final 100% bar.

    tqdm rewrites its line with carriage returns, so the last 100% occurrence of an
    epoch is its completed training loop (evaluation has its own bar).
    """
    seconds: Dict[int, float] = {}
    for match in _TQDM_DONE.finditer(text.replace("\r", "\n")):
        seconds[int(match.group(1))] = clock_seconds(match.group(2))
    return seconds


def parse_loma_train_seconds(text: str) -> Dict[int, float]:
    """Per-epoch ``time/train_s`` from the JSON epoch records of the LoMa training log.

    Epochs are zero-based, like the ``epoch_E`` checkpoint directories.
    """
    return {int(epoch): float(value) for epoch, value in _LOMA_EPOCH.findall(text)}


def parse_loma_eval_seconds(text: str) -> Dict[int, float]:
    """Per-epoch ``time/epoch_eval_s`` (validation) from the LoMa training log."""
    return {int(epoch): float(value) for epoch, value in
            re.findall(r'"epoch":\s*(\d+),.*?"time/epoch_eval_s":\s*([0-9.eE+-]+)', text, flags=re.S)}


def job_cumulative_gpu_hours(train: Mapping[int, float], evaluation: Mapping[int, float],
                             job_seconds: float, gpus: int) -> Dict[int, float]:
    """Whole-job GPU-hours up to each epoch: overhead up front, then training plus evaluation.

    The overhead is everything the job spent outside the per-epoch loops (setup,
    checkpointing, final evaluation). Charging it at epoch 0 is the conservative choice.
    """
    per_epoch = {epoch: train[epoch] + evaluation.get(epoch, 0.0) for epoch in train}
    overhead = job_seconds - sum(per_epoch.values())
    if overhead < 0:
        raise ValueError(f"per-epoch time {sum(per_epoch.values()):.0f}s exceeds the job's {job_seconds:.0f}s")
    hours = cumulative_gpu_hours(per_epoch, gpus)
    return {epoch: value + overhead * gpus / 3600.0 for epoch, value in hours.items()}


def cumulative_gpu_hours(seconds_by_epoch: Mapping[int, float], gpus: int) -> Dict[int, float]:
    """GPU-hours spent up to and including each epoch; fails on a missing epoch."""
    total, out = 0.0, {}
    epochs = sorted(seconds_by_epoch)
    if epochs != list(range(epochs[0], epochs[0] + len(epochs))):
        raise ValueError(f"epoch timings are not contiguous: {epochs[:3]}...{epochs[-3:]}")
    for epoch in epochs:
        total += seconds_by_epoch[epoch] * gpus / 3600.0
        out[epoch] = total
    return out


def checkpoint_epoch(path: str) -> int:
    match = _CHECKPOINT_EPOCH.search(str(path))
    if not match:
        raise ValueError(f"no epoch_<n> directory in checkpoint path {path}")
    return int(match.group(1))


# ── data assembly ─────────────────────────────────────────────────────────────
def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_seconds(meta: Mapping[str, str]) -> float:
    from datetime import datetime

    start, end = (datetime.fromisoformat(meta[key].replace("Z", "+00:00")) for key in ("start_time", "end_time"))
    return (end - start).total_seconds()


def probe_series(cost: str = "train", job_dir: Path = PROBE_JOB_DIR, allow_incomplete: bool = False) -> List[dict]:
    """Weighted probe curves from one launcher array.

    With ``allow_incomplete`` a still-running task contributes the epochs it has logged
    so far (both its metric line and its finished training bar); it has no final metrics
    to check against and no final-epoch marker.
    """
    series = []
    for meta_path in sorted(job_dir.glob("task-*.json")):
        meta = _json(meta_path)
        complete = meta.get("status") == "completed"
        if meta.get("method") != "linear_probe" or meta.get("class_weighting") != CLASS_WEIGHTING:
            continue
        if not complete and not (allow_incomplete and meta.get("status") == "running"):
            continue
        out_path, err_path = Path(meta["stdout_path"]), Path(meta["stderr_path"])
        out_path = out_path if out_path.is_absolute() else ROOT_DIR / out_path
        err_path = err_path if err_path.is_absolute() else ROOT_DIR / err_path
        epochs = parse_probe_epochs(out_path.read_text(errors="ignore"))
        seconds = parse_tqdm_train_seconds(err_path.read_text(errors="ignore"))
        if not complete:  # the current epoch may have finished training but not evaluation
            shared = sorted(set(epochs) & set(seconds))
            epochs = {e: epochs[e] for e in shared}
            seconds = {e: seconds[e] for e in shared}
            if not epochs:
                continue
        last = max(epochs)
        if set(epochs) != set(seconds) or last != max(seconds):
            raise ValueError(f"{meta_path.name}: metric epochs {sorted(epochs)[-1]} and timing epochs "
                             f"{sorted(seconds)[-1]} disagree")
        curve_metrics = list(CURVE_METRICS) + [m for m in OPTIONAL_CURVE_METRICS
                                               if all(m in values for values in epochs.values())]
        if not complete:
            if cost == "job":
                raise ValueError(f"{meta_path.name}: --cost job needs finished tasks (whole-job time)")
            series.append({
                "train_mode": meta["train_mode"], "class_weighting": meta["class_weighting"],
                "run": None, "complete": False, "log": str(out_path.relative_to(ROOT_DIR)),
                "curve_metrics": curve_metrics,
                "points": [{"epoch": e, "gpu_hours": h, **{m: epochs[e][m] for m in curve_metrics}}
                           for e, h in sorted(cumulative_gpu_hours(seconds, gpus=1).items())],
                "final": None,
            })
            print(f"[training-cost] {meta['train_mode']} probe still running: using epochs 1-{last}")
            continue
        run_dir = Path(meta["experiment_run_directory"])
        run_dir = run_dir if run_dir.is_absolute() else ROOT_DIR / run_dir
        metrics = _json(run_dir / "metrics.json")
        for metric in curve_metrics:
            if abs(epochs[last][metric] - float(metrics[metric])) > TOP1_TOLERANCE:
                raise ValueError(f"{meta_path.name}: epoch-{last} logged {metric} {epochs[last][metric]} differs "
                                 f"from the run's {metrics[metric]}; the log is not the test-split metric")
        if cost == "job":
            # The run's linear_probe_train_sec covers training plus the per-epoch test
            # evaluation; spread that evaluation time evenly over the epochs.
            loop = float(_json(run_dir / "timings.json")["linear_probe_train_sec"])
            per_eval = (loop - sum(seconds.values())) / len(seconds)
            hours = job_cumulative_gpu_hours(seconds, {e: per_eval for e in seconds}, _task_seconds(meta), gpus=1)
        else:
            hours = cumulative_gpu_hours(seconds, gpus=1)
        series.append({
            "train_mode": meta["train_mode"], "class_weighting": meta["class_weighting"],
            "run": run_dir.name, "complete": True, "log": str(out_path.relative_to(ROOT_DIR)),
            "curve_metrics": curve_metrics,
            "points": [{"epoch": e, "gpu_hours": hours[e], **{m: epochs[e][m] for m in curve_metrics}}
                       for e in sorted(epochs)],
            "final": {m: float(metrics[m]) for m, _ in PANELS},
        })
    if sorted(item["train_mode"] for item in series) != ["all", "classifier", "partial"]:
        raise ValueError(f"expected one {CLASS_WEIGHTING} probe per train mode in {job_dir} "
                         f"(running ones need --allow-incomplete)")
    return series


def _loma_run_record(run_dir: Path, candidate_k: int) -> Optional[dict]:
    from wildmatch.utils.fingerprints import sha256_file

    manifest = _json(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed":  # still running, or failed
        return None
    checkpoint = manifest.get("vismatch_checkpoint") or {}
    timings = _json(run_dir / "timings.json")
    mode = checkpoint.get("resolved_component_mode") or checkpoint.get("component_mode")
    if checkpoint.get("source") != "custom" or mode != "matcher_only":
        raise ValueError(f"{run_dir}: not a fine-tuned LoMa matcher_only run")
    if int(timings.get("vismatch_candidate_k", -1)) != candidate_k:  # another budget's curve
        return None
    component = checkpoint["components"][0]
    path = Path(component["path"])
    if path.parent.parent != LOMA_CHECKPOINT_ROOT:
        raise ValueError(f"{run_dir}: checkpoint {path} is not from {LOMA_CHECKPOINT_ROOT}")
    if sha256_file(path) != component["sha256"]:
        raise ValueError(f"{run_dir}: checkpoint {path} changed since the run")
    metrics = _json(run_dir / "metrics.json")
    return {"epoch": checkpoint_epoch(str(path)), "run": run_dir.name, "checkpoint": str(path),
            "checkpoint_sha256": component["sha256"], **{m: float(metrics[m]) for m, _ in PANELS}}


def loma_series(candidate_k: int, cost: str = "train") -> dict:
    if LOMA_TRAIN_LOG is None:
        raise ValueError("no lynx-finetuning checkout configured (paths external.finetuning_repo)")
    log = LOMA_TRAIN_LOG.read_text(errors="ignore")
    if cost == "job":
        hours = job_cumulative_gpu_hours(parse_loma_train_seconds(log), parse_loma_eval_seconds(log),
                                         LOMA_JOB_ELAPSED_SEC, gpus=LOMA_TRAIN_GPUS)
    else:
        hours = cumulative_gpu_hours(parse_loma_train_seconds(log), gpus=LOMA_TRAIN_GPUS)
    final_run = PROBE_ROOT / "vismatch/loma" / LOMA_RUNS_BY_K[candidate_k][0]
    records = [_loma_run_record(final_run, candidate_k)]
    if records[0] is None:
        raise ValueError(f"{final_run} is not the k={candidate_k} epoch-299 run")
    for manifest in sorted(EPOCH_CURVE_ROOT.rglob("run_manifest.json")) if EPOCH_CURVE_ROOT.is_dir() else []:
        record = _loma_run_record(manifest.parent, candidate_k)
        if record is not None:
            records.append(record)
    by_epoch: Dict[int, dict] = {}
    for record in filter(None, records):  # newest completed run per epoch wins
        if record["epoch"] not in by_epoch or record["run"] > by_epoch[record["epoch"]]["run"]:
            by_epoch[record["epoch"]] = record
    expected = sorted(checkpoint_epoch(p.name) for p in LOMA_CHECKPOINT_ROOT.glob("epoch_*"))
    missing = [e for e in expected if e not in by_epoch]
    if missing:
        print(f"[training-cost] WARNING: LoMa epochs without a completed evaluation: {missing}")
    points = []
    for epoch in sorted(by_epoch):
        points.append({**by_epoch[epoch], "gpu_hours": hours[epoch]})
    return {"points": points, "missing_epochs": missing, "train_log": str(LOMA_TRAIN_LOG),
            "total_train_gpu_hours": hours[max(hours)]}


def references(candidate_k: int) -> Dict[str, dict]:
    out = {}
    runs = {"Cosine (no training)": COSINE_RUN,
            LOMA_START: PROBE_ROOT / "vismatch/loma" / LOMA_RUNS_BY_K[candidate_k][1]}
    for name, run_dir in runs.items():
        metrics = _json(run_dir / "metrics.json")
        out[name] = {"run": run_dir.name, **{m: float(metrics[m]) for m, _ in PANELS}}
    return out


# ── figure ────────────────────────────────────────────────────────────────────
def _style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def render(probes: List[dict], loma: dict, refs: Dict[str, dict], out_dir: Path, width: float,
           font_size: float, stem: str, xlabel: str = COST_MODES["train"][1],
           panels: Sequence[Tuple[str, str]] = PANELS, x_scale: str = "linear") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    _style()
    height = width * (0.34 if len(panels) >= 3 else 0.42)
    figure, axes = plt.subplots(1, len(panels), figsize=(width, height), sharex=True, squeeze=False)
    axes = axes[0]
    x_max = max([p["points"][-1]["gpu_hours"] for p in probes] + [loma["total_train_gpu_hours"]]) * 1.03
    for axis, (metric, ylabel) in zip(axes, panels):
        for name, ref in refs.items():
            style = REFERENCE_STYLES[name]
            axis.axhline(100 * ref[metric], color=style["color"], linestyle=style["linestyle"], linewidth=1.0, zorder=1)
        for probe in probes:
            style = PROBE_STYLES[probe["train_mode"]]
            if metric in probe["curve_metrics"]:
                axis.plot([p["gpu_hours"] for p in probe["points"]], [100 * p[metric] for p in probe["points"]],
                          color=style["color"], linewidth=1.2, zorder=2)
            # Final epoch of a finished run: the reported number. A running probe's curve
            # simply ends at its last logged epoch, without a marker.
            if probe["final"] is not None:
                axis.plot([probe["points"][-1]["gpu_hours"]], [100 * probe["final"][metric]], marker=style["marker"],
                          markersize=4.5, color=style["color"], linestyle="none", zorder=3)
        # The fine-tuned LoMa curve starts from the default weights at zero training cost.
        # A log axis cannot show 0, so there the default-LoMa reference line carries it.
        xs = [0.0] + [p["gpu_hours"] for p in loma["points"]]
        ys = [100 * refs[LOMA_START][metric]] + [100 * p[metric] for p in loma["points"]]
        start = 0 if x_scale == "linear" else 1
        axis.plot(xs[start:], ys[start:], color=LOMA_STYLE["color"], linewidth=1.6, zorder=4)
        axis.plot(xs[1:], ys[1:], color=LOMA_STYLE["color"], marker=LOMA_STYLE["marker"], markersize=4.5,
                  linestyle="none", zorder=5)
        if x_scale == "linear":
            axis.plot(xs[:1], ys[:1], color=LOMA_STYLE["color"], marker=LOMA_STYLE["marker"], markersize=4.5,
                      markerfacecolor="white", linestyle="none", zorder=5, clip_on=False)
            axis.set_xlim(0, x_max)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            positive = [x for line in axis.get_lines() for x in line.get_xdata() if x > 0]
            axis.set_xscale("log")
            axis.set_xlim(10 ** np.floor(np.log10(min(positive))), x_max * 1.2)
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlabel(xlabel, fontsize=font_size)
        axis.set_ylabel(ylabel, fontsize=font_size)
        axis.tick_params(labelsize=font_size - 1)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
        y_top = 1.15 * max(max(line.get_ydata()) for line in axis.get_lines() if len(line.get_ydata()))
        axis.set_ylim(0, y_top)
    handles = [Line2D([], [], color=LOMA_STYLE["color"], marker=LOMA_STYLE["marker"], linewidth=1.6,
                      markersize=4.5, label=LOMA_STYLE["label"])]
    for mode in ("classifier", "partial", "all"):
        style = PROBE_STYLES[mode]
        handles.append(Line2D([], [], color=style["color"], marker=style["marker"], markersize=4.5,
                              linewidth=1.2, label=style["label"]))
    handles += [Line2D([], [], color=style["color"], linestyle=style["linestyle"], linewidth=1.0, label=name)
                for name, style in REFERENCE_STYLES.items()]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=font_size - 1.5,
                  handlelength=2.4, columnspacing=1.5, bbox_to_anchor=(0.5, -0.01))
    figure.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        figure.savefig(out_dir / f"{stem}.{fmt}", dpi=300, facecolor="white",
                       metadata={"Creator": "paper/figures/plot_training_cost.py"})
    plt.close(figure)


def write_tables(probes: List[dict], loma: dict, refs: Dict[str, dict], out_dir: Path, stem: str,
                 candidate_k: int, cost_note: str = COST_MODES["train"][2]) -> None:
    import csv

    rows = []
    metrics = [m for m, _ in PANELS]
    for probe in probes:
        for point in probe["points"]:
            final = point is probe["points"][-1]
            rows.append({"method": "linear_probe", "train_mode": probe["train_mode"],
                         "class_weighting": probe["class_weighting"], "epoch": point["epoch"],
                         "gpu_hours": round(point["gpu_hours"], 4),
                         **{m: (probe["final"][m] if final and probe["final"] else point.get(m, "")) for m in metrics},
                         "run": probe["run"]})
    for point in loma["points"]:
        rows.append({"method": "vismatch_loma_finetuned", "train_mode": "matcher", "class_weighting": "",
                     "epoch": point["epoch"], "gpu_hours": round(point["gpu_hours"], 4),
                     **{m: point[m] for m in metrics}, "run": point["run"]})
    for name, ref in refs.items():
        rows.append({"method": name, "train_mode": "", "class_weighting": "", "epoch": "", "gpu_hours": 0.0,
                     **{m: ref[m] for m in metrics}, "run": ref["run"]})
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sidecar = {
        "dataset": "CzechLynx_v2/CzechLynx", "split_protocol": "split-time_closed", "candidate_k": candidate_k,
        "gpu": "RTX 4090 (rtx4090_batch)", "cost": cost_note,
        "curves": "test split, no epoch selection",
        "probe_job_dirs": sorted({p["log"].rsplit("/", 1)[0] for p in probes}),
        "probes": [{k: v for k, v in p.items() if k != "points"} | {"train_gpu_hours": p["points"][-1]["gpu_hours"],
                                                                   "last_epoch": p["points"][-1]["epoch"]}
                   for p in probes],
        "loma": loma, "references": refs,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "reports/figures")
    parser.add_argument("--width", type=float, default=6.875, help="Figure width in inches (CVPR text width)")
    parser.add_argument("--font-size", type=float, default=9.0)
    parser.add_argument("--probe-job-dir", type=Path, default=PROBE_JOB_DIR,
                        help="logs/parallel_run/.../job-<id> holding the weighted probe tasks")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="draw still-running probes up to their last logged epoch (no final marker)")
    parser.add_argument("--metrics", default=",".join(m for m, _ in PANELS),
                        help="comma-separated panels in order, from: " + ", ".join(m for m, _ in PANELS))
    parser.add_argument("--output-stem", default=None, help="override the output file stem")
    parser.add_argument("--x-scale", choices=("linear", "log"), default="linear",
                        help="log spreads the first LoMa epoch (0.02 GPU-h) from the rest; zero-cost "
                             "methods then appear only as reference lines")
    parser.add_argument("--cost", choices=sorted(COST_MODES), default="train",
                        help="train: pure training steps (both arms); job: whole Slurm job, conservative")
    parser.add_argument("--candidate-k", type=int, default=50, choices=sorted(LOMA_RUNS_BY_K),
                        help="LoMa shortlist budget; outputs other than k=50 get a _k<k> suffix")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    k = args.candidate_k
    base, xlabel, note = COST_MODES[args.cost]
    stem = args.output_stem or (base if k == 50 else f"{base}_k{k}")
    labels = dict(PANELS)
    chosen = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in labels]
    if unknown or not chosen:
        raise SystemExit(f"--metrics must name panels from {list(labels)}; got {unknown or chosen}")
    panels = [(m, labels[m]) for m in chosen]
    job_dir = args.probe_job_dir if args.probe_job_dir.is_absolute() else ROOT_DIR / args.probe_job_dir
    probes = probe_series(args.cost, job_dir, args.allow_incomplete)
    loma, refs = loma_series(k, args.cost), references(k)
    render(probes, loma, refs, args.output_dir, args.width, args.font_size, stem, xlabel, panels, args.x_scale)
    write_tables(probes, loma, refs, args.output_dir, stem, k, note)
    print(f"[training-cost] LoMa: {len(loma['points'])} points, {loma['total_train_gpu_hours']:.2f} GPU-h ({args.cost} cost) "
          f"in total; {CLASS_WEIGHTING} probes: " + ", ".join(f"{p['train_mode']} {p['points'][-1]['gpu_hours']:.2f} GPU-h"
                                           for p in probes))
    print(f"[training-cost] wrote {args.output_dir / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
