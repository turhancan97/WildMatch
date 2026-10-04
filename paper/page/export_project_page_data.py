#!/usr/bin/env python3
"""Export project-page data and page figures from the paper's results snapshot.

The manuscript's tables and figures are built from the CSV files in the paper
repository's ``results/`` directory (copies of this repository's paper-table
exports, frozen when the paper was assembled). The project page must show the same
numbers, so this script reads that directory, never the live tables under
``reports/``, and writes:

``docs/data/results.json``       one record per (dataset, method, budget) following
                                 the paper's conventions (eight datasets, matcher-only
                                 fine-tuning, class-weighted classifiers only, no
                                 CzechLynx open split)
``docs/data/curves.json``        accuracy-versus-k series for the explorer, plus the
                                 unseen-identity protocol
``docs/data/adapt.json``         the "descriptor or matching module" comparison
                                 (CzechLynx closed, k=250) with training GPU-hours
``docs/data/training_cost.json`` the training-cost curves at k=250
``docs/data/manifest.json``      source hashes, paper commit, conventions

and, unless ``--no-figures`` is given, static page figures in the brand palette under
``docs/assets/figures/page/`` as SVG, in a light and a dark variant each.

Absolute paths are never written. Missing inputs fail closed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_REPO = REPO_ROOT.parent / "ECIR-Animal-ReID-Paper"
DEFAULT_OUT = REPO_ROOT / "docs" / "data"
DEFAULT_FIGURES = REPO_ROOT / "docs" / "assets" / "figures" / "page"

# Paper datasets in the order of the main figure (alphabetical by display name).
DATASETS: List[Tuple[str, str, str]] = [  # (CSV stem, page key, display label)
    ("CzechLynx_split-time_closed", "czechlynx", "CzechLynx"),
    ("HyenaID2022_split", "hyena", "Hyena"),
    ("LeopardID2022_split", "leopard", "Leopard"),
    ("NyalaData_split", "nyala", "Nyala"),
    ("SalamanderID2025_split", "salamander", "Salamander"),
    ("SeaStarReID2023_split", "sea_star", "Sea star"),
    ("WhaleSharkID_split", "whale_shark", "Whale shark"),
    ("ZindiTurtleRecall_split", "turtle", "Turtle"),
]
UNSEEN = ("CzechLynx_unseen_eval_split", "czechlynx_unseen", "CzechLynx, unseen identities")
KS = [10, 50, 100, 250, 500, 1000]
UNSEEN_KS = [10, 50, 100, 160]
MAIN_K = 250
METRICS = ["top_1", "top_5", "top_10", "balanced_top_1"]
METRIC_LABELS = {"top_1": "Top-1", "top_5": "Top-5", "top_10": "Top-10", "balanced_top_1": "Balanced top-1"}
CANDIDATE_BACKBONE = "megadescriptor-l"

# Budget-dependent series: key -> (method, matcher, checkpoint label, display label, family).
SERIES: Dict[str, Tuple[str, str, str, str, str]] = {
    "loma_finetuned": ("Vismatch", "loma", "fine-tuned", "LoMa + WildMatch (ours)", "loma"),
    "loma_default": ("Vismatch", "loma", "default", "LoMa (default)", "loma"),
    "rdd_finetuned": ("Vismatch", "rdd-lightglue", "fine-tuned", "RDD + WildMatch (ours)", "rdd"),
    "rdd_default": ("Vismatch", "rdd-lightglue", "default", "RDD-LightGlue (default)", "rdd"),
    "wildfusion": ("WildFusion", "-", "default", "WildFusion", "wildfusion"),
}
# Budget-independent baselines: key -> (method, backbone, checkpoint label, display label).
FLATS: Dict[str, Tuple[str, str, Optional[str], str]] = {
    "cosine_megadescriptor": ("Cosine", "megadescriptor-l", "default", "MegaDescriptor-L cosine"),
    "cosine_dinov3": ("Cosine", "dinov3-l", "default", "DINOv3-L cosine"),
    "classifier_frozen": ("Linear Probe", "megadescriptor-l", "frozen (weighted)", "Classifier, frozen (weighted)"),
    "classifier_partial": (
        "Linear Probe",
        "megadescriptor-l",
        "partial fine-tuned (weighted)",
        "Classifier, partial (weighted)",
    ),
    "classifier_full": (
        "Linear Probe",
        "megadescriptor-l",
        "full fine-tuned (weighted)",
        "Classifier, full (weighted)",
    ),
}
# "Descriptor or matching module" (paper Table 2): training GPU-hours of training
# steps at epoch 299, as recorded in the paper repository's results/make_tables.py.
ADAPT_VARIANTS = [  # (page key, checkpoint label in the CSVs, display label)
    ("default", "default", "none (default)"),
    ("descriptor", "descriptor fine-tuned", "descriptor branch"),
    ("matcher", "fine-tuned", "matching module (ours)"),
    ("joint", "joint fine-tuned", "both"),
]
ADAPT_GPU_HOURS = {
    ("loma", "descriptor"): 77.0,
    ("loma", "matcher"): 5.1,
    ("loma", "joint"): 84.0,
    ("rdd-lightglue", "descriptor"): 89.0,
    ("rdd-lightglue", "matcher"): 5.0,
    ("rdd-lightglue", "joint"): 103.0,
}
ADAPT_MATCHERS = [("loma", "LoMa"), ("rdd-lightglue", "RDD-LightGlue")]
TRAINING_COST_CSV = "training_cost_k250_balanced_top5.csv"

# Brand palette (team decision 2026-09-28). Colour follows the matcher family; the
# fine-tuned series is solid with filled markers, the default dashed with hollow
# markers, so identity never rests on hue alone. Cosine and classifier baselines are
# context and stay grey (emphasis form).
BLUE, RED, GOLD = "#3a7eab", "#cf4832", "#b8860b"
GREY_TEXT, GREY_HEAD, GREY_RULE = "#6d6e71", "#58595b", "#d1d3d4"
FAMILY_COLOR = {"loma": BLUE, "rdd": RED, "wildfusion": GOLD}
FAMILY_MARKER = {"loma": "o", "rdd": "s", "wildfusion": "^"}


# ----------------------------------------------------------------------------- IO


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing paper results file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def paper_commit(paper_repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(paper_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


# ------------------------------------------------------------------------ records


def _metrics(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {m: to_float(row.get(m)) for m in METRICS}
    out["mAP"] = to_float(row.get("mAP"))
    out["mAP_at_k"] = to_float(row.get("mAP_at_k"))
    out["runtime_min"] = to_float(row.get("runtime_min"))
    return {k: (round(v, 6) if v is not None else None) for k, v in out.items()}


def _match(
    row: Dict[str, str], method: str, matcher: Optional[str], checkpoint: Optional[str], backbone: Optional[str]
) -> bool:
    if row.get("method") != method:
        return False
    if matcher is not None and row.get("matcher", "-") != matcher:
        return False
    if checkpoint is not None and row.get("checkpoint") != checkpoint:
        return False
    if backbone is not None and row.get("backbone") != backbone:
        return False
    return True


def collect_dataset(results_dir: Path, stem: str, key: str, label: str, ks: List[int]) -> Dict[str, Any]:
    """Series and flat baselines for one dataset, following the paper's conventions."""
    ablation = read_csv(results_dir / f"{stem}_ablation.csv")
    main = read_csv(results_dir / f"{stem}_main.csv")
    rows: List[Dict[str, Any]] = []
    series: Dict[str, Any] = {}
    for skey, (method, matcher, checkpoint, slabel, family) in SERIES.items():
        points: Dict[str, Dict[str, Optional[float]]] = {}
        for row in ablation:
            if not _match(row, method, matcher, checkpoint, CANDIDATE_BACKBONE):
                continue
            k = to_float(row.get("candidate_k"))
            if k is None or int(k) not in ks:
                continue
            metrics = _metrics(row)
            if metrics["top_5"] is None:
                continue
            points[str(int(k))] = metrics
            rows.append(
                {
                    "dataset": key,
                    "method_key": skey,
                    "method_label": slabel,
                    "family": family,
                    "matcher": matcher if matcher != "-" else None,
                    "backbone": CANDIDATE_BACKBONE,
                    "checkpoint": checkpoint,
                    "k": int(k),
                    "run_id": row.get("run_id") or None,
                    **metrics,
                }
            )
        if points:
            series[skey] = {"label": slabel, "family": family, "points": points}
    flats: Dict[str, Any] = {}
    for fkey, (method, backbone, checkpoint, flabel) in FLATS.items():
        source = main if method == "Cosine" else ablation
        candidates = [r for r in source if _match(r, method, None, checkpoint, backbone)]
        if not candidates:
            continue
        row = candidates[0]
        metrics = _metrics(row)
        if metrics["top_5"] is None:
            continue
        flats[fkey] = {"label": flabel, "backbone": backbone, **{m: metrics[m] for m in METRICS}}
        rows.append(
            {
                "dataset": key,
                "method_key": fkey,
                "method_label": flabel,
                "family": "baseline",
                "matcher": None,
                "backbone": backbone,
                "checkpoint": checkpoint,
                "k": None,
                "run_id": row.get("run_id") or None,
                **metrics,
            }
        )
    if "loma_finetuned" not in series or "loma_default" not in series:
        raise ValueError(f"{stem}: LoMa default/fine-tuned series missing from the paper results")
    return {"key": key, "label": label, "stem": stem, "ks": ks, "series": series, "flats": flats, "rows": rows}


def collect_adapt(results_dir: Path) -> List[Dict[str, Any]]:
    """Paper Table 2: descriptor branch, matching module, or both (CzechLynx closed, k=250)."""
    stem = "CzechLynx_split-time_closed"
    out: List[Dict[str, Any]] = []
    for matcher, mlabel in ADAPT_MATCHERS:
        short = "rdd" if matcher == "rdd-lightglue" else matcher
        pools = {
            "default": read_csv(results_dir / f"{stem}_ablation.csv"),
            "descriptor": read_csv(results_dir / f"{stem}_descriptor_{short}_ablation.csv"),
            "matcher": read_csv(results_dir / f"{stem}_ablation.csv"),
            "joint": read_csv(results_dir / f"{stem}_joint_{short}_ablation.csv"),
        }
        base: Optional[Dict[str, Optional[float]]] = None
        for vkey, checkpoint, vlabel in ADAPT_VARIANTS:
            hits = [
                r
                for r in pools[vkey]
                if _match(r, "Vismatch", matcher, checkpoint, CANDIDATE_BACKBONE)
                and to_float(r.get("candidate_k")) == MAIN_K
                and to_float(r.get("top_5")) is not None
            ]
            if not hits:
                raise ValueError(f"{stem}: no {checkpoint!r} row for {matcher} at k={MAIN_K}")
            metrics = _metrics(hits[0])
            if vkey == "default":
                base = metrics
            delta = (
                None
                if base is None or vkey == "default" or metrics["balanced_top_1"] is None
                else round(metrics["balanced_top_1"] - (base["balanced_top_1"] or 0.0), 6)
            )
            out.append(
                {
                    "matcher": matcher,
                    "matcher_label": mlabel,
                    "variant": vkey,
                    "variant_label": vlabel,
                    "k": MAIN_K,
                    "top_5": metrics["top_5"],
                    "balanced_top_1": metrics["balanced_top_1"],
                    "delta_balanced_top_1": delta,
                    "gpu_hours": ADAPT_GPU_HOURS.get((matcher, vkey)),
                    "run_id": hits[0].get("run_id") or None,
                }
            )
    return out


def collect_training_cost(results_dir: Path) -> Dict[str, Any]:
    rows = read_csv(results_dir / TRAINING_COST_CSV)
    series: Dict[str, Any] = {}
    flats: Dict[str, Any] = {}
    labels = {
        ("vismatch_loma_finetuned", "matcher"): ("loma_finetuned", "LoMa + WildMatch (ours)", "loma"),
        ("linear_probe", "all"): ("classifier_full", "Classifier, full (weighted)", "baseline"),
        ("linear_probe", "partial"): ("classifier_partial", "Classifier, partial (weighted)", "baseline"),
        ("linear_probe", "classifier"): ("classifier_frozen", "Classifier, frozen (weighted)", "baseline"),
    }
    for row in rows:
        method, mode = row.get("method", ""), row.get("train_mode", "")
        if method.endswith("(no training)"):
            key = "loma_default" if method.startswith("LoMa") else "cosine_megadescriptor"
            flats[key] = {
                "label": "LoMa (default)" if key == "loma_default" else "MegaDescriptor-L cosine",
                **{m: to_float(row.get(m)) for m in ("top_1", "top_5", "balanced_top_1")},
            }
            continue
        if (method, mode) not in labels:
            continue
        key, label, family = labels[(method, mode)]
        entry = series.setdefault(key, {"label": label, "family": family, "points": []})
        entry["points"].append(
            {
                "epoch": int(float(row["epoch"])),
                "gpu_hours": to_float(row["gpu_hours"]),
                **{m: to_float(row.get(m)) for m in ("top_1", "top_5", "balanced_top_1")},
            }
        )
    for entry in series.values():
        entry["points"].sort(key=lambda p: (p["gpu_hours"] or 0.0, p["epoch"]))
    if "loma_finetuned" not in series or "classifier_full" not in series:
        raise ValueError(f"{TRAINING_COST_CSV}: expected matcher and classifier curves")
    return {
        "dataset": "czechlynx",
        "k": MAIN_K,
        "cost": "training steps, GPU-hours on RTX 4090",
        "series": series,
        "flats": flats,
    }


# ------------------------------------------------------------------------ figures


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "svg.fonttype": "none",
            "figure.dpi": 110,
        }
    )
    return plt


def _theme(dark: bool) -> Dict[str, str]:
    if dark:
        return {"ink": "#e6e7e8", "muted": "#b3b5b8", "grid": "#3b3f44", "surface": "none", "ring": "#1e2227"}
    return {"ink": GREY_HEAD, "muted": GREY_TEXT, "grid": "#e7e8e9", "surface": "none", "ring": "#ffffff"}


def _apply_theme(ax, theme):
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelcolor=theme["muted"])
    ax.yaxis.label.set_color(theme["ink"])
    ax.xaxis.label.set_color(theme["ink"])
    ax.title.set_color(theme["ink"])
    ax.grid(True, axis="y", color=theme["grid"], lw=1.0)
    ax.set_axisbelow(True)


def _series_style(skey: str, theme) -> Dict[str, Any]:
    family = SERIES[skey][4]
    color = FAMILY_COLOR[family]
    marker = FAMILY_MARKER[family]
    if skey == "wildfusion":
        return dict(color=color, ls=":", marker=marker, mec=color, mfc=color, mew=1.5, lw=1.8)
    if skey.endswith("finetuned"):
        return dict(color=color, ls="-", marker=marker, mfc=color, mec=theme["ring"], mew=1.2, zorder=4)
    return dict(color=color, ls="--", marker=marker, mfc=theme["ring"], mec=color, mew=1.6, alpha=0.9, zorder=3)


def _pct(points: Dict[str, Dict[str, Optional[float]]], metric: str, ks: List[int]):
    xs, ys = [], []
    for k in ks:
        value = points.get(str(k), {}).get(metric)
        if value is not None:
            xs.append(k)
            ys.append(value * 100)
    return xs, ys


def _save(fig, out_dir: Path, name: str, dark: bool):
    path = out_dir / f"{name}{'_dark' if dark else ''}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=True)
    return path


def fig_small_multiples(
    plt, datasets: List[Dict[str, Any]], out_dir: Path, metric: str, name: str, series_keys: List[str], dark: bool
) -> Path:
    theme = _theme(dark)
    cols = 4
    rows_n = (len(datasets) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.9 * rows_n + 0.6), squeeze=False)
    handles: Dict[str, Any] = {}
    for idx, ds in enumerate(datasets):
        ax = axes[idx // cols][idx % cols]
        _apply_theme(ax, theme)
        for skey in series_keys:
            entry = ds["series"].get(skey)
            if not entry:
                continue
            xs, ys = _pct(entry["points"], metric, ds["ks"])
            (line,) = ax.plot(xs, ys, label=entry["label"], **_series_style(skey, theme))
            handles.setdefault(entry["label"], line)
        ax.set_xscale("log")
        ax.set_xticks(ds["ks"])
        ax.minorticks_off()
        ax.set_xticklabels(["1k" if k == 1000 else str(k) for k in ds["ks"]])
        ax.set_title(ds["label"], loc="left", fontweight="bold")
        if idx % cols == 0:
            ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        if idx // cols == rows_n - 1:
            ax.set_xlabel("candidate budget k")
    for idx in range(len(datasets), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    order = [SERIES[k][3] for k in series_keys if SERIES[k][3] in handles]
    leg = fig.legend(
        [handles[o] for o in order],
        order,
        loc="upper center",
        ncol=len(order),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    for text in leg.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = _save(fig, out_dir, name, dark)
    plt.close(fig)
    return path


def fig_gain_dumbbell(
    plt, datasets: List[Dict[str, Any]], out_dir: Path, dark: bool, family: str = "loma", k: int = MAIN_K
) -> Path:
    """Page figure: default -> fine-tuned at the main budget, one dumbbell per dataset."""
    theme = _theme(dark)
    color = FAMILY_COLOR[family]
    default_key, tuned_key = f"{family}_default", f"{family}_finetuned"
    metrics = ["top_5", "balanced_top_1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    names = [ds["label"] for ds in datasets]
    ypos = list(range(len(datasets)))[::-1]
    for ax, metric in zip(axes, metrics):
        _apply_theme(ax, theme)
        ax.grid(False)
        ax.grid(True, axis="x", color=theme["grid"], lw=1.0)
        for y, ds in zip(ypos, datasets):
            d = ds["series"].get(default_key, {}).get("points", {}).get(str(k), {}).get(metric)
            t = ds["series"].get(tuned_key, {}).get("points", {}).get(str(k), {}).get(metric)
            if d is None or t is None:
                continue
            d *= 100
            t *= 100
            ax.plot([d, t], [y, y], color=color, lw=2.0, alpha=0.55, zorder=2, solid_capstyle="round")
            ax.plot(d, y, marker="o", ms=8, mfc=theme["ring"], mec=color, mew=1.8, zorder=3, ls="none")
            ax.plot(t, y, marker="o", ms=8, mfc=color, mec=theme["ring"], mew=1.2, zorder=4, ls="none")
            ax.annotate(
                f"{t - d:+.1f}",
                (max(d, t), y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=8.5,
                color=theme["ink"],
            )
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.set_xlabel(f"{METRIC_LABELS[metric]} (%) at k = {k}")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.set_xlim(0, 100)
        for spine in ("left",):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    hollow = plt.Line2D(
        [], [], marker="o", ms=8, mfc=theme["ring"], mec=color, mew=1.8, ls="none", label=SERIES[default_key][3]
    )
    filled = plt.Line2D(
        [], [], marker="o", ms=8, mfc=color, mec=theme["ring"], mew=1.2, ls="none", label=SERIES[tuned_key][3]
    )
    leg = fig.legend(handles=[filled, hollow], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    for text in leg.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = _save(fig, out_dir, f"gain_{family}_k{k}", dark)
    plt.close(fig)
    return path


def fig_adapt_cost(plt, adapt: List[Dict[str, Any]], out_dir: Path, dark: bool) -> Path:
    """Page figure: training cost against the change in balanced top-1 for each fine-tuned part."""
    theme = _theme(dark)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _apply_theme(ax, theme)
    ax.axhline(0, color=theme["muted"], lw=1.0, zorder=1)
    marker_for = {"descriptor": "D", "matcher": "o", "joint": "s"}
    # Label placement per variant: cheap points label to the right, expensive ones to
    # the left, so no label crosses the title, the axis or another label.
    placement = {"matcher": (12, "left"), "descriptor": (-12, "right"), "joint": (-12, "right")}
    handles: Dict[str, Any] = {}
    for row in adapt:
        if row["variant"] == "default" or row["gpu_hours"] is None or row["delta_balanced_top_1"] is None:
            continue
        family = "rdd" if row["matcher"] == "rdd-lightglue" else "loma"
        color = FAMILY_COLOR[family]
        x, y = row["gpu_hours"], row["delta_balanced_top_1"] * 100
        (pt,) = ax.plot(
            x,
            y,
            marker=marker_for[row["variant"]],
            ms=10,
            mfc=color,
            mec=theme["ring"],
            mew=1.4,
            ls="none",
            zorder=4,
            label=row["matcher_label"],
        )
        handles.setdefault(row["matcher_label"], pt)
        dx, ha = placement[row["variant"]]
        short = row["matcher_label"].split("-")[0]
        ax.annotate(
            f"{short} {row['variant_label'].replace(' (ours)', '')}: {y:+.1f} pt, {x:g} GPU-h",
            (x, y),
            xytext=(dx, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8,
            color=theme["ink"],
        )
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlim(3.2, 180)
    ax.set_ylim(-9.5, 5.5)
    ax.set_xlabel("training cost (GPU-hours, log scale)")
    ax.set_ylabel("change in balanced top-1 vs default (pt)")
    ax.set_title("What to adapt: CzechLynx closed, k = 250", loc="left", fontweight="bold")
    shape_handles = [
        plt.Line2D([], [], marker=marker_for[v], ms=9, mfc=theme["muted"], mec=theme["ring"], ls="none", label=lab)
        for v, lab in (("matcher", "matching module (ours)"), ("descriptor", "descriptor branch"), ("joint", "both"))
    ]
    leg = fig.legend(
        handles=list(handles.values()) + shape_handles,
        loc="lower center",
        frameon=False,
        ncol=5,
        bbox_to_anchor=(0.5, -0.04),
    )
    for text in leg.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = _save(fig, out_dir, "adapt_cost_k250", dark)
    plt.close(fig)
    return path


def fig_unseen(plt, unseen: Dict[str, Any], out_dir: Path, dark: bool) -> Path:
    theme = _theme(dark)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    handles: Dict[str, Any] = {}
    for ax, metric in zip(axes, ["top_5", "balanced_top_1"]):
        _apply_theme(ax, theme)
        for skey in ("wildfusion", "loma_default", "loma_finetuned", "rdd_default", "rdd_finetuned"):
            entry = unseen["series"].get(skey)
            if not entry:
                continue
            xs, ys = _pct(entry["points"], metric, unseen["ks"])
            (line,) = ax.plot(xs, ys, label=entry["label"], **_series_style(skey, theme))
            handles.setdefault(entry["label"], line)
        for fkey, ls in (("cosine_megadescriptor", (0, (1, 1.5))), ("cosine_dinov3", (0, (4, 2)))):
            flat = unseen["flats"].get(fkey)
            if flat and flat.get(metric) is not None:
                (line,) = ax.plot(
                    unseen["ks"],
                    [flat[metric] * 100] * len(unseen["ks"]),
                    color=theme["muted"],
                    ls=ls,
                    lw=1.5,
                    marker="",
                    label=flat["label"],
                    zorder=1,
                )
                handles.setdefault(flat["label"], line)
        ax.set_xscale("log")
        ax.set_xticks(unseen["ks"])
        ax.set_xticklabels([str(k) for k in unseen["ks"]])
        ax.minorticks_off()
        ax.set_xlabel("candidate budget k (160 = whole gallery)")
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
    order = [lab for lab in handles]
    leg = fig.legend(
        [handles[o] for o in order], order, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08)
    )
    for text in leg.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = _save(fig, out_dir, "unseen_vs_k", dark)
    plt.close(fig)
    return path


def fig_training_cost(plt, cost: Dict[str, Any], out_dir: Path, dark: bool) -> Path:
    theme = _theme(dark)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    handles: Dict[str, Any] = {}
    clf_style = {"classifier_full": "-", "classifier_partial": "--", "classifier_frozen": "-."}
    for ax, metric in zip(axes, ["top_5", "balanced_top_1"]):
        _apply_theme(ax, theme)
        cos = cost["flats"].get("cosine_megadescriptor")
        if cos and cos.get(metric) is not None:
            (line,) = ax.plot(
                [0, 10.5],
                [cos[metric] * 100] * 2,
                color=theme["muted"],
                ls=(0, (1, 1.5)),
                lw=1.5,
                label=cos["label"],
                zorder=1,
            )
            handles.setdefault(cos["label"], line)
        for key, ls in clf_style.items():
            entry = cost["series"].get(key)
            if not entry:
                continue
            xs = [p["gpu_hours"] for p in entry["points"]]
            ys = [p[metric] * 100 for p in entry["points"]]
            (line,) = ax.plot(xs, ys, color=theme["muted"], ls=ls, lw=1.6, label=entry["label"], zorder=2)
            ax.plot(xs[-1], ys[-1], marker="o", ms=6, mfc=theme["muted"], mec=theme["ring"], zorder=2)
            handles.setdefault(entry["label"], line)
        dflt = cost["flats"].get("loma_default")
        if dflt and dflt.get(metric) is not None:
            (line,) = ax.plot(
                [0, 10.5],
                [dflt[metric] * 100] * 2,
                color=BLUE,
                ls="--",
                lw=1.6,
                alpha=0.9,
                label=dflt["label"],
                zorder=2,
            )
            handles.setdefault(dflt["label"], line)
        ours = cost["series"]["loma_finetuned"]
        xs = [p["gpu_hours"] for p in ours["points"]]
        ys = [p[metric] * 100 for p in ours["points"]]
        (line,) = ax.plot(
            xs,
            ys,
            color=BLUE,
            ls="-",
            marker="o",
            ms=6,
            mfc=BLUE,
            mec=theme["ring"],
            mew=1.2,
            lw=2.2,
            label=ours["label"],
            zorder=4,
        )
        handles.setdefault(ours["label"], line)
        ax.annotate(
            f"{ys[-1]:.1f}",
            (xs[-1], ys[-1]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=theme["ink"],
            fontweight="bold",
        )
        full = cost["series"].get("classifier_full")
        if full:
            fx, fy = full["points"][-1]["gpu_hours"], full["points"][-1][metric] * 100
            ax.annotate(
                f"{fy:.1f}",
                (fx, fy),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=theme["ink"],
            )
        ax.set_xlim(-0.2, 10.6)
        ax.set_xticks(range(0, 11, 2))
        ax.set_ylim(0, None)
        ax.set_xlabel("training cost (GPU-hours)")
        ax.set_ylabel(f"{METRIC_LABELS[metric]} (%)")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
    order = [lab for lab in handles]
    leg = fig.legend(
        [handles[o] for o in order], order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1)
    )
    for text in leg.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = _save(fig, out_dir, "training_cost_k250", dark)
    plt.close(fig)
    return path


def render_figures(
    datasets: List[Dict[str, Any]],
    unseen: Dict[str, Any],
    adapt: List[Dict[str, Any]],
    cost: Dict[str, Any],
    out_dir: Path,
) -> List[Path]:
    plt = _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for dark in (False, True):
        written.append(
            fig_small_multiples(
                plt,
                datasets,
                out_dir,
                "top_5",
                "main_top5_vs_k",
                ["loma_finetuned", "loma_default", "wildfusion"],
                dark,
            )
        )
        written.append(
            fig_small_multiples(
                plt,
                datasets,
                out_dir,
                "balanced_top_1",
                "main_balanced_top1_vs_k",
                ["loma_finetuned", "loma_default", "wildfusion"],
                dark,
            )
        )
        written.append(
            fig_small_multiples(
                plt, datasets, out_dir, "top_5", "rdd_top5_vs_k", ["rdd_finetuned", "rdd_default", "wildfusion"], dark
            )
        )
        written.append(fig_gain_dumbbell(plt, datasets, out_dir, dark))
        written.append(fig_adapt_cost(plt, adapt, out_dir, dark))
        written.append(fig_unseen(plt, unseen, out_dir, dark))
        written.append(fig_training_cost(plt, cost, out_dir, dark))
    return written


# ----------------------------------------------------------------------------- main


def _check_no_private_paths(payload: Any) -> None:
    text = json.dumps(payload)
    for needle in ("/shared/", "/home/"):
        if needle in text:
            raise ValueError(f"exported data contains a private path fragment {needle!r}")


def export(paper_repo: Path, results_dir: Path, out_dir: Path, figures_dir: Optional[Path]) -> Dict[str, Any]:
    datasets = [collect_dataset(results_dir, stem, key, label, KS) for stem, key, label in DATASETS]
    unseen = collect_dataset(results_dir, UNSEEN[0], UNSEEN[1], UNSEEN[2], UNSEEN_KS)
    adapt = collect_adapt(results_dir)
    cost = collect_training_cost(results_dir)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    commit = paper_commit(paper_repo)

    rows = [row for ds in datasets for row in ds["rows"]] + unseen["rows"]
    results = {
        "generated_at": generated_at,
        "paper_commit": commit,
        "main_k": MAIN_K,
        "metrics": METRICS,
        "metric_labels": METRIC_LABELS,
        "datasets": [{"key": ds["key"], "label": ds["label"], "ks": ds["ks"]} for ds in datasets + [unseen]],
        "rows": rows,
    }
    curves = {
        "generated_at": generated_at,
        "paper_commit": commit,
        "metrics": METRICS,
        "metric_labels": METRIC_LABELS,
        "series_order": list(SERIES),
        "families": {k: v[4] for k, v in SERIES.items()},
        "datasets": {ds["key"]: {k: ds[k] for k in ("label", "ks", "series", "flats")} for ds in datasets},
        "unseen": {k: unseen[k] for k in ("label", "ks", "series", "flats")},
    }
    adapt_payload = {
        "generated_at": generated_at,
        "paper_commit": commit,
        "dataset": "czechlynx",
        "k": MAIN_K,
        "gpu_hours_note": "training steps at epoch 299, GPU-hours on RTX 4090; from the paper's make_tables.py",
        "rows": adapt,
    }
    cost_payload = {"generated_at": generated_at, "paper_commit": commit, **cost}

    sources = sorted(p for p in results_dir.glob("*.csv"))
    manifest = {
        "generated_at": generated_at,
        "paper_commit": commit,
        "paper_results_dir": str(results_dir.relative_to(paper_repo))
        if results_dir.is_relative_to(paper_repo)
        else results_dir.name,
        "conventions": {
            "datasets": [d[1] for d in DATASETS],
            "main_k": MAIN_K,
            "budgets": KS,
            "unseen_budgets": UNSEEN_KS,
            "classifier_weighting": "weighted only",
            "matcher_checkpoints": "default and matcher-only fine-tuned; descriptor and joint only in adapt.json",
            "excluded": ["BelugaID", "split-time_open", "unweighted and unknown-weighting probes"],
        },
        "sources": [{"file": p.name, "sha256": sha256_file(p)} for p in sources],
    }

    for payload in (results, curves, adapt_payload, cost_payload, manifest):
        _check_no_private_paths(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "results.json": results,
        "curves.json": curves,
        "adapt.json": adapt_payload,
        "training_cost.json": cost_payload,
        "manifest.json": manifest,
    }
    for name, payload in outputs.items():
        (out_dir / name).write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    written: List[Path] = []
    if figures_dir is not None:
        written = render_figures(datasets, unseen, adapt, cost, figures_dir)
    return {
        "outputs": sorted(outputs),
        "figures": [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in written],
        "rows": len(rows),
        "paper_commit": commit,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=DEFAULT_PAPER_REPO,
        help="clone of the paper repository (default: sibling ECIR-Animal-ReID-Paper)",
    )
    parser.add_argument(
        "--paper-results", type=Path, default=None, help="results directory (default: <paper-repo>/results)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON output directory (default: docs/data)")
    parser.add_argument(
        "--figures",
        type=Path,
        default=DEFAULT_FIGURES,
        help="static figure output directory (default: docs/assets/figures/page)",
    )
    parser.add_argument("--no-figures", action="store_true", help="write JSON only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    results_dir = args.paper_results or (args.paper_repo / "results")
    if not results_dir.is_dir():
        print(f"error: paper results directory not found: {results_dir}", file=sys.stderr)
        return 2
    summary = export(args.paper_repo, results_dir, args.out, None if args.no_figures else args.figures)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
