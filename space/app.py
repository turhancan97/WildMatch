"""WildMatch demo: match two photos of an animal, or find a lynx in a gallery of individuals.

Runs on Hugging Face ZeroGPU (``spaces.GPU``) or any machine with the ``wildmatch`` package, SAM 3 and Gradio.
Uploaded photos are processed in memory and never stored.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Installed without their dependencies, which requirements.txt already pins: plain pip would add vismatch's
# `uniception` (excluded in the repository's uv lock) and SAM 3's numpy<2 pin (SAM 3 runs on numpy 2).
NO_DEPS = {
    "vismatch": "vismatch @ git+https://github.com/gmberton/vismatch.git@4a743b75749a3770af59d275483ed341dea51ff0",
    "sam3": "sam3 @ git+https://github.com/facebookresearch/sam3.git@f6e51f59500a87c576c2df2323ce56b9fd7a12de",
}
for module, requirement in NO_DEPS.items():
    if importlib.util.find_spec(module) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", requirement])

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from wildmatch_space import masking, matching, retrieval  # noqa: E402
from wildmatch_space.drawing import match_figure  # noqa: E402
from wildmatch_space.species import DEFAULT_SPECIES, MATCHERS, SPECIES  # noqa: E402

try:
    import spaces

    gpu = spaces.GPU
except ImportError:  # local runs: no ZeroGPU

    def gpu(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


HERE = Path(__file__).resolve().parent
EXAMPLES = HERE / "examples"
WEIGHTS = ["Fine-tuned (WildMatch)", "Default"]
GALLERIES = {"CzechLynx: 44 lynx the matcher never saw": "czechlynx_unseen", "Synthetic lynx renders": "synthetic"}
GALLERY_CHECKPOINT = "czechlynx_open"  # fine-tuned without any of the unseen-split individuals
LINKS = (
    "[Paper (arXiv)](https://arxiv.org/abs/2610.07384) · [Project page](https://wildmatch.gmum.net) · "
    "[Code](https://github.com/turhancan97/WildMatch) · "
    "[Checkpoints](https://huggingface.co/turhancan97/wildmatch-checkpoints)"
)


def _prepare(image: Image.Image, species, use_sam3: bool) -> Tuple[masking.MaskResult, str]:
    image = image.convert("RGB")
    if not use_sam3:
        whole = np.ones((image.height, image.width), dtype=bool)
        return masking.MaskResult(whole, image, None, 0), "background kept (SAM 3 off)"
    result = masking.segment(image, species.prompts, species.merge)
    if result.prompt is None:
        return result, "**SAM 3 found no animal; the whole photo was used.**"
    return result, f'SAM 3 prompt "{result.prompt}"'


def _matcher(matcher_label: str, weights: str, dataset: str):
    name = MATCHERS[matcher_label]
    return matching.matcher(name, dataset if weights.startswith("Fine-tuned") else None)


@gpu(duration=60)
def match_pair(left, right, species_label, matcher_label, weights, use_sam3, lines):
    if left is None or right is None:
        raise gr.Error("Add two photos.")
    species = SPECIES[species_label]
    (mask_l, note_l), (mask_r, note_r) = _prepare(left, species, use_sam3), _prepare(right, species, use_sam3)
    backend = _matcher(matcher_label, weights, species.dataset)
    f_l, f_r = matching.features(backend, mask_l.masked), matching.features(backend, mask_r.masked)
    result = matching.match(backend, f_l, f_r, left.size, right.size)
    figure, drawn = match_figure(
        left.convert("RGB"),
        right.convert("RGB"),
        (mask_l.foreground, mask_r.foreground),
        result.kpts_left,
        result.kpts_right,
        result.confidences,
        lines=int(lines),
    )
    which = f"fine-tuned on {species_label}" if weights.startswith("Fine-tuned") else "default weights"
    summary = (
        f"**Match score {result.score:.3f}** · {result.match_count} matches ({drawn} drawn) · "
        f"{matcher_label}, {which}\n\nLeft: {note_l}. Right: {note_r}."
    )
    return figure, summary


@gpu(duration=120)
def find_lynx(query, gallery_label, matcher_label, weights, use_sam3, k):
    if query is None:
        raise gr.Error("Add a photo of a lynx.")
    gallery = retrieval.load(GALLERIES[gallery_label])
    species = SPECIES[DEFAULT_SPECIES]
    mask_q, note = _prepare(query, species, use_sam3)
    shortlist = retrieval.candidates(retrieval.embed([mask_q.masked])[0], gallery, int(k))
    backend = _matcher(matcher_label, weights, GALLERY_CHECKPOINT)
    f_q = matching.features(backend, mask_q.masked)
    results = []
    for index in shortlist:
        f_g = matching.features(backend, gallery.masked(int(index)))
        results.append((int(index), matching.match(backend, f_q, f_g, query.size, gallery.raw(int(index)).size), f_g))
    ranked = retrieval.order_by_score(np.array([r[0] for r in results]), np.array([r[1].score for r in results]))
    by_index = {index: (match, f) for index, match, f in results}
    top = []
    for rank, index in enumerate(ranked[:5], 1):
        identity = gallery.metadata.iloc[int(index)]["identity"]
        top.append((gallery.raw(int(index)), f"{rank}. {identity} · score {by_index[int(index)][0].score:.3f}"))
    best = int(ranked[0])
    best_match = by_index[best][0]
    figure, drawn = match_figure(
        query.convert("RGB"),
        gallery.raw(best),
        (mask_q.foreground, gallery.foreground(best)),
        best_match.kpts_left,
        best_match.kpts_right,
        best_match.confidences,
    )
    identity = gallery.metadata.iloc[best]["identity"]
    images = int((gallery.metadata["identity"] == identity).sum())
    which = (
        "fine-tuned (CzechLynx, without these individuals)" if weights.startswith("Fine-tuned") else "default weights"
    )
    summary = (
        f"**Best match: {identity}** (score {best_match.score:.3f}, {best_match.match_count} matches; "
        f"{images} photo(s) of this lynx in the gallery).\n\n"
        + (
            f"{matcher_label} ({which}) scored all {len(shortlist)} gallery photos by feature matching."
            if len(shortlist) == len(gallery.metadata)
            else f"MegaDescriptor-L picked the {len(shortlist)} most similar of {len(gallery.metadata)} gallery photos; "
            f"{matcher_label} ({which}) scored each by feature matching."
        )
        + f" Query: {note}."
    )
    return top, figure, summary


def _examples(pattern: str) -> List[str]:
    return sorted(str(p) for p in EXAMPLES.glob(pattern))


def _query_examples() -> List[list]:
    table = EXAMPLES / "examples.csv"
    lynx = pd.read_csv(table)["file"].tolist() if table.is_file() else []
    files = [str(EXAMPLES / f) for f in lynx if f.startswith("lynx_query")]
    return [[f, list(GALLERIES)[0]] for f in files] + [[f, list(GALLERIES)[1]] for f in _examples("synthetic_query_*")]


def build() -> gr.Blocks:
    with gr.Blocks(title="WildMatch") as demo:
        gr.Markdown(
            "# WildMatch: matching individual animals by their markings\n"
            f"{LINKS}\n\n"
            "WildMatch adapts pretrained feature matchers (LoMa, RDD-LightGlue) to individual animals using only "
            "identity labels. Pick the species, compare the default matcher with its fine-tuned WildMatch "
            "checkpoint, and see which spots, stripes and markings it matches. SAM 3 removes the background "
            "first, as in the paper: matching runs on the masked photos, and the lines are drawn on the originals."
        )
        with gr.Tab("Match two photos"):
            with gr.Row():
                left = gr.Image(type="pil", label="Photo 1", height=320)
                right = gr.Image(type="pil", label="Photo 2", height=320)
            with gr.Row():
                species = gr.Dropdown(list(SPECIES), value=DEFAULT_SPECIES, label="Species (selects the checkpoint)")
                matcher = gr.Radio(list(MATCHERS), value="LoMa", label="Matcher")
                weights = gr.Radio(WEIGHTS, value=WEIGHTS[0], label="Weights")
            with gr.Row():
                use_sam3 = gr.Checkbox(value=True, label="Remove background with SAM 3")
                lines = gr.Slider(5, 50, value=20, step=5, label="Lines drawn")
            run = gr.Button("Match", variant="primary")
            figure = gr.Image(label="Correspondences", type="pil")
            summary = gr.Markdown()
            pair = _examples("lynx_pair_*.jpg")
            if len(pair) == 2:
                gr.Examples(
                    [[pair[0], pair[1], DEFAULT_SPECIES]],
                    [left, right, species],
                    label="Example: one lynx, day and night",
                )
            run.click(match_pair, [left, right, species, matcher, weights, use_sam3, lines], [figure, summary])
        with gr.Tab("Find this lynx"):
            with gr.Row():
                query = gr.Image(type="pil", label="Lynx photo", height=320)
                with gr.Column():
                    gallery = gr.Dropdown(list(GALLERIES), value=list(GALLERIES)[0], label="Gallery")
                    matcher_r = gr.Radio(list(MATCHERS), value="LoMa", label="Matcher")
                    weights_r = gr.Radio(WEIGHTS, value=WEIGHTS[0], label="Weights")
                    use_sam3_r = gr.Checkbox(value=True, label="Remove background with SAM 3")
                    k = gr.Slider(10, 160, value=160, step=10, label="Candidates matched (k; 160 = the whole gallery)")
            gr.Markdown(
                "These 44 lynx are the paper's unseen-individual test: none of them was in the fine-tuning data. "
                "Matching every gallery photo, the paper measures top-1 33.5 % and top-5 46.1 % for fine-tuned LoMa "
                "(default LoMa 29.0 % / 40.4 %), so the right lynx is often not first. Matching only the k most "
                "similar photos by MegaDescriptor-L is faster but can leave the right lynx out of the list."
            )
            find = gr.Button("Find", variant="primary")
            top = gr.Gallery(label="Top 5 gallery photos", columns=5, height=220)
            best = gr.Image(label="Query and best match", type="pil")
            summary_r = gr.Markdown()
            examples = _query_examples()
            if examples:
                gr.Examples(examples, [query, gallery], label="Examples (photos not in the gallery)")
            find.click(find_lynx, [query, gallery, matcher_r, weights_r, use_sam3_r, k], [top, best, summary_r])
        gr.Markdown(
            "Fine-tuned checkpoints: CC BY-NC 4.0, non-commercial use only. MegaDescriptor-L, which picks the "
            "gallery candidates, was trained on six of the paper's eight datasets (not on CzechLynx). Background "
            "removal uses SAM 3 (Meta, SAM License). Gallery and example photos: CzechLynx dataset (Picek et al.), "
            "CC BY 4.0. Uploaded photos are processed in memory and not stored."
        )
    return demo


def preload() -> None:
    """On ZeroGPU, build the models used most at start-up, outside any GPU call, so they stay resident."""
    masking.processor()
    retrieval.embedder()
    for name in MATCHERS.values():
        matching.matcher(name, None, keep=True)
        matching.matcher(name, GALLERY_CHECKPOINT, keep=True)


if matching.on_zero_gpu() or os.environ.get("WILDMATCH_SPACE_PRELOAD") == "1":
    preload()
demo = build()

if __name__ == "__main__":
    demo.launch()
