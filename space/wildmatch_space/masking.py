"""SAM 3 background removal for uploaded photos (GPU).

Same call sequence as ``wildmatch.data.prepare.sam3_masks.run_segmentation``, which built the masks the
paper's WildlifeReID-10k reruns and SalamanderID2025 use: each prompt of the species is tried at the main
threshold, then at the fallback threshold, until SAM 3 detects something; several detections are merged with
``wildmatch.data.prepare.sam3_masks.merge_instances``. When nothing is detected the whole photo is kept and
the caller tells the visitor.

The checkpoint ``sam3.pt`` comes from the gated Hub repository ``facebook/sam3`` (``HF_TOKEN`` of an account
that accepted its licence) unless ``WILDMATCH_SAM3_CHECKPOINT`` names a local file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

THRESHOLD = 0.5
FALLBACK_THRESHOLD = 0.25

_processor = None
_unavailable: Optional[str] = None  # why SAM 3 could not be loaded, if it could not


@dataclass
class MaskResult:
    foreground: np.ndarray  # (H, W) bool; all True when nothing was detected
    masked: Image.Image  # background set to black, the matcher input
    prompt: Optional[str]  # the prompt that detected the animal, None for the whole-photo fallback
    detections: int


def _checkpoint() -> str:
    local = os.environ.get("WILDMATCH_SAM3_CHECKPOINT")
    if local:
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download("facebook/sam3", "sam3.pt", token=os.environ.get("HF_TOKEN"))


def unavailable() -> Optional[str]:
    """Why SAM 3 is not available (for example a gated-access request not yet approved), or None."""
    return _unavailable


def load() -> bool:
    """Build SAM 3 now; on failure remember the reason so the app runs without background removal."""
    global _unavailable
    try:
        processor()
        _unavailable = None
    except Exception as error:  # the gated checkpoint (401/403), a missing package or CUDA
        _unavailable = f"{type(error).__name__}: {str(error).strip().splitlines()[-1][:200]}"
    return _unavailable is None


def processor():
    """The SAM 3 image processor, built once on CUDA."""
    global _processor
    if _unavailable is not None:
        raise RuntimeError(f"SAM 3 is not available: {_unavailable}")
    if _processor is None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(checkpoint_path=_checkpoint(), load_from_HF=False, device="cuda", eval_mode=True)
        _processor = Sam3Processor(model, device="cuda", confidence_threshold=THRESHOLD)
    return _processor


def apply_mask(image: Image.Image, foreground: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB")).copy()
    array[~np.asarray(foreground, dtype=bool)] = 0
    return Image.fromarray(array)


def segment(image: Image.Image, prompts: Sequence[str], merge: str) -> MaskResult:
    """Mask the animal in ``image``; the first prompt and threshold that detect anything win."""
    import torch

    from wildmatch.data.prepare.sam3_masks import merge_instances

    image = image.convert("RGB")
    proc = processor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = proc.set_image(image, state={})
        for threshold in (THRESHOLD, FALLBACK_THRESHOLD):
            proc.set_confidence_threshold(threshold)
            for prompt in prompts:
                proc.reset_all_prompts(base)
                state = proc.set_text_prompt(prompt=prompt, state=base)
                masks, scores = state.get("masks"), state.get("scores")
                if masks is None or scores is None:
                    continue
                mask, count, _ = merge_instances(
                    masks.detach().cpu().numpy(), scores.detach().float().cpu().numpy(), merge
                )
                if mask is not None and mask.any():
                    return MaskResult(mask, apply_mask(image, mask), prompt, count)
    whole = np.ones((image.height, image.width), dtype=bool)
    return MaskResult(whole, image, None, 0)
