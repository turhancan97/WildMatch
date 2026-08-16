import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.loma_backend import resize_image


def test_loma_resize_aligns_both_dimensions_to_dinov2_patch_size():
    image = torch.zeros(1, 3, 240, 960)
    resized = resize_image(image, 512)

    assert resized.shape[-1] == 504
    assert resized.shape[-2] % 14 == 0
    assert resized.shape[-1] % 14 == 0
