"""Training and evaluation resize the same way (integration plan, parity level L1).

The trainers keep their own resize helpers; Vismatch evaluation uses
`resize_long_side_divisible` (`lynx_finetuning_v1`: /32, `lynx_loma_finetuning_v1`: /14). Both must
give bit-identical tensors, or fine-tuned matchers are evaluated on inputs they were not trained on.
The RDD trainer has no lower clamp (an image whose short side scales below 32 px gives a zero-sized
tensor and fails); evaluation clamps to one block. For those shapes the test pins that the trainer
fails rather than training on a different size.
"""

import pytest
import torch

from wildmatch.matcher_finetune.train_common import resize_long_side as rdd_train_resize
from wildmatch.matcher_finetune.train_loma_matches import resize_long_side as loma_train_resize
from wildmatch.matchers.vismatch_preprocessing import resize_long_side_divisible

SHAPES = [(100, 50), (480, 640), (640, 480), (1080, 1920), (511, 512), (512, 512), (333, 777), (2000, 129)]
TARGETS = [512, 448, 1024]


def _image(height: int, width: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(height * 10_000 + width)
    return torch.rand(1, 3, height, width, generator=g)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "train_resize, divisor", [(rdd_train_resize, 32), (loma_train_resize, 14)], ids=["rdd", "loma"]
)
def test_training_resize_is_bit_identical_to_evaluation(train_resize, divisor, shape, target):
    image = _image(*shape)
    if divisor == 32 and int(min(shape) * target / max(shape)) < 32:
        with pytest.raises(RuntimeError):
            train_resize(image, target)
        return
    expected, _source, processed = resize_long_side_divisible(image[0], target, divisible_by=divisor)
    actual = train_resize(image, target)[0]
    assert tuple(actual.shape[-2:]) == processed
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
