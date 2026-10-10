"""Match drawings for the Space: masked inputs are matched, the raw photos are drawn.

The helpers are copied from ``paper/figures/plot_match_examples.py`` (the project page's match figures)
because that module loads the dataset registry and the cluster path profile at import time. Keep the two
in step when the page style changes.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LINE_COLOR = "#00d9ff"  # cyan: contrasts with yellow spots, grass, water and pink IR frames
PADDING_THRESHOLD = 12  # max(RGB) at or below this counts as black source padding


def processed_to_raw_pixels(points: np.ndarray, processed_hw: Sequence[int], raw_hw: Sequence[int]) -> np.ndarray:
    """Pixel coordinates on the matcher input (``FrameFeatures.image_size``) -> the raw photo.

    The matcher input is the raw photo resampled without cropping, so both axes scale independently,
    with the pixel-centre convention x_raw = (x + 0.5) W / w - 0.5.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    (h, w), (height, width) = (float(v) for v in processed_hw), (float(v) for v in raw_hw)
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] + 0.5) * width / w - 0.5
    out[:, 1] = (points[:, 1] + 0.5) * height / h - 0.5
    return out


def padding_box(
    array: np.ndarray, threshold: int = PADDING_THRESHOLD, fraction: float = 0.98
) -> Tuple[int, int, int, int]:
    """(left, top, right, bottom) without the near-black bands stored in the source file."""
    dark = np.asarray(array).max(axis=2) <= threshold
    rows, cols = dark.mean(axis=1) >= fraction, dark.mean(axis=0) >= fraction

    def span(flags: np.ndarray) -> Tuple[int, int]:
        start, stop = 0, len(flags)
        while start < stop and flags[start]:
            start += 1
        while stop > start and flags[stop - 1]:
            stop -= 1
        return (start, stop) if stop > start else (0, len(flags))

    left, right = span(cols)
    top, bottom = span(rows)
    return left, top, right, bottom


def spread_selection(
    p0: np.ndarray, p1: np.ndarray, confidences: np.ndarray, count: int, min_distance: float
) -> np.ndarray:
    """Up to ``count`` strong matches whose endpoints are at least ``min_distance`` apart."""
    from wildmatch.evaluate.ranking import stable_rank_1d

    order = stable_rank_1d(np.asarray(confidences, dtype=np.float64))
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    distance = float(min_distance)
    while True:
        kept: List[int] = []
        for index in order:
            if len(kept) == count:
                break
            if kept:
                d0 = np.hypot(*(p0[kept] - p0[index]).T).min()
                d1 = np.hypot(*(p1[kept] - p1[index]).T).min()
                if min(d0, d1) < distance:
                    continue
            kept.append(int(index))
        if len(kept) >= min(count, len(order)) or distance <= 1e-9:
            return np.asarray(kept, dtype=np.int64)
        distance = distance / 2 if distance > 0.5 else 0.0


def dim_background(
    image: Image.Image, foreground: np.ndarray, brightness: float = 0.55, saturation: float = 0.45, feather: float = 1.0
) -> Image.Image:
    """Darken and desaturate the background, leaving the animal untouched."""
    from scipy import ndimage

    mask = np.asarray(foreground, dtype=bool)
    radius = max(2, round(0.004 * max(mask.shape)))
    mask = ndimage.binary_closing(mask, iterations=radius)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_dilation(mask, iterations=radius)
    alpha = ndimage.gaussian_filter(mask.astype(np.float32), sigma=radius * feather)[..., None]
    rgb = np.asarray(image, dtype=np.float32)
    gray = rgb.mean(axis=2, keepdims=True)
    background = (gray + saturation * (rgb - gray)) * brightness
    return Image.fromarray(np.clip(alpha * rgb + (1 - alpha) * background, 0, 255).astype(np.uint8))


def crop_box_around(size: Sequence[int], points: np.ndarray, aspect: float) -> Tuple[int, int, int, int]:
    """Largest ``aspect`` (w/h) crop of an image of ``size`` (w, h), centred on the points."""
    width, height = int(size[0]), int(size[1])
    crop_w, crop_h = (width, round(width / aspect)) if width / height <= aspect else (round(height * aspect), height)
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cx, cy = ((points.min(0) + points.max(0)) / 2.0) if len(points) else (width / 2.0, height / 2.0)
    left = int(round(min(max(cx - crop_w / 2.0, 0), width - crop_w)))
    top = int(round(min(max(cy - crop_h / 2.0, 0), height - crop_h)))
    return left, top, left + crop_w, top + crop_h


def _prepare_photo(
    image: Image.Image, foreground: Optional[np.ndarray], points: np.ndarray, aspect: float, height: int, dim: float
):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if dim > 0 and foreground is not None:
        image = dim_background(image, foreground, brightness=1 - dim, saturation=1 - dim, feather=6.0)
    l, t, r, b = padding_box(np.asarray(image))
    inside = (points[:, 0] >= l) & (points[:, 0] < r) & (points[:, 1] >= t) & (points[:, 1] < b)
    image, points = image.crop((l, t, r, b)), points - np.array([l, t], dtype=np.float64)
    l, t, r, b = crop_box_around(image.size, points[inside], aspect)
    inside &= (points[:, 0] >= l) & (points[:, 0] < r) & (points[:, 1] >= t) & (points[:, 1] < b)
    image, points = image.crop((l, t, r, b)), points - np.array([l, t], dtype=np.float64)
    scale = height / image.height
    image = image.resize((round(image.width * scale), height), Image.Resampling.LANCZOS)
    return image, points * scale, inside


def match_figure(
    left: Image.Image,
    right: Image.Image,
    foregrounds: Sequence[Optional[np.ndarray]],
    kpts_left: np.ndarray,
    kpts_right: np.ndarray,
    confidences: np.ndarray,
    lines: int = 20,
    aspect: float = 1.25,
    dim: float = 0.35,
    height: int = 480,
    gutter: int = 8,
    spacing: float = 0.08,
) -> Tuple[Image.Image, int]:
    """Both photos side by side with up to ``lines`` strong, spread-out correspondences.

    Returns the rendered RGB image and the number of lines drawn. Photos are cropped to ``aspect``
    around their matched points and the background is lightly dimmed with the mask.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.collections import LineCollection

    l_img, l_pts, l_in = _prepare_photo(left, foregrounds[0], kpts_left, aspect, height, dim)
    r_img, r_pts, r_in = _prepare_photo(right, foregrounds[1], kpts_right, aspect, height, dim)
    inside = l_in & r_in
    l_pts, r_pts = l_pts[inside], r_pts[inside] + np.array([l_img.width + gutter, 0.0])
    confidences = np.asarray(confidences, dtype=np.float64)[inside]
    canvas = Image.new("RGB", (l_img.width + gutter + r_img.width, height), (255, 255, 255))
    canvas.paste(l_img, (0, 0))
    canvas.paste(r_img, (l_img.width + gutter, 0))
    keep = spread_selection(l_pts, r_pts, confidences, lines, spacing * height) if len(confidences) else []

    dpi = 100
    figure = plt.figure(figsize=(canvas.width / dpi, height / dpi), dpi=dpi)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(canvas, interpolation="none")
    axis.set_xlim(0, canvas.width)
    axis.set_ylim(height, 0)
    axis.set_axis_off()
    if len(keep):
        segments = np.stack([l_pts[keep], r_pts[keep]], axis=1)
        collection = LineCollection(segments, colors=LINE_COLOR, linewidths=1.2, alpha=0.95, zorder=2)
        collection.set_path_effects(
            [patheffects.Stroke(linewidth=2.6, foreground="black", alpha=0.7), patheffects.Normal()]
        )
        axis.add_collection(collection)
        ends = np.r_[l_pts[keep], r_pts[keep]]
        axis.scatter(ends[:, 0], ends[:, 1], s=14, facecolors=LINE_COLOR, edgecolors="white", linewidths=0.6, zorder=3)
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return image, int(len(keep))
