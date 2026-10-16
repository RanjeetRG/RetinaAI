"""Fundus preprocessing shared by training and inference (FR-3).

Pipeline (PRD order): crop black border -> pad to square -> resize -> light denoise -> CLAHE -> standard field of
view -> normalize.

Every image is first brought to CACHE_SIZE by `load_fundus`. Training caches that result on disk and the
app calls the same function, so both paths see identical pixels. CNN and GAN inputs are then downscaled
from the cached image (never 224 -> 256 upscaling).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Union

import cv2
import numpy as np

import config

ImageSource = Union[str, Path, bytes, np.ndarray]


def read_image(source: ImageSource) -> np.ndarray:
    """Read an image as RGB uint8. Accepts a path, raw file bytes, or an RGB array.

    Uses imdecode rather than imread so non-ASCII Windows paths work.
    """
    if isinstance(source, np.ndarray):
        img = source
    else:
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("File is not a readable image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return np.ascontiguousarray(img, dtype=np.uint8)


def fundus_mask(img: np.ndarray, threshold: int = config.BORDER_THRESHOLD) -> np.ndarray:
    """Boolean mask of the illuminated fundus (True) versus the black surround (False)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    # Opening removes isolated bright pixels (JPEG noise, text labels) from the border
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask.astype(bool)


def crop_border(img: np.ndarray, threshold: int = config.BORDER_THRESHOLD) -> np.ndarray:
    """Crop to the bounding box of the fundus. Returns the input unchanged if no fundus is found."""
    mask = fundus_mask(img, threshold)
    # A row/column belongs to the fundus if at least 1% of it is illuminated
    rows = np.where(mask.mean(axis=1) > 0.01)[0]
    cols = np.where(mask.mean(axis=0) > 0.01)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad with black so the fundus stays centred (EyePACS often clips the top and bottom of the circle)."""
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a square image to size x size (area interpolation when shrinking, cubic when enlarging)."""
    interp = cv2.INTER_AREA if img.shape[0] >= size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def load_fundus(source: ImageSource, size: int = config.CACHE_SIZE) -> np.ndarray:
    """Read, crop the black border, pad to square and resize. This is the cached training image."""
    img = read_image(source)
    return resize(pad_to_square(crop_border(img)), size)


def denoise(img: np.ndarray, kernel: int = config.DENOISE_KERNEL) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel only, so colours (haemorrhages, exudates) keep their hue."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@lru_cache(maxsize=8)
def field_template(size: int) -> np.ndarray:
    """The standard field of view shared by every model input (see config.FIELD_HEIGHT_RATIO).

    A centred circle spanning the full width, clipped to a horizontal band. After crop_border and pad_to_square
    every fundus spans the full width, so the template lines up with both full and clipped camera fields.
    """
    yy, xx = np.mgrid[:size, :size] + 0.5 - size / 2
    radius = size / 2
    template = (xx**2 + yy**2 <= radius**2) & (np.abs(yy) <= config.FIELD_HEIGHT_RATIO * radius)
    template.setflags(write=False)
    return template


def enhance(img: np.ndarray, size: int) -> np.ndarray:
    """Resize -> denoise -> CLAHE, then black out everything outside the fundus and the standard field."""
    small = resize(img, size)
    mask = fundus_mask(small) & field_template(size)
    out = apply_clahe(denoise(small))
    out[~mask] = 0
    return out


def prepare_cnn(img: np.ndarray, size: int = config.CNN_SIZE) -> np.ndarray:
    """CNN input from a cached (CACHE_SIZE) fundus: uint8 RGB, `size` square (the classifier's input size).

    Values stay 0-255; each saved classifier contains its backbone's normalization as a layer.
    """
    return enhance(img, size)


def prepare_gan(img: np.ndarray) -> np.ndarray:
    """GAN input from a cached (CACHE_SIZE) fundus: float32 in [-1, 1], GAN_SIZE square."""
    return to_gan_range(enhance(img, config.GAN_SIZE))


def to_gan_range(img_uint8: np.ndarray) -> np.ndarray:
    return img_uint8.astype(np.float32) / 127.5 - 1.0


def from_gan_range(img: np.ndarray) -> np.ndarray:
    """Generator output in [-1, 1] -> uint8 RGB."""
    return np.clip((img + 1.0) * 127.5, 0, 255).round().astype(np.uint8)


def cnn_input_from_gan_image(img_uint8: np.ndarray, size: int = config.CNN_SIZE) -> np.ndarray:
    """CNN input for re-grading a simulated image (FR-9).

    Generator output is already in the denoised, CLAHE-enhanced domain, so it is only resized; applying
    CLAHE a second time would shift it away from what the classifier was trained on. For a classifier larger than
    GAN_SIZE this upscales, so fine lesions look blurrier than in real inputs of that size.
    """
    return resize(img_uint8, size)
