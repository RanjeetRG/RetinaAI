"""Similarity between a fundus image and its simulated next stage (FR-11, M4 evaluation).

SSIM and PSNR are computed inside the fundus only: the black surround is identical in every image and would
inflate both scores.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from preprocessing.fundus import fundus_mask


def _mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return fundus_mask(a) | fundus_mask(b)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM over the fundus of two uint8 RGB images of the same size."""
    _, ssim_map = structural_similarity(a, b, channel_axis=-1, data_range=255, full=True)
    return float(ssim_map.mean(axis=-1)[_mask(a, b)].mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB over the fundus of two uint8 RGB images (inf if identical)."""
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float((diff[_mask(a, b)] ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def change_decomposition(inputs: np.ndarray, outputs: np.ndarray) -> tuple[float, float]:
    """Split each image's change into a global colour shift and the local change left after removing it.

    The first 0->1 generator changed exposure and colour (camera traits) more than it added lesions, so the paper
    reports both parts separately. Returns means over images of (|global shift|, local change), in 0-255 units.
    Read them together: a pure exposure change gives local ~0, while a purely local change still moves the image
    mean a little, giving local ~2x global.
    """
    global_parts, local_parts = [], []
    for a, b in zip(inputs, outputs):
        inside = a.max(axis=-1) > 25
        diff = b.astype(np.float32) - a.astype(np.float32)
        shift = diff[inside].mean(axis=0)
        global_parts.append(np.abs(shift).mean())
        local_parts.append(np.abs(diff - shift)[inside].mean())
    return float(np.mean(global_parts)), float(np.mean(local_parts))


def change_map_correlation(inputs: np.ndarray, outputs: np.ndarray, max_images: int = 100) -> float:
    """Mean pairwise Pearson correlation between different images' change maps (|output - input|).

    Detects shortcut "stamps": classifier-guided generators learned to add the same dots at the same position in
    every eye, giving correlations near 1; eye-specific changes give values near 0.
    """
    n = min(len(inputs), max_images)
    maps = np.abs(outputs[:n].astype(np.float32) - inputs[:n].astype(np.float32)).mean(axis=-1)
    inside = (inputs[:n].max(axis=-1) > 25).all(axis=0)  # pixels inside every image's fundus
    flat = maps[:, inside]
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    valid = norms > 1e-6
    if valid.sum() < 2:
        return float("nan")
    flat = flat[valid] / norms[valid, None]
    corr = flat @ flat.T
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    return float(off_diagonal.mean())


def difference_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel mean absolute difference over RGB, 0-255 float32, zero outside the fundus."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=-1)
    diff[~_mask(a, b)] = 0
    return diff
