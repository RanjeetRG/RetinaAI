"""Colours and image renderings shared by the app and the PDF report (FR-4, FR-5, FR-11).

Follows the data-viz rules used across the project: magnitude (difference, Grad-CAM importance) uses ONE hue
from light (near zero) to dark (large), never a rainbow "jet" map; the single-series probability chart uses
one colour for every bar, and text stays in neutral ink.
"""

from __future__ import annotations

import cv2
import numpy as np

# Reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_1 = "#2a78d6"
# Sequential blue ramp, steps 100 -> 700
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _ramp_lut(steps: list[str]) -> np.ndarray:
    """256-entry RGB lookup table interpolating the ramp steps."""
    anchors = np.array([hex_to_rgb(c) for c in steps], dtype=np.float32)
    positions = np.linspace(0, 255, len(anchors))
    return np.stack([np.interp(np.arange(256), positions, anchors[:, ch]) for ch in range(3)], axis=1).astype(np.uint8)


SEQUENTIAL_LUT = _ramp_lut(SEQUENTIAL_BLUE)


def sequential_image(values: np.ndarray, mask: np.ndarray | None = None, vmax: float | None = None) -> np.ndarray:
    """Map non-negative values to the blue ramp (light = small, dark = large). Outside `mask` is surface.

    vmax fixes the top of the scale (e.g. 255 for absolute differences) so images stay comparable; by
    default the 99th percentile is used, which keeps a few extreme pixels from washing everything out.
    """
    top = vmax if vmax is not None else max(float(np.percentile(values, 99)), 1e-6)
    idx = np.clip(values / top * 255, 0, 255).astype(np.uint8)
    out = SEQUENTIAL_LUT[idx]
    if mask is not None:
        out[~mask] = hex_to_rgb(SURFACE)
        # The near-zero end of the ramp is close to the surface, so outline the fundus to keep its shape visible
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, hex_to_rgb(INK_MUTED), 1, cv2.LINE_AA)
    return out


def importance_overlay(image: np.ndarray, importance: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Tint a fundus image with the darkest ramp blue where `importance` (0-1, any size) is high (Grad-CAM).

    Blue against the orange-red retina stays visible; low-importance areas keep the original image.
    """
    imp = cv2.resize(importance.astype(np.float32), image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    alpha = (np.clip(imp, 0, 1) * strength)[..., None]
    tint = np.array(hex_to_rgb(SEQUENTIAL_BLUE[-3]), dtype=np.float32)
    return (image.astype(np.float32) * (1 - alpha) + tint * alpha).astype(np.uint8)


def label_simulated(image: np.ndarray) -> np.ndarray:
    """Burn a 'SIMULATED' band into a generated image (FR-15), so the label survives downloads and screenshots."""
    out = image.copy()
    h, w = out.shape[:2]
    band = max(18, h // 11)
    out[h - band :] = (out[h - band :].astype(np.float32) * 0.35).astype(np.uint8)
    scale = band / 30
    text = "SIMULATED"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 2)))
    cv2.putText(out, text, ((w - tw) // 2, h - (band - th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)  # fmt: skip
    return out


def colorbar(width: int = 256, height: int = 14) -> np.ndarray:
    """Horizontal scale for the sequential ramp, low on the left."""
    return np.repeat(SEQUENTIAL_LUT[np.linspace(0, 255, width).astype(np.uint8)][None], height, axis=0)
