"""Image-grid helpers for paper figures (labelled rows of equally sized images on white)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CELL = 224
GAP = 6
LABEL_W = 150
WHITE = 255


def label_column(text: str, height: int, width: int = LABEL_W) -> np.ndarray:
    col = np.full((height, width, 3), WHITE, np.uint8)
    for i, line in enumerate(text.split("\n")):
        y = height // 2 - 10 + i * 24
        cv2.putText(col, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return col


def row_of(images: list[np.ndarray], label: str, cell: int = CELL) -> np.ndarray:
    cells = []
    for img in images:
        cells += [cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA), np.full((cell, GAP, 3), WHITE, np.uint8)]
    return np.hstack([label_column(label, cell), *cells[:-1]])


def header(titles: list[str], width: int, cell: int = CELL) -> np.ndarray:
    out = np.full((40, width, 3), WHITE, np.uint8)
    for i, title in enumerate(titles):
        x = LABEL_W + i * (cell + GAP) + 10
        cv2.putText(out, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def stack(rows: list[np.ndarray]) -> np.ndarray:
    gap = np.full((GAP, rows[0].shape[1], 3), WHITE, np.uint8)
    out = []
    for r in rows:
        out += [r, gap]
    return np.vstack(out[:-1])


def save_jpeg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", path)
