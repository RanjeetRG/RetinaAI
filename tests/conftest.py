import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_fundus(seed: int = 0, size: tuple[int, int] = (600, 800)) -> np.ndarray:
    """Synthetic colour fundus: orange disc with a bright optic disc, random dark vessels and texture,
    on a black surround with extra black margin left and right (like many real photos)."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w, 3), np.uint8)
    centre, radius = (w // 2, h // 2), int(h * 0.46)
    cv2.circle(img, centre, radius, (200, 90, 40), -1)
    od = (centre[0] + int(rng.integers(-150, 150)), centre[1] + int(rng.integers(-60, 60)))
    cv2.circle(img, od, 35, (250, 210, 150), -1)
    for _ in range(14):
        pts = (od + np.cumsum(rng.normal(0, 25, size=(8, 2)), axis=0)).astype(np.int32)
        cv2.polylines(img, [pts], False, (110, 30, 20), int(rng.integers(2, 6)))
    noise = rng.normal(0, 8, img.shape)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, centre, radius, 1, -1)
    img = np.clip(img + noise * mask[..., None], 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def fundus() -> np.ndarray:
    return make_fundus()


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()
