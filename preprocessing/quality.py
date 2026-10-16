"""Upload validation (FR-1) and image-quality checks (FR-2).

Hard errors (wrong type, too large, not decodable) reject the upload. Quality problems only produce
warnings, because a borderline image can still be graded.

Thresholds were calibrated on APTOS 2019 (3,486 images after de-duplication) by inspecting images at each
percentile; re-check them once EyePACS is added (outputs/eval/data/quality_stats.csv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config
from preprocessing.fundus import fundus_mask, read_image

BLANK_STD = 5.0  # grey-level std below this means an empty frame
DARK_MEAN = 25.0  # mean brightness inside the fundus; the darkest APTOS image is 27.5
MIN_FUNDUS_FRACTION = 0.25  # illuminated area must cover at least this share of the cropped square
# A fundus disc leaves the corners of its bounding box at least partly dark; an ordinary rectangular photo
# has fully lit corners (~1.0). Real fundus photos cropped on all four sides reach 0.968 in APTOS, so only
# essentially fully lit corners are flagged.
MAX_CORNER_LIT = 0.99
# Laplacian variance of the green channel at CACHE_SIZE. Images below ~6 (about the 2nd percentile) look
# hazy; at 8-15 vessels are still sharp, so a higher threshold flags good images.
BLUR_VARIANCE = 6.0
# Red/blue ratio inside the fundus. Real photos go down to 0.97 (grey-blue cast from some cameras), so only
# clearly blue or green images are flagged.
MIN_RED_BLUE_RATIO = 0.85


class UploadError(ValueError):
    """The upload cannot be processed at all."""


@dataclass
class QualityReport:
    issues: dict[str, str] = field(default_factory=dict)  # issue code -> message shown to the user
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def warnings(self) -> list[str]:
        return list(self.issues.values())

    @property
    def unusable(self) -> bool:
        """Blank or no visible fundus: grading would be meaningless."""
        return bool(UNUSABLE_ISSUES & self.issues.keys())


UNUSABLE_ISSUES = {"blank", "no_fundus"}


def validate_upload(filename: str, data: bytes) -> np.ndarray:
    """Check extension, size and decodability. Returns the decoded RGB image or raises UploadError."""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise UploadError(f"Unsupported file type '{ext}'. Upload a JPG or PNG fundus photograph.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise UploadError(f"File is {size_mb:.1f} MB; the limit is {config.MAX_UPLOAD_MB} MB.")
    try:
        return read_image(data)
    except ValueError as exc:
        raise UploadError("The file could not be read as an image.") from exc


def _corner_lit_fraction(mask: np.ndarray) -> float:
    """Share of illuminated pixels in the four corner patches of the illuminated bounding box."""
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    # Inset by 2 px: resizing blends the outermost row/column with the black padding
    box = mask[rows[0] + 2 : rows[-1] - 1, cols[0] + 2 : cols[-1] - 1]
    h, w = box.shape
    ph, pw = max(1, h // 12), max(1, w // 12)
    corners = [box[:ph, :pw], box[:ph, -pw:], box[-ph:, :pw], box[-ph:, -pw:]]
    return float(np.mean([c.mean() for c in corners]))


def check_quality(img: np.ndarray) -> QualityReport:
    """Quality checks on a cached fundus image (output of load_fundus)."""
    report = QualityReport()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = fundus_mask(img)
    fraction = float(mask.mean())
    report.metrics["fundus_fraction"] = fraction

    std = float(gray.std())
    report.metrics["grey_std"] = std
    if std < BLANK_STD:
        report.issues["blank"] = "The image looks blank."
        return report

    if fraction < MIN_FUNDUS_FRACTION:
        report.issues["no_fundus"] = (
            "Very little of the image is illuminated; it may be too dark or not a fundus photo."
        )
        return report
    corner_lit = _corner_lit_fraction(mask)
    report.metrics["corner_lit"] = corner_lit
    if corner_lit > MAX_CORNER_LIT:
        report.issues["no_border"] = (
            "There is no dark circular border around the image; this may not be a fundus photograph."
        )

    mean_inside = float(gray[mask].mean())
    report.metrics["mean_brightness"] = mean_inside
    if mean_inside < DARK_MEAN:
        report.issues["dark"] = "The image is too dark for reliable grading."

    # Erode so the sharp fundus edge does not count as image detail
    inner = cv2.erode(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    blur = float(cv2.Laplacian(img[..., 1], cv2.CV_64F)[inner].var()) if inner.any() else 0.0
    report.metrics["laplacian_var"] = blur
    if blur < BLUR_VARIANCE:
        report.issues["blurry"] = "The image looks blurry; small lesions may be missed."

    r, b = img[..., 0][mask].mean(), img[..., 2][mask].mean()
    ratio = float(r / (b + 1.0))
    report.metrics["red_blue_ratio"] = ratio
    if ratio < MIN_RED_BLUE_RATIO:
        report.issues["not_colour_fundus"] = "This does not look like a colour fundus photograph."

    return report
