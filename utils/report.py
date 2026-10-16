"""PDF report for one analysis (FR-12): input, grade, probabilities, Grad-CAM, simulated stages, metrics and the
not-a-diagnosis disclaimer (FR-15) on every page.

Text is plain ASCII: the PDF core fonts cannot encode characters such as arrows or long dashes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fpdf import FPDF
from PIL import Image

import config
from prediction.pipeline import Analysis
from utils.visuals import (
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES_1,
    colorbar,
    hex_to_rgb,
    importance_overlay,
    label_simulated,
    sequential_image,
)
from preprocessing.fundus import fundus_mask

MARGIN = 15
PAGE_W = 210
CONTENT_W = PAGE_W - 2 * MARGIN
SHORT_DISCLAIMER = "Research and education prototype - not a diagnosis. Simulated images are not a prediction for this patient."
# Fixed top of the difference scale (mean absolute RGB difference), so maps are comparable across runs
DIFF_SCALE_MAX = 96


class _Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=7.5)
        self.set_text_color(*hex_to_rgb(INK_MUTED))
        self.cell(0, 5, f"{SHORT_DISCLAIMER}   Page {self.page_no()}", align="C")


def _pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def _ink(pdf: FPDF, color: str = INK) -> None:
    pdf.set_text_color(*hex_to_rgb(color))


def _heading(pdf: FPDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    _ink(pdf)
    pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")


def _ensure_space(pdf: FPDF, height: float) -> None:
    if pdf.get_y() + height > pdf.h - pdf.b_margin:
        pdf.add_page()


def _probability_bars(pdf: FPDF, probabilities: np.ndarray) -> None:
    """One series, one colour; value at each bar tip in ink; rounded data-end, square baseline."""
    label_w, bar_max, bar_h, row_h = 42, 105, 4.2, 7
    x0 = MARGIN + label_w
    pdf.set_font("Helvetica", size=9)
    for k, p in enumerate(probabilities):
        y = pdf.get_y()
        _ink(pdf, INK_SECONDARY)
        pdf.set_xy(MARGIN, y)
        pdf.cell(label_w - 3, row_h, f"{k}  {config.STAGE_NAMES[k]}", align="R")
        width = max(float(p) * bar_max, 0.6)
        pdf.set_fill_color(*hex_to_rgb(SERIES_1))
        pdf.rect(x0, y + (row_h - bar_h) / 2, width, bar_h, style="F",
                 round_corners=("TOP_RIGHT", "BOTTOM_RIGHT"), corner_radius=min(1.0, width / 2))  # fmt: skip
        _ink(pdf, INK)
        pdf.set_xy(x0 + width + 1.5, y)
        pdf.cell(20, row_h, f"{p:.0%}")
        pdf.set_y(y + row_h)
    # Baseline
    pdf.set_draw_color(*hex_to_rgb("#c3c2b7"))
    pdf.line(x0, pdf.get_y() - len(probabilities) * row_h, x0, pdf.get_y())


def _image_row(pdf: FPDF, images: list[np.ndarray], captions: list[str], size: float) -> None:
    _ensure_space(pdf, size + 12)
    gap = (CONTENT_W - len(images) * size) / max(len(images) - 1, 1)
    y = pdf.get_y()
    pdf.set_font("Helvetica", size=8)
    _ink(pdf, INK_SECONDARY)
    for i, (img, caption) in enumerate(zip(images, captions)):
        x = MARGIN + i * (size + gap)
        pdf.image(_pil(img), x=x, y=y, w=size, h=size)
        pdf.set_xy(x, y + size + 1)
        pdf.multi_cell(size, 3.6, caption, align="C")
    pdf.set_y(y + size + 10)


def build_report(analysis: Analysis, run_id: int | None = None, created_at: datetime | None = None) -> bytes:
    created_at = created_at or datetime.now(timezone.utc)
    pdf = _Report(format="A4")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    _ink(pdf)
    pdf.cell(0, 9, "RetinaAI analysis report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    _ink(pdf, INK_SECONDARY)
    run = f"Run {run_id}  |  " if run_id is not None else ""
    pdf.cell(0, 5, f"{run}{created_at:%Y-%m-%d %H:%M} UTC  |  classifier {analysis.model_version}", new_x="LMARGIN", new_y="NEXT")

    # Disclaimer box (FR-15)
    pdf.ln(3)
    pdf.set_fill_color(*hex_to_rgb("#f0efec"))
    pdf.set_font("Helvetica", "B", 9)
    _ink(pdf)
    pdf.multi_cell(0, 4.6, "Not a diagnosis. " + config.DISCLAIMER, fill=True, padding=3)

    stage = analysis.stage
    _heading(pdf, f"Detected stage: {stage} - {config.STAGE_NAMES[stage]}  (confidence {analysis.confidence:.0%})")
    images, captions = [analysis.cnn_input], ["Uploaded image after preprocessing (as graded)"]
    if analysis.gradcam is not None:
        images.append(importance_overlay(analysis.cnn_input, analysis.gradcam))
        captions.append("Grad-CAM: darker blue = more influence on the grade")
    _image_row(pdf, images, captions, size=62 if len(images) == 2 else 70)

    _heading(pdf, "Probability of each stage")
    _probability_bars(pdf, analysis.probabilities)

    if analysis.quality.warnings:
        _heading(pdf, "Image quality warnings")
        pdf.set_font("Helvetica", size=9)
        _ink(pdf, INK_SECONDARY)
        for warning in analysis.quality.warnings:
            pdf.multi_cell(0, 5, f"- {warning}", new_x="LMARGIN", new_y="NEXT")
    for note in analysis.notes:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        _ink(pdf, INK_SECONDARY)
        pdf.multi_cell(0, 5, note, new_x="LMARGIN", new_y="NEXT")

    for sim in analysis.simulations:
        _ensure_space(pdf, 85)
        _heading(pdf, f"Simulated stage {sim.to_stage} - {config.STAGE_NAMES[sim.to_stage]} (from stage {sim.from_stage})")
        diff = sequential_image(sim.difference, fundus_mask(analysis.current), vmax=DIFF_SCALE_MAX)
        size = 52
        _image_row(
            pdf,
            [analysis.current, label_simulated(sim.image), diff],
            ["Current retina", "SIMULATED next-stage appearance", "Difference from current"],
            size=size,
        )
        # Scale for the difference map, under its column
        x = MARGIN + 2 * (size + (CONTENT_W - 3 * size) / 2)
        y = pdf.get_y() - 3
        pdf.image(_pil(colorbar()), x=x, y=y, w=size, h=2.5)
        pdf.set_font("Helvetica", size=7)
        _ink(pdf, INK_MUTED)
        pdf.set_xy(x, y + 2.8)
        pdf.cell(size / 2, 3.5, "no change")
        pdf.cell(size / 2, 3.5, "large change", align="R")
        pdf.set_y(y + 8)
        pdf.set_font("Helvetica", size=9)
        _ink(pdf)
        verdict = "matches the simulated stage" if sim.regrade_matches else "does not match the simulated stage"
        pdf.multi_cell(
            0, 5,
            f"SSIM {sim.ssim:.3f}  |  PSNR {sim.psnr:.1f} dB  |  Re-graded by the classifier as stage {sim.regraded_stage} - "
            f"{config.STAGE_NAMES[sim.regraded_stage]} ({verdict}).  Generator {sim.generator_version}.",
            new_x="LMARGIN", new_y="NEXT",
        )  # fmt: skip

    return bytes(pdf.output())
