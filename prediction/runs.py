"""Persist an analysis: images under outputs/runs/<run>/, rows in SQLite (FR-13), PDF in reports/ (FR-12).

Simulated images are written with the "SIMULATED" band already burned in (FR-15), so a file opened straight
from disk is still labelled.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from database import db
from prediction.pipeline import Analysis
from utils.report import build_report
from utils.visuals import label_simulated

RUNS_DIR = config.OUTPUTS_DIR / "runs"


def _write_png(path: Path, img: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"could not encode {path.name}")
    path.write_bytes(buf.tobytes())
    return path.relative_to(config.ROOT).as_posix()


def save_run(conn: sqlite3.Connection, analysis: Analysis, with_report: bool = True) -> tuple[int, Path | None]:
    """Save everything about one analysis; returns (prediction id, report path or None)."""
    created = datetime.now(timezone.utc)
    run_dir = RUNS_DIR / created.strftime("%Y%m%d-%H%M%S-%f")
    user = db.get_or_create_user(conn)
    input_path = _write_png(run_dir / "input.png", analysis.cached)
    _write_png(run_dir / "current.png", analysis.current)
    prediction_id = db.save_prediction(
        conn, user, input_path, analysis.probabilities.tolist(), analysis.model_version, analysis.quality.warnings
    )
    for sim in analysis.simulations:
        path = _write_png(run_dir / f"simulated_stage_{sim.to_stage}.png", label_simulated(sim.image))
        db.save_generated(
            conn, prediction_id, sim.from_stage, sim.to_stage, path, sim.ssim, sim.psnr, sim.regraded_stage, sim.generator_version
        )

    report_path = None
    if with_report:
        report_path = config.REPORTS_DIR / f"retinagen_run_{prediction_id}.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(build_report(analysis, prediction_id, created))
        db.set_report_path(conn, prediction_id, report_path.relative_to(config.ROOT).as_posix())
    return prediction_id, report_path
