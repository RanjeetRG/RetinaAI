"""SQLite run history (FR-13, FR-14; PRD "Data model").

Three tables plus a view. Images live as files under outputs/; the database stores their paths only.
v1 has a single local user.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY,
    user_id             INTEGER NOT NULL REFERENCES users(id),
    image_path          TEXT NOT NULL,
    stage               INTEGER NOT NULL CHECK (stage BETWEEN 0 AND 4),
    confidence          REAL NOT NULL,
    probabilities_json  TEXT NOT NULL,
    quality_warnings    TEXT NOT NULL DEFAULT '',
    model_version       TEXT NOT NULL,
    report_path         TEXT,
    created_at          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generated_images (
    id                  INTEGER PRIMARY KEY,
    prediction_id       INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    from_stage          INTEGER NOT NULL,
    to_stage            INTEGER NOT NULL,
    image_path          TEXT NOT NULL,
    ssim                REAL,
    psnr                REAL,
    regraded_stage      INTEGER,
    generator_version   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generated_prediction ON generated_images(prediction_id);
CREATE VIEW IF NOT EXISTS history AS
    SELECT p.id AS prediction_id, p.created_at, u.name AS user_name, p.image_path, p.stage, p.confidence,
           p.quality_warnings, p.model_version, p.report_path,
           COUNT(g.id) AS simulated_steps, MAX(g.to_stage) AS simulated_to_stage
    FROM predictions p
    JOIN users u ON u.id = p.user_id
    LEFT JOIN generated_images g ON g.prediction_id = p.id
    GROUP BY p.id
    ORDER BY p.created_at DESC, p.id DESC;
"""

LOCAL_USER = "local"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.DB_PATH)  # read at call time, not import time, so tests can redirect it
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)  # Streamlit reruns may come from another thread
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def get_or_create_user(conn: sqlite3.Connection, name: str = LOCAL_USER) -> int:
    conn.execute("INSERT OR IGNORE INTO users (name, created_at) VALUES (?, ?)", (name, _now()))
    conn.commit()
    return conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()["id"]


def save_prediction(
    conn: sqlite3.Connection,
    user_id: int,
    image_path: str,
    probabilities: list[float],
    model_version: str,
    quality_warnings: list[str] = (),
) -> int:
    stage = max(range(len(probabilities)), key=probabilities.__getitem__)
    cur = conn.execute(
        """INSERT INTO predictions (user_id, image_path, stage, confidence, probabilities_json, quality_warnings,
                                    model_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, image_path, stage, float(probabilities[stage]), json.dumps([float(p) for p in probabilities]),
         "\n".join(quality_warnings), model_version, _now()),
    )  # fmt: skip
    conn.commit()
    return cur.lastrowid


def save_generated(
    conn: sqlite3.Connection,
    prediction_id: int,
    from_stage: int,
    to_stage: int,
    image_path: str,
    ssim: float | None,
    psnr: float | None,
    regraded_stage: int | None,
    generator_version: str,
) -> int:
    cur = conn.execute(
        """INSERT INTO generated_images (prediction_id, from_stage, to_stage, image_path, ssim, psnr, regraded_stage,
                                         generator_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (prediction_id, from_stage, to_stage, image_path, ssim, psnr, regraded_stage, generator_version),
    )
    conn.commit()
    return cur.lastrowid


def set_report_path(conn: sqlite3.Connection, prediction_id: int, report_path: str) -> None:
    conn.execute("UPDATE predictions SET report_path = ? WHERE id = ?", (report_path, prediction_id))
    conn.commit()


def history(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM history LIMIT ?", (limit,)).fetchall()


def get_run(conn: sqlite3.Connection, prediction_id: int) -> tuple[sqlite3.Row, list[sqlite3.Row]] | None:
    """A past run: its prediction row and its simulated stages in order (FR-14)."""
    prediction = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if prediction is None:
        return None
    generated = conn.execute(
        "SELECT * FROM generated_images WHERE prediction_id = ? ORDER BY to_stage", (prediction_id,)
    ).fetchall()
    return prediction, generated
