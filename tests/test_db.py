import json
import sqlite3

import pytest

from database import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    yield c
    c.close()


def test_single_local_user_is_reused(conn):
    assert db.get_or_create_user(conn) == db.get_or_create_user(conn)


def test_run_round_trip(conn):
    user = db.get_or_create_user(conn)
    probs = [0.1, 0.6, 0.2, 0.05, 0.05]
    pid = db.save_prediction(conn, user, "outputs/runs/1/input.png", probs, "cnn_efficientnetb0@2026-10-14", ["blurry"])
    db.save_generated(conn, pid, 1, 2, "outputs/runs/1/sim_2.png", 0.91, 28.4, 2, "generator_1to2@2026-10-15")
    db.save_generated(conn, pid, 2, 3, "outputs/runs/1/sim_3.png", 0.85, 25.1, 2, "generator_2to3@2026-10-15")
    db.set_report_path(conn, pid, "reports/run_1.pdf")

    prediction, generated = db.get_run(conn, pid)
    assert prediction["stage"] == 1 and prediction["confidence"] == pytest.approx(0.6)
    assert json.loads(prediction["probabilities_json"]) == probs
    assert prediction["quality_warnings"] == "blurry"
    assert [g["to_stage"] for g in generated] == [2, 3]

    (row,) = db.history(conn)
    assert row["simulated_steps"] == 2 and row["simulated_to_stage"] == 3
    assert row["report_path"] == "reports/run_1.pdf"


def test_history_newest_first_and_includes_runs_without_simulation(conn):
    user = db.get_or_create_user(conn)
    first = db.save_prediction(conn, user, "a.png", [0, 0, 0, 0, 1.0], "v1")  # stage 4: no simulation
    second = db.save_prediction(conn, user, "b.png", [1.0, 0, 0, 0, 0], "v1")
    rows = db.history(conn)
    assert [r["prediction_id"] for r in rows] == [second, first]
    assert rows[1]["simulated_steps"] == 0 and rows[1]["simulated_to_stage"] is None


def test_stage_must_be_valid(conn):
    user = db.get_or_create_user(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO predictions (user_id, image_path, stage, confidence, probabilities_json, model_version, created_at) "
            "VALUES (?, 'x', 7, 0.5, '[]', 'v', 'now')",
            (user,),
        )


def test_missing_run_returns_none(conn):
    assert db.get_run(conn, 999) is None
