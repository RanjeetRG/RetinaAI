"""Headless Streamlit test of the full app flow with the trained classifier (skipped if it is not trained)."""

from pathlib import Path

import cv2
import pytest

import config
from database import db
from prediction import runs
from tests.conftest import make_fundus

APP = Path(__file__).resolve().parents[1] / "app.py"
pytestmark = pytest.mark.skipif(not config.CNN_MODEL_PATH.exists(), reason="classifier not trained")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    samples = tmp_path / "static" / "samples"
    samples.mkdir(parents=True)
    cv2.imwrite(str(samples / "synthetic.jpg"), cv2.cvtColor(make_fundus(), cv2.COLOR_RGB2BGR))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "STATIC_DIR", tmp_path / "static")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "database" / "test.db")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "outputs" / "runs")
    return tmp_path


@pytest.mark.slow
def test_sample_image_is_graded_saved_and_reportable(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Not a diagnosis" in w.value for w in at.warning)  # disclaimer before any upload (FR-15)

    sample = next(o for o in at.selectbox[0].options if "synthetic" in str(o))
    at.selectbox[0].set_value(next(p for p in (app_env / "static" / "samples").iterdir() if p.stem == sample)).run()
    assert not at.exception
    assert at.subheader[0].value.startswith("Detected: Stage")
    assert any("Not a diagnosis" in w.value for w in at.warning)

    conn = db.connect(config.DB_PATH)
    (row,) = db.history(conn)
    assert (app_env / row["report_path"]).read_bytes()[:4] == b"%PDF"
