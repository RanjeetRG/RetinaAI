"""End-to-end pipeline tests with stand-in models: a classifier rigged to predict a chosen stage and tiny
untrained generators. They check the workflow (FR-6 to FR-15), not model quality."""

import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import config  # noqa: E402
from cnn.model import build_classifier  # noqa: E402
from database import db  # noqa: E402
from gan.networks import build_generator  # noqa: E402
from prediction import runs  # noqa: E402
from prediction.pipeline import Pipeline  # noqa: E402
from tests.conftest import make_fundus  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "outputs" / "runs")
    (tmp_path / "models").mkdir()
    return tmp_path


def rigged_classifier(path, stage: int, size: int = config.CNN_SIZE):
    """A real classifier architecture whose head always outputs `stage`."""
    model = build_classifier("mobilenetv2", weights=None, input_size=size)
    head = model.get_layer("probs")
    kernel, bias = head.get_weights()
    kernel[:] = 0
    bias[:] = -8
    bias[stage] = 8
    head.set_weights([kernel, bias])
    model.save(path)
    path.with_suffix(".json").write_text(json.dumps({"backbone": "mobilenetv2", "trained_at": "2026-10-14T00:00:00"}))
    return path


def tiny_generators(*stages):
    for k in stages:
        build_generator(base_filters=8).save(config.generator_path(k))
        config.generator_path(k).with_suffix(".json").write_text(
            json.dumps({"transition": f"{k}to{k + 1}", "trained_at": "2026-10-15T00:00:00"})
        )


def test_grades_and_chains_simulations(workspace):
    tiny_generators(1, 2, 3)
    pipeline = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=1))
    result = pipeline.analyze(make_fundus(), steps=2)

    assert result.stage == 1 and result.confidence > 0.99
    assert result.model_version == "cnn_mobilenetv2@2026-10-14"
    assert result.gradcam.shape == (7, 7)
    assert [(s.from_stage, s.to_stage) for s in result.simulations] == [(1, 2), (2, 3)]
    for sim in result.simulations:
        assert sim.image.shape == (config.GAN_SIZE, config.GAN_SIZE, 3) and sim.image.dtype == np.uint8
        assert -1 <= sim.ssim <= 1 and sim.psnr > 0
        assert sim.regraded_probabilities.shape == (config.NUM_STAGES,)
        assert sim.generator_version.startswith(f"generator_{sim.from_stage}to{sim.to_stage}@")
    assert result.notes == []


def test_stage_4_skips_simulation(workspace):
    tiny_generators(0, 1, 2, 3)
    result = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=4)).analyze(make_fundus(), steps=3)
    assert result.simulations == []
    assert "last stage" in result.notes[0]


def test_missing_generator_stops_the_chain_with_a_note(workspace):
    tiny_generators(1)  # 2to3 and 3to4 not trained yet
    result = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=1)).analyze(make_fundus(), steps=3)
    assert [s.to_stage for s in result.simulations] == [2]
    assert "stage 2 to 3 generator is not available" in result.notes[0]


def test_generator_cache_keeps_only_recent(workspace):
    tiny_generators(1, 2)
    pipeline = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=1), max_cached_generators=1)
    pipeline.generator(1)
    pipeline.generator(2)
    assert list(pipeline._generators) == [2]


def test_save_run_writes_database_report_and_labelled_images(workspace):
    tiny_generators(2)
    pipeline = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=2))
    result = pipeline.analyze(make_fundus(), steps=1)
    conn = db.connect(workspace / "test.db")

    prediction_id, report = runs.save_run(conn, result)

    assert report.read_bytes()[:4] == b"%PDF"
    prediction, generated = db.get_run(conn, prediction_id)
    assert prediction["stage"] == 2 and prediction["report_path"].endswith(".pdf")
    (sim_row,) = generated
    assert (sim_row["from_stage"], sim_row["to_stage"]) == (2, 3)
    saved = workspace / sim_row["image_path"]
    assert saved.exists() and saved.name == "simulated_stage_3.png"
    assert (workspace / prediction["image_path"]).exists()


def test_preprocessing_follows_the_classifier_input_size(workspace):
    tiny_generators(1)
    pipeline = Pipeline(rigged_classifier(workspace / "models" / "model.keras", stage=1, size=160))
    result = pipeline.analyze(make_fundus(), steps=1)
    assert result.cnn_input.shape == (160, 160, 3)
    assert result.simulations[0].regraded_probabilities.shape == (config.NUM_STAGES,)
