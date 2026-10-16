"""Smoke test of the full training loop on tiny random data (CPU, no ImageNet download)."""

import argparse
import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import config  # noqa: E402
from cnn import train as T  # noqa: E402
from cnn.data import SplitArrays  # noqa: E402
from cnn.model import build_classifier, load_classifier  # noqa: E402


def fake_split(n, seed):
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 255, (n, config.CNN_SIZE, config.CNN_SIZE, 3), dtype=np.uint8)
    stages = np.arange(n) % config.NUM_STAGES
    return SplitArrays(images, stages, np.array(["aptos"] * n), np.array([f"img{i}" for i in range(n)]))


def test_class_weights_favour_rare_stages_and_average_one():
    stages = np.array([0] * 80 + [1] * 10 + [2] * 6 + [3] * 2 + [4] * 2)
    w = T.class_weights(stages, power=0.5)
    assert w[0] < w[1] < w[3]
    assert w[stages].mean() == pytest.approx(1.0)


def test_batch_loader_covers_every_sample_once_per_epoch():
    data = fake_split(10, 0)
    loader = T.BatchLoader(data, batch_size=4, sample_weights=np.ones(10), shuffle=True, seed=1)
    seen = np.concatenate([loader[i][1] for i in range(len(loader))])
    assert len(loader) == 3 and len(seen) == 10
    x, y, w = loader[0]
    assert x.dtype == np.float32 and x.shape[1:] == (config.CNN_SIZE, config.CNN_SIZE, 3)


def test_augmentation_keeps_surround_black_and_is_reproducible():
    from preprocessing.fundus import load_fundus, prepare_cnn
    from tests.conftest import make_fundus

    image = prepare_cnn(load_fundus(make_fundus()))
    data = SplitArrays(np.stack([image] * 4), np.zeros(4, int), np.array(["aptos"] * 4), np.array(list("abcd")))
    a = T.BatchLoader(data, batch_size=4, augment=True, seed=3)[0][0]
    b = T.BatchLoader(data, batch_size=4, augment=True, seed=3)[0][0]
    assert a.dtype == np.float32 and a.shape == (4, *image.shape)
    np.testing.assert_array_equal(a, b)  # same seed, epoch and batch -> same augmentation
    assert not np.array_equal(a[0], a[1])  # images in a batch get different transforms
    corners = a[:, :3, :3]  # outside the fundus after any rotation of the centred field
    assert corners.max() == 0
    assert a.max() <= 255 and a.min() >= 0


@pytest.mark.slow
def test_train_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path / "eval")
    monkeypatch.setattr(config, "CNN_MODEL_PATH", tmp_path / "models" / "model.keras")
    splits = {"train": fake_split(20, 0), "validation": fake_split(10, 1)}
    monkeypatch.setattr(T, "load_split", lambda split, *args, **kwargs: splits[split])
    # No ImageNet download in tests: force weights=None even where train.py asks for "imagenet"
    monkeypatch.setattr(T, "build_classifier", lambda backbone, **kw: build_classifier(backbone, **{**kw, "weights": None}))

    args = argparse.Namespace(
        backbone="mobilenetv2", size=config.CNN_SIZE, promote=True, batch_size=8, head_epochs=1, head_lr=1e-3, epochs=1, lr=1e-4,
        weight_decay=1e-4, dropout=0.3, class_weight_power=0.5, patience=2, seed=0, workers=1,
    )  # fmt: skip
    try:
        T.train(args)
    finally:
        keras.mixed_precision.set_global_policy("float32")

    saved = tmp_path / "models" / "cnn_mobilenetv2_224.keras"
    model = load_classifier(saved)
    assert model.dtype_policy.name == "float32"  # CPU-friendly for the app
    probs = keras.ops.convert_to_numpy(model(splits["validation"].images[:2].astype("float32")))
    assert probs.shape == (2, config.NUM_STAGES)
    meta = json.loads(saved.with_suffix(".json").read_text())
    assert meta["best_epoch"] in (1, 2)
    assert meta["input_size"] == config.CNN_SIZE
    assert (tmp_path / "eval" / "cnn" / "mobilenetv2_224" / "history.csv").exists()
    assert not (tmp_path / "models" / "cnn_mobilenetv2_224.best.weights.h5").exists()
    assert (tmp_path / "models" / "model.keras").exists()  # --promote
