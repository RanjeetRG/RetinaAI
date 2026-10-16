import argparse

import numpy as np
import pytest

keras = pytest.importorskip("keras")

import config  # noqa: E402
from gan import train as T  # noqa: E402
from gan.cyclegan import ImagePool  # noqa: E402
from gan.networks import build_discriminator, build_generator, load_generator  # noqa: E402


def test_generator_shapes_and_range():
    g = build_generator(base_filters=8)
    x = np.random.default_rng(0).uniform(-1, 1, (1, config.GAN_SIZE, config.GAN_SIZE, 3)).astype("float32")
    y = keras.ops.convert_to_numpy(g(x))
    assert y.shape == x.shape and y.min() >= -1 and y.max() <= 1


def test_generator_leaves_the_black_surround_black(tmp_path):
    """The first 0->1 run painted blobs outside the retina; the output mask makes that impossible."""
    from preprocessing.fundus import load_fundus, prepare_gan, resize
    from tests.conftest import make_fundus

    x = prepare_gan(load_fundus(make_fundus()))[None]
    surround = x[0].max(axis=-1) <= -0.95
    assert surround.any() and (~surround).any()
    g = build_generator(base_filters=8)
    # Push the untrained generator towards white everywhere: only the fundus may change
    out_conv = g.get_layer("out_conv")
    kernel, bias = out_conv.get_weights()
    out_conv.set_weights([kernel * 0, bias * 0 + 5.0])
    y = keras.ops.convert_to_numpy(g(x))[0]
    assert np.all(y[surround] == -1.0)
    assert y[~surround].mean() > 0.9

    path = tmp_path / "g.keras"
    g.save(path)
    reloaded = keras.ops.convert_to_numpy(load_generator(path)(x))[0]
    np.testing.assert_allclose(reloaded, y, atol=1e-6)


def test_no_transposed_convolutions():
    """Checkerboard artifacts come from transposed convolutions (PRD risk); upsampling must be resize + conv."""
    g = build_generator(base_filters=8)
    assert not any(isinstance(layer, keras.layers.Conv2DTranspose) for layer in g.layers)


def test_discriminator_is_70px_patchgan():
    d = build_discriminator(base_filters=8)
    out = d(np.zeros((1, config.GAN_SIZE, config.GAN_SIZE, 3), "float32"))
    assert tuple(out.shape) == (1, 30, 30, 1)


@pytest.mark.parametrize("text, expected", [("0to1", 0), ("3to4", 3)])
def test_parse_transition(text, expected):
    assert T.parse_transition(text) == expected


@pytest.mark.parametrize("text", ["1to3", "4to5", "2to1"])
def test_parse_transition_rejects_invalid(text):
    with pytest.raises(argparse.ArgumentTypeError):
        T.parse_transition(text)


def test_learning_rate_constant_then_linear_decay():
    assert T.learning_rate(0, 100, 1.0) == 1.0
    assert T.learning_rate(49, 100, 1.0) == 1.0
    assert T.learning_rate(75, 100, 1.0) == pytest.approx(0.5)
    assert T.learning_rate(100, 100, 1.0) == 0.0


def test_image_pool_returns_current_images_until_full():
    pool = ImagePool(size=3, seed=0)
    batch = np.arange(2)[:, None]
    assert np.array_equal(pool.query(batch), batch)
    later = pool.query(np.array([[7], [8], [9], [10]]))
    assert later.shape == (4, 1) and len(pool.images) == 3


@pytest.mark.slow
def test_train_resume_and_export(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    rng = np.random.default_rng(0)
    imgs = lambda n: rng.integers(0, 255, (n, config.GAN_SIZE, config.GAN_SIZE, 3), dtype=np.uint8)  # noqa: E731
    args = argparse.Namespace(
        transition=1, steps=2, batch_size=1, base_filters=8, lr=2e-4, lambda_cycle=10.0, lambda_identity=5.0,
        log_every=1, sample_every=2, checkpoint_every=1, seed=0, guide=None, lambda_class=1.0, lambda_diversity=0.0, rotate=False,
    )  # fmt: skip
    T.train(args, imgs(4), imgs(4), imgs(2))
    args.steps = 3  # rerunning resumes from the saved checkpoint
    T.train(args, imgs(4), imgs(4), imgs(2))

    history = (tmp_path / "outputs" / "gan" / "1to2" / "history.csv").read_text().splitlines()
    assert [line.split(",")[0] for line in history[1:]] == ["1", "2", "3"]
    assert (tmp_path / "outputs" / "gan" / "1to2" / "samples" / "step_000003.jpg").exists()
    g = load_generator(tmp_path / "models" / "generator_1to2.keras")
    assert g.output_shape == (None, config.GAN_SIZE, config.GAN_SIZE, 3)


@pytest.mark.slow
def test_classifier_guided_training(tmp_path, monkeypatch):
    """Guidance runs inside the compiled training step, is logged and recorded, and never trains the classifier."""
    import json

    from cnn.model import build_classifier
    from gan.cyclegan import CycleGAN

    # The guiding classifier's weights are untouched by a guided generator step
    guide = build_classifier("mobilenetv2", weights=None, input_size=96)
    before = [w.copy() for w in guide.get_weights()]
    gan = CycleGAN(base_filters=8, lambda_identity=0.5, guide=guide, source_stage=2)
    x = np.random.default_rng(1).uniform(-1, 1, (1, config.GAN_SIZE, config.GAN_SIZE, 3)).astype("float32")
    _, _, losses = gan.generator_step(x, x)
    assert float(losses["class"]) > 0
    assert all(np.array_equal(a, b) for a, b in zip(before, guide.get_weights()))

    # End to end through train(): the guidance loss is logged and the guide is recorded in the metadata
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    guide_path = tmp_path / "guide.keras"
    guide.save(guide_path)
    rng = np.random.default_rng(0)
    imgs = lambda n: rng.integers(0, 255, (n, config.GAN_SIZE, config.GAN_SIZE, 3), dtype=np.uint8)  # noqa: E731
    args = argparse.Namespace(
        transition=2, steps=2, batch_size=1, base_filters=8, lr=2e-4, lambda_cycle=10.0, lambda_identity=0.5,
        log_every=1, sample_every=2, checkpoint_every=10, seed=0, guide=str(guide_path), lambda_class=1.0, lambda_diversity=1.0, rotate=True,
    )  # fmt: skip
    T.train(args, imgs(4), imgs(4), imgs(2))
    header, *rows = (tmp_path / "outputs" / "gan" / "2to3" / "history.csv").read_text().splitlines()
    assert "class" in header.split(",") and len(rows) == 2
    meta = json.loads((tmp_path / "models" / "generator_2to3.json").read_text())
    assert meta["guided_by"] == str(guide_path)


def test_diversity_loss_penalises_fixed_position_stamps():
    """The training-time anti-stamp loss: high for the same change in every eye, low for eye-specific changes."""
    import tensorflow as tf

    from gan.cyclegan import _change_correlation
    from preprocessing.fundus import load_fundus, prepare_gan
    from tests.conftest import make_fundus

    x = np.stack([prepare_gan(load_fundus(make_fundus(seed))) for seed in range(4)])
    stamped = x.copy()
    stamped[:, 60:66, 90:96] = (0.2, -0.8, -0.8)
    varied = x.copy()
    for i, (r, c) in enumerate([(80, 70), (150, 170), (110, 200), (170, 60)]):
        varied[i, r : r + 6, c : c + 6] = (0.2, -0.8, -0.8)
    assert float(_change_correlation(tf.constant(x), tf.constant(stamped))) > 0.9
    assert float(_change_correlation(tf.constant(x), tf.constant(varied))) < 0.1


def test_rotation_augmentation_keeps_shape_and_range():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (3, config.GAN_SIZE, config.GAN_SIZE, 3), dtype=np.uint8)
    batch = next(T.batches(images, 2, rng, rotate=True))
    assert batch.shape == (2, config.GAN_SIZE, config.GAN_SIZE, 3) and batch.min() >= -1 and batch.max() <= 1
