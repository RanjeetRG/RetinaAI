import numpy as np
import pytest

keras = pytest.importorskip("keras")

import config  # noqa: E402
from cnn.model import BACKBONES, BackboneNormalization, build_classifier, get_backbone, input_size, load_classifier  # noqa: E402

PREPROCESS = {
    "caffe": keras.applications.resnet50.preprocess_input,
    "torch": keras.applications.densenet.preprocess_input,
    "tf": keras.applications.mobilenet_v2.preprocess_input,
}


@pytest.mark.parametrize("mode", PREPROCESS)
def test_normalization_matches_keras_preprocess_input(mode):
    x = np.random.default_rng(0).uniform(0, 255, (2, 8, 8, 3)).astype("float32")
    expected = PREPROCESS[mode](x.copy())
    actual = keras.ops.convert_to_numpy(BackboneNormalization(mode)(x))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_classifier_outputs_probabilities(backbone):
    model = build_classifier(backbone, weights=None)
    x = np.random.default_rng(0).uniform(0, 255, (2, config.CNN_SIZE, config.CNN_SIZE, 3)).astype("float32")
    probs = keras.ops.convert_to_numpy(model(x))
    assert probs.shape == (2, config.NUM_STAGES)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert isinstance(get_backbone(model), keras.Model)


def test_save_and_load_round_trip(tmp_path):
    model = build_classifier("mobilenetv2", weights=None)
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = load_classifier(path)
    x = np.random.default_rng(1).uniform(0, 255, (1, config.CNN_SIZE, config.CNN_SIZE, 3)).astype("float32")
    np.testing.assert_allclose(keras.ops.convert_to_numpy(loaded(x)), keras.ops.convert_to_numpy(model(x)), atol=1e-6)


def test_input_size_is_recorded_in_the_model(tmp_path):
    model = build_classifier("mobilenetv2", weights=None, input_size=160)
    path = tmp_path / "m.keras"
    model.save(path)
    assert input_size(load_classifier(path)) == 160
