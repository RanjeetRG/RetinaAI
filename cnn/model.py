"""Stage classifier: an ImageNet backbone plus a shared 5-class head (M3).

Every saved classifier takes square RGB images with values 0-255 at its own input size (224 or 384 px; see
input_size). Each backbone's own input scaling is a layer inside the model, so the app never needs to know which
backbone or size it is running.
"""

from __future__ import annotations

import keras
from keras import layers, ops

import config

# name -> (constructor, input scaling mode)
BACKBONES = {
    "efficientnetb0": (keras.applications.EfficientNetB0, "none"),  # rescales internally
    "resnet50": (keras.applications.ResNet50, "caffe"),
    "densenet121": (keras.applications.DenseNet121, "torch"),
    "mobilenetv2": (keras.applications.MobileNetV2, "tf"),
}

_CAFFE_BGR_MEAN = (103.939, 116.779, 123.68)
_TORCH_MEAN = (0.485, 0.456, 0.406)
_TORCH_STD = (0.229, 0.224, 0.225)


@keras.saving.register_keras_serializable(package="retinagen")
class BackboneNormalization(layers.Layer):
    """The keras.applications preprocess_input modes, as a serializable layer."""

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        if mode not in {"none", "caffe", "torch", "tf"}:
            raise ValueError(f"Unknown normalization mode {mode!r}")
        self.mode = mode

    def call(self, x):
        x = ops.cast(x, self.compute_dtype)
        if self.mode == "caffe":
            x = ops.flip(x, axis=-1) - ops.convert_to_tensor(_CAFFE_BGR_MEAN, dtype=x.dtype)
        elif self.mode == "torch":
            mean = ops.convert_to_tensor(_TORCH_MEAN, dtype=x.dtype)
            std = ops.convert_to_tensor(_TORCH_STD, dtype=x.dtype)
            x = (x / 255.0 - mean) / std
        elif self.mode == "tf":
            x = x / 127.5 - 1.0
        return x

    def get_config(self):
        return {**super().get_config(), "mode": self.mode}


def build_classifier(
    backbone: str = "efficientnetb0",
    dropout: float = 0.3,
    weights: str | None = "imagenet",
    input_size: int = config.CNN_SIZE,
) -> keras.Model:
    constructor, mode = BACKBONES[backbone]
    size = input_size
    inputs = keras.Input((size, size, 3), name="image")
    x = BackboneNormalization(mode, name="normalize")(inputs)
    base = constructor(include_top=False, weights=weights, input_shape=(size, size, 3))
    # training=False keeps BatchNorm statistics frozen even after the backbone is unfrozen for fine-tuning
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    # float32 output so softmax stays stable under mixed precision
    outputs = layers.Dense(config.NUM_STAGES, activation="softmax", dtype="float32", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"dr_{backbone}")


def input_size(model: keras.Model) -> int:
    """The square input size a saved classifier was built for (224, 384, ...)."""
    return int(model.input_shape[1])


def get_backbone(model: keras.Model) -> keras.Model:
    """The nested keras.applications model (used to freeze/unfreeze and for Grad-CAM)."""
    return next(layer for layer in model.layers if isinstance(layer, keras.Model))


def load_classifier(path=None) -> keras.Model:
    # Importing this module registers BackboneNormalization, which loading needs
    return keras.models.load_model(path or config.CNN_MODEL_PATH, compile=False)
