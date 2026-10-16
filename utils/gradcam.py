"""Grad-CAM for the stage classifier (FR-5): which regions pushed the model toward its grade.

Uses the backbone's final convolutional feature map (7x7 for every backbone at 224 px): the gradient of the
chosen class score, averaged over space, weights each feature channel (Selvaraju et al. 2017).
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from cnn.model import get_backbone


def gradcam(model: keras.Model, image_uint8: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Importance map in [0, 1] at the feature-map resolution (resize it for display)."""
    backbone = get_backbone(model)
    normalize, pool, head = model.get_layer("normalize"), model.get_layer("pool"), model.get_layer("probs")
    x = tf.convert_to_tensor(image_uint8[None].astype("float32"))
    with tf.GradientTape() as tape:
        features = backbone(normalize(x), training=False)
        tape.watch(features)
        probs = head(pool(features))  # dropout is inactive at inference, so it is skipped
        k = int(tf.argmax(probs[0])) if class_index is None else class_index
        score = probs[:, k]
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2))  # one weight per channel
    cam = tf.nn.relu(tf.reduce_sum(features * weights[:, None, None, :], axis=-1))[0].numpy()
    return cam / cam.max() if cam.max() > 0 else cam
