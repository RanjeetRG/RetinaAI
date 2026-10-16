"""Fixed random seeds for reproducible training runs (PRD: reproducibility)."""

import os
import random

import numpy as np

import config


def set_seed(seed: int = config.SEED, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow/Keras.

    deterministic_ops makes GPU kernels deterministic too, at a noticeable speed cost; use it for the
    final runs reported in the paper.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()
