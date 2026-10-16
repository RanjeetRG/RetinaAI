"""Time classifier training steps under different settings, to find what makes training slow (run on the GPU).

    source scripts/wsl_env.sh
    python -m scripts.bench_cnn                   # 224 px
    python -m scripts.bench_cnn --size 384 --batch-size 16

Each variant runs a few warm-up steps and then times 30 training steps of EfficientNetB0 on random data:
    baseline        what cnn/train.py first did: float32 batches, GPU augmentation, jit_compile="auto"
    uint8_input     batches sent as uint8 (4x less host-to-GPU traffic), cast on the GPU
    no_augment      the same model without the augmentation layers
    xla             jit_compile=True forced
    xla_no_augment  both
"""

from __future__ import annotations

import argparse
import time

import keras
import numpy as np
from keras import layers

import config
from cnn.model import build_classifier


def augmentation() -> keras.Sequential:
    """The GPU augmentation cnn/train.py originally used, kept here for comparison runs."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.5, fill_mode="constant", fill_value=0.0),
            layers.RandomBrightness(0.1, value_range=(0, 255)),
            layers.RandomContrast(0.1, value_range=(0, 255)),
        ],
        name="augment",
    )


class ToFloat(layers.Layer):
    """Explicit cast: Keras does not auto-cast integer inputs."""

    def call(self, x):
        return keras.ops.cast(x, "float32")


def make_trainer(size: int, augment: bool, uint8_input: bool) -> keras.Model:
    inputs = keras.Input((size, size, 3), dtype="uint8" if uint8_input else "float32")
    x = ToFloat(dtype="float32")(inputs) if uint8_input else inputs  # cast on the GPU
    if augment:
        x = augmentation()(x)
    return keras.Model(inputs, build_classifier("efficientnetb0", input_size=size, weights=None)(x))


def time_variant(name: str, size: int, batch: int, augment: bool, uint8_input: bool, jit) -> None:
    keras.backend.clear_session()
    trainer = make_trainer(size, augment, uint8_input)
    trainer.compile(optimizer="adam", loss="sparse_categorical_crossentropy", jit_compile=jit)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (batch, size, size, 3), dtype=np.uint8)
    x = x if uint8_input else x.astype("float32")
    y = rng.integers(0, config.NUM_STAGES, batch)
    for _ in range(5):  # warm-up includes compilation
        trainer.train_on_batch(x, y)
    start = time.perf_counter()
    for _ in range(30):
        trainer.train_on_batch(x, y)
    per_step = (time.perf_counter() - start) / 30
    print(f"{name:<16} {per_step * 1000:7.1f} ms/step  {batch / per_step:7.1f} images/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=config.CNN_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"EfficientNetB0 training steps, {args.size} px, batch {args.batch_size}, mixed precision")
    variants = [
        ("baseline", True, False, "auto"),
        ("uint8_input", True, True, "auto"),
        ("no_augment", False, False, "auto"),
        ("xla", True, False, True),
        ("xla_no_augment", False, False, True),
    ]
    for name, augment, uint8_input, jit in variants:
        try:
            time_variant(name, args.size, args.batch_size, augment, uint8_input, jit)
        except Exception as exc:  # e.g. an augmentation op without an XLA kernel
            print(f"{name:<16} failed: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}", flush=True)


if __name__ == "__main__":
    main()
