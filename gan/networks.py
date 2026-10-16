"""Transition generator and discriminator (M4).

Generator: Pix2Pix-style U-Net (8 down/up levels at 256 px, skip connections), but every upsampling step is
resize-then-convolve instead of a transposed convolution, which removes the checkerboard artifacts listed in
the PRD's risks. Instance normalization, as in CycleGAN. Its output is masked to the input's fundus
(FundusMask), so it can only change the retina, never the black surround.

Discriminator: 70x70 PatchGAN; outputs a 30x30 grid of real/fake logits, one per overlapping patch.
"""

from __future__ import annotations

import keras
from keras import layers, ops

import config

INIT = keras.initializers.RandomNormal(0.0, 0.02)  # DCGAN/Pix2Pix weight init
# Pixels of the input at or below this (in [-1, 1]) are the black surround; preprocessing sets it to exactly -1
SURROUND_LEVEL = -0.95


@keras.saving.register_keras_serializable(package="retinagen")
class FundusMask(layers.Layer):
    """Keeps generated pixels inside the input's fundus and forces the surround back to black (-1).

    Without it the first 0->1 run (Oct 2026) learned to paint saturated blobs in the empty area outside the
    retina from step 10,000 on: an easy place for a CycleGAN generator to put content or hide information
    for the cycle loss, and not a fundus feature.
    """

    def call(self, inputs):
        image, generated = inputs
        inside = ops.cast(ops.max(image, axis=-1, keepdims=True) > SURROUND_LEVEL, generated.dtype)
        return generated * inside - (1.0 - inside)


def instance_norm(channels: int, name: str) -> layers.Layer:
    # Group norm with one group per channel normalizes each image and channel separately
    return layers.GroupNormalization(groups=channels, epsilon=1e-5, name=name)


def _down(x, filters: int, name: str, norm: bool = True):
    x = layers.Conv2D(filters, 4, strides=2, padding="same", use_bias=not norm, kernel_initializer=INIT, name=f"{name}_conv")(x)
    if norm:
        x = instance_norm(filters, f"{name}_norm")(x)
    return layers.LeakyReLU(0.2, name=f"{name}_act")(x)


def _up(x, skip, filters: int, name: str):
    x = layers.UpSampling2D(2, interpolation="nearest", name=f"{name}_resize")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_initializer=INIT, name=f"{name}_conv")(x)
    x = instance_norm(filters, f"{name}_norm")(x)
    x = layers.ReLU(name=f"{name}_act")(x)
    return layers.Concatenate(name=f"{name}_skip")([x, skip])


def build_generator(base_filters: int = 64, name: str = "generator") -> keras.Model:
    """U-Net: (GAN_SIZE, GAN_SIZE, 3) in [-1, 1] -> same shape in [-1, 1]."""
    f = base_filters
    down_filters = [f, f * 2, f * 4, f * 8, f * 8, f * 8, f * 8]  # 256 -> 2 px
    inputs = keras.Input((config.GAN_SIZE, config.GAN_SIZE, 3), name="image")
    x, skips = inputs, []
    for i, filters in enumerate(down_filters):
        x = _down(x, filters, f"down{i}", norm=i > 0)  # no norm on the first layer (Pix2Pix)
        skips.append(x)
    # Bottleneck at 1x1: instance norm over a single pixel would zero the activations, so none here
    x = layers.Conv2D(f * 8, 4, strides=2, padding="same", kernel_initializer=INIT, name="bottleneck_conv")(x)
    x = layers.ReLU(name="bottleneck_act")(x)
    for i, (filters, skip) in enumerate(zip(reversed(down_filters), reversed(skips))):
        x = _up(x, skip, filters, f"up{i}")
    x = layers.UpSampling2D(2, interpolation="nearest", name="out_resize")(x)
    x = layers.Conv2D(3, 3, padding="same", kernel_initializer=INIT, name="out_conv")(x)
    # float32 output keeps tanh accurate under mixed precision
    x = layers.Activation("tanh", dtype="float32", name="out")(x)
    outputs = FundusMask(dtype="float32", name="fundus_mask")([inputs, x])
    return keras.Model(inputs, outputs, name=name)


def build_discriminator(base_filters: int = 64, name: str = "discriminator") -> keras.Model:
    """70x70 PatchGAN: image in [-1, 1] -> (30, 30, 1) logits (LSGAN, so no sigmoid)."""
    f = base_filters
    inputs = keras.Input((config.GAN_SIZE, config.GAN_SIZE, 3), name="image")
    x = _down(inputs, f, "d0", norm=False)
    x = _down(x, f * 2, "d1")
    x = _down(x, f * 4, "d2")  # 32 x 32
    # Two stride-1 4x4 convs with 1 px padding (32 -> 31 -> 30), as in Pix2Pix, give a 70x70 receptive field
    x = layers.ZeroPadding2D(1, name="d3_pad")(x)
    x = layers.Conv2D(f * 8, 4, strides=1, padding="valid", use_bias=False, kernel_initializer=INIT, name="d3_conv")(x)
    x = instance_norm(f * 8, "d3_norm")(x)
    x = layers.LeakyReLU(0.2, name="d3_act")(x)
    x = layers.ZeroPadding2D(1, name="out_pad")(x)
    outputs = layers.Conv2D(1, 4, strides=1, padding="valid", kernel_initializer=INIT, dtype="float32", name="patch_logits")(x)
    return keras.Model(inputs, outputs, name=name)


def load_generator(path) -> keras.Model:
    return keras.models.load_model(path, compile=False)
