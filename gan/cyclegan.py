"""CycleGAN training steps for one stage transition k -> k+1 (M4).

Unpaired training (PLAN.md): there are no follow-up photos of the same eye, so instead of Pix2Pix's paired
L1 loss, two generators are trained in opposite directions with
    LSGAN adversarial loss   - outputs must look like real images of the target stage
    cycle-consistency (L1)   - X -> Y -> X must return the input, so the retina's anatomy is preserved
    identity (L1)            - a target-stage image fed to G should come back unchanged
G (k -> k+1) is the generator the app uses; F (k+1 -> k) only exists for the cycle loss.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import keras

from gan.networks import SURROUND_LEVEL, build_discriminator, build_generator


class ImagePool:
    """History of generated images (Shrivastava et al. 2017, used by CycleGAN): the discriminators see a
    mix of current and past fakes, which damps oscillation between generator and discriminator."""

    def __init__(self, size: int = 50, seed: int = 0):
        self.size, self.images = size, []
        self.rng = np.random.default_rng(seed)

    def query(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for img in batch:
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif self.rng.random() < 0.5:
                i = self.rng.integers(self.size)
                out.append(self.images[i])
                self.images[i] = img
            else:
                out.append(img)
        return np.stack(out)


def _lsgan(pred, target_is_real: bool):
    target = tf.ones_like(pred) if target_is_real else tf.zeros_like(pred)
    return tf.reduce_mean(tf.square(tf.cast(pred, tf.float32) - target))


def _l1(a, b):
    return tf.reduce_mean(tf.abs(tf.cast(a, tf.float32) - tf.cast(b, tf.float32)))


def _guide_input(images: tf.Tensor, size: int) -> tf.Tensor:
    """Generator output in [-1, 1] at GAN_SIZE -> classifier input (0-255 at its own size), differentiably.

    Same mapping as preprocessing.fundus.cnn_input_from_gan_image: the image is already enhanced, so only resized.
    """
    pixels = (tf.cast(images, tf.float32) + 1.0) * 127.5
    # No antialias: its kernel has no XLA implementation, and the 256 -> 224 step barely aliases
    return tf.image.resize(pixels, (size, size), method="bilinear")


def _change_correlation(inputs: tf.Tensor, outputs: tf.Tensor) -> tf.Tensor:
    """Mean positive pairwise correlation between the images' change maps |output - input| within a batch.

    Classifier-guided generators learned to add the same dots at the same position in every eye; this is high for
    such a stamp and near 0 for eye-specific changes (numpy twin: utils.image_metrics.change_map_correlation).
    Only pixels inside each image's own field count, so the shared black surround adds no correlation.
    """
    inputs, outputs = tf.cast(inputs, tf.float32), tf.cast(outputs, tf.float32)
    inside = tf.cast(tf.reduce_max(inputs, axis=-1) > SURROUND_LEVEL, tf.float32)
    maps = tf.reduce_mean(tf.abs(outputs - inputs), axis=-1) * inside
    count = tf.maximum(tf.reduce_sum(inside, axis=[1, 2], keepdims=True), 1.0)
    centred = (maps - tf.reduce_sum(maps, axis=[1, 2], keepdims=True) / count) * inside
    flat = tf.reshape(centred, [tf.shape(centred)[0], -1])
    flat = flat / (tf.norm(flat, axis=1, keepdims=True) + 1e-6)
    corr = tf.matmul(flat, flat, transpose_b=True)
    off_diagonal = corr - tf.linalg.diag(tf.linalg.diag_part(corr))
    n = tf.cast(tf.shape(corr)[0], tf.float32)
    return tf.reduce_sum(tf.nn.relu(off_diagonal)) / tf.maximum(n * (n - 1.0), 1.0)


class CycleGAN:
    """CycleGAN for one transition, optionally classifier-guided.

    Plain CycleGAN generators barely changed the images: lesions occupy few 70 px patches, so the PatchGAN
    discriminators could not tell stage k+1 from stage k (their loss sat at the chance value 0.25), and the identity
    and cycle losses then make "copy the input" nearly optimal. With `guide` (a frozen stage classifier) the
    generators are also rewarded when the classifier grades G's output as stage k+1 and F's as stage k.

    Guidance alone then found a shortcut: one or two dots at a fixed position in every eye. `lambda_diversity`
    penalises change maps that correlate across the eyes of a batch, so changes have to depend on the input.
    """

    def __init__(
        self,
        base_filters: int = 64,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lr: float = 2e-4,
        guide: keras.Model | None = None,
        source_stage: int = 0,
        lambda_class: float = 1.0,
        lambda_diversity: float = 0.0,
    ):
        self.G = build_generator(base_filters, name="G_forward")  # stage k -> k+1 (shipped)
        self.F = build_generator(base_filters, name="F_backward")  # stage k+1 -> k
        self.D_Y = build_discriminator(name="D_target")
        self.D_X = build_discriminator(name="D_source")
        self.lambda_cycle, self.lambda_identity = lambda_cycle, lambda_identity
        self.guide, self.lambda_class, self.lambda_diversity = guide, lambda_class, lambda_diversity
        self.source_stage, self.target_stage = source_stage, source_stage + 1
        if guide is not None:
            guide.trainable = False
            self.guide_size = int(guide.input_shape[1])
        # Both optimizers share a learning-rate variable so the linear decay can be set from the training loop
        self.lr = tf.Variable(lr, trainable=False, dtype=tf.float32)
        self.opt_gen = keras.optimizers.Adam(learning_rate=lambda: self.lr, beta_1=0.5)
        self.opt_disc = keras.optimizers.Adam(learning_rate=lambda: self.lr, beta_1=0.5)
        self.gen_vars = self.G.trainable_variables + self.F.trainable_variables
        self.disc_vars = self.D_Y.trainable_variables + self.D_X.trainable_variables
        self.opt_gen.build(self.gen_vars)
        self.opt_disc.build(self.disc_vars)

    @tf.function(jit_compile=True)
    def generator_step(self, x, y):
        with tf.GradientTape() as tape:
            fake_y = self.G(x, training=True)
            fake_x = self.F(y, training=True)
            cycled_x = self.F(fake_y, training=True)
            cycled_y = self.G(fake_x, training=True)
            same_y = self.G(y, training=True)
            same_x = self.F(x, training=True)

            adversarial = _lsgan(self.D_Y(fake_y, training=True), True) + _lsgan(self.D_X(fake_x, training=True), True)
            cycle = _l1(x, cycled_x) + _l1(y, cycled_y)
            identity = _l1(y, same_y) + _l1(x, same_x)
            total = adversarial + self.lambda_cycle * cycle + self.lambda_identity * identity
            losses = {"g_adv": adversarial, "cycle": cycle, "identity": identity}
            if self.guide is not None:
                guided = self._class_loss(fake_y, self.target_stage) + self._class_loss(fake_x, self.source_stage)
                total = total + self.lambda_class * guided
                losses["class"] = guided
            if self.lambda_diversity > 0:
                diversity = _change_correlation(x, fake_y) + _change_correlation(y, fake_x)
                total = total + self.lambda_diversity * diversity
                losses["diversity"] = diversity
        grads = tape.gradient(total, self.gen_vars)
        self.opt_gen.apply(grads, self.gen_vars)
        return fake_y, fake_x, {"g_total": total, **losses}

    def _class_loss(self, images, stage: int):
        """Cross-entropy of the frozen classifier's grade against `stage` (0 when it is certain of that stage)."""
        probs = tf.cast(self.guide(_guide_input(images, self.guide_size), training=False), tf.float32)
        return -tf.reduce_mean(tf.math.log(probs[:, stage] + 1e-7))

    @tf.function(jit_compile=True)
    def discriminator_step(self, x, y, pooled_fake_x, pooled_fake_y):
        with tf.GradientTape() as tape:
            loss_y = 0.5 * (_lsgan(self.D_Y(y, training=True), True) + _lsgan(self.D_Y(pooled_fake_y, training=True), False))
            loss_x = 0.5 * (_lsgan(self.D_X(x, training=True), True) + _lsgan(self.D_X(pooled_fake_x, training=True), False))
            total = loss_y + loss_x
        grads = tape.gradient(total, self.disc_vars)
        self.opt_disc.apply(grads, self.disc_vars)
        return {"d_target": loss_y, "d_source": loss_x}

    def checkpoint(self, step: tf.Variable) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(
            G=self.G, F=self.F, D_Y=self.D_Y, D_X=self.D_X, opt_gen=self.opt_gen, opt_disc=self.opt_disc, lr=self.lr, step=step
        )
