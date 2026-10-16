"""Evaluate the transition generators (M4; PRD "Success metrics").

    python -m gan.gan_metrics                       # every models/generator_*.keras, judged by models/model.keras
    python -m gan.gan_metrics --transitions 0to1
    python -m gan.gan_metrics --transitions 1to2 --generator outputs/gan/_pilots/x/generator_1to2.keras         --method pilot_x --judge models/cnn_resnet50_224.keras --out outputs/gan/_pilots/x/metrics_resnet.csv

For each transition k -> k+1, held-out test images of stage k are run through the generator and scored:
    ssim, psnr           structure preserved between input and output (inside the fundus)
    fid, kid             realism: distance to real stage k+1 images (validation + test, never trained on)
    stage_consistency    share of outputs the CNN grades as k+1 (target >= 60%)
    mean_grade_shift     change in the CNN's expected grade (sum of stage x probability), output minus input
    global_shift         mean |change in an image's average colour| inside the fundus (0-255): exposure/colour
                         changes, a camera trait rather than disease
    local_change         mean |per-pixel change| after removing that global shift: local, lesion-like changes
    change_correlation   mean correlation of change maps between different eyes: ~1 for a fixed-position "stamp"
                         (a shortcut classifier-guided generators found), ~0 for eye-specific changes
The judge should differ from any classifier that guided training; a different architecture is the stronger test.
Each is also computed for the unchanged input ("identity" baseline), so the table shows how far the generator
moves images toward the next stage. FID/KID use Keras' ImageNet InceptionV3 pool features: comparable
across our models, not with FIDs published using the original TF-Inception weights.

Writes outputs/eval/gan/metrics.csv (+ .json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from scipy import linalg

import config
from cnn.model import input_size, load_classifier
from gan.data import load_stage
from gan.networks import load_generator
from gan.train import parse_transition
from preprocessing.fundus import cnn_input_from_gan_image, from_gan_range, to_gan_range
from utils.image_metrics import change_decomposition, change_map_correlation, psnr, ssim

INCEPTION_SIZE = 299


class InceptionFeatures:
    def __init__(self):
        self.model = keras.applications.InceptionV3(include_top=False, pooling="avg", weights="imagenet")

    def __call__(self, images_uint8: np.ndarray, batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(images_uint8), batch_size):
            batch = keras.ops.image.resize(images_uint8[i : i + batch_size].astype("float32"), (INCEPTION_SIZE, INCEPTION_SIZE))
            batch = keras.applications.inception_v3.preprocess_input(batch)
            feats.append(keras.ops.convert_to_numpy(self.model(batch, training=False)))
        return np.concatenate(feats).astype(np.float64)


def fid(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussian fits of two feature sets."""
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(cov_a @ cov_b))  # tiny imaginary parts come from numerical error
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * covmean))


def kid(a: np.ndarray, b: np.ndarray, subsets: int = 50, subset_size: int = 100, seed: int = 0) -> float:
    """Kernel Inception Distance (unbiased MMD^2, cubic polynomial kernel), averaged over random subsets.
    Unlike FID it is unbiased, so it stays meaningful for the few hundred stage-3/4 images available."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(subset_size, len(a), len(b))
    scores = []
    for _ in range(subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        k_xx = (x @ x.T / d + 1) ** 3
        k_yy = (y @ y.T / d + 1) ** 3
        k_xy = (x @ y.T / d + 1) ** 3
        scores.append(
            (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
            + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            - 2 * k_xy.mean()
        )
    return float(np.mean(scores))


def generate(generator: keras.Model, images_uint8: np.ndarray, batch_size: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(images_uint8), batch_size):
        fake = generator(to_gan_range(images_uint8[i : i + batch_size]), training=False)
        out.append(from_gan_range(keras.ops.convert_to_numpy(fake)))
    return np.concatenate(out)


def grade(cnn: keras.Model, images_uint8: np.ndarray) -> np.ndarray:
    """CNN probabilities for GAN-domain (256 px, already enhanced) images."""
    size = input_size(cnn)
    batch = np.stack([cnn_input_from_gan_image(img, size) for img in images_uint8]).astype("float32")
    return cnn.predict(batch, batch_size=64, verbose=0)


def score(name: str, k: int, inputs: np.ndarray, outputs: np.ndarray, reference_feats: np.ndarray,
          inception: InceptionFeatures, cnn: keras.Model, input_probs: np.ndarray) -> dict:  # fmt: skip
    feats = inception(outputs)
    probs = grade(cnn, outputs)
    expected = lambda p: (p * np.arange(config.NUM_STAGES)).sum(axis=1)  # noqa: E731
    global_shift, local_change = change_decomposition(inputs, outputs)
    return {
        "transition": f"{k}to{k + 1}",
        "method": name,
        "n": len(inputs),
        "ssim": float(np.mean([ssim(a, b) for a, b in zip(inputs, outputs)])),
        "psnr": float(np.mean([min(psnr(a, b), 100.0) for a, b in zip(inputs, outputs)])),
        "fid": fid(feats, reference_feats),
        "kid": kid(feats, reference_feats),
        "stage_consistency": float((probs.argmax(axis=1) == k + 1).mean()),
        "mean_grade_shift": float((expected(probs) - expected(input_probs)).mean()),
        "global_shift": global_shift,
        "local_change": local_change,
        "change_correlation": change_map_correlation(inputs, outputs),
    }


def evaluate(k: int, inception: InceptionFeatures, cnn: keras.Model, judge: str,
             generator_path=None, method: str = "cyclegan") -> list[dict]:  # fmt: skip
    inputs, _ = load_stage("test", k)
    reference = np.concatenate([load_stage("validation", k + 1)[0], load_stage("test", k + 1)[0]])
    reference_feats = inception(reference)
    input_probs = grade(cnn, inputs)
    rows = [score("identity (input unchanged)", k, inputs, inputs, reference_feats, inception, cnn, input_probs)]
    generator = load_generator(generator_path or config.generator_path(k))
    rows.append(score(method, k, inputs, generate(generator, inputs), reference_feats, inception, cnn, input_probs))
    for row in rows:
        row["reference_n"] = len(reference)
        row["judge"] = judge
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transitions", nargs="+", type=parse_transition)
    parser.add_argument("--judge", default=str(config.CNN_MODEL_PATH), help="classifier that grades the outputs")
    parser.add_argument("--generator", help="generator file to evaluate instead of models/ (one transition only)")
    parser.add_argument("--method", default="cyclegan", help="label for this generator in the table")
    parser.add_argument("--out", default=str(config.EVAL_DIR / "gan" / "metrics.csv"))
    args = parser.parse_args()
    transitions = args.transitions or [k for k in range(config.MAX_STAGE) if config.generator_path(k).exists()]
    if not transitions:
        raise SystemExit("No generators in models/. Run `python -m gan.train` first.")
    if args.generator and len(transitions) != 1:
        raise SystemExit("--generator needs exactly one --transitions value")

    inception, cnn = InceptionFeatures(), load_classifier(args.judge)
    judge = Path(args.judge).stem
    rows = [row for k in transitions for row in evaluate(k, inception, cnn, judge, args.generator, args.method)]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if path.exists():  # keep rows for other transitions, methods or judges
        old = pd.read_csv(path)
        keys = set(zip(table["transition"], table["method"], table["judge"]))
        old = old[[k not in keys for k in zip(old["transition"], old["method"], old.get("judge", judge))]]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["transition", "judge", "method"]).round(4)
    table.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(table.to_dict("records"), indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
