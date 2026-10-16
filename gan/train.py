"""Train one transition generator k -> k+1 (M4). In WSL on the GPU:

    source scripts/wsl_env.sh
    python -m gan.train --transition 0to1 --guide models/cnn_efficientnetb0_224.keras

With --guide, a frozen stage classifier also rewards outputs it grades as the target stage (see gan/cyclegan.py);
without it this is plain CycleGAN, kept as the baseline. Every transition gets the same step budget (PRD risk: uneven
budgets make transitions incomparable). The
learning rate is constant for the first half and decays linearly to zero over the second, as in CycleGAN.
Checkpoints are kept in outputs/gan/<transition>/checkpoints/, so rerunning the same command resumes.

Outputs: models/generator_<k>to<k+1>.keras (+ .json), outputs/gan/<transition>/history.csv and
samples/step_*.jpg (validation inputs next to their simulated next stage).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone

import cv2
import keras
import numpy as np
import tensorflow as tf

import config
from cnn.model import load_classifier
from gan.cyclegan import CycleGAN, ImagePool
from gan.data import load_stage
from preprocessing.fundus import from_gan_range, to_gan_range
from utils.seed import set_seed

PREVIEW_IMAGES = 4


def parse_transition(text: str) -> int:
    """'1to2' -> 1 (the source stage)."""
    src, dst = (int(s) for s in text.split("to"))
    if dst != src + 1 or not 0 <= src < config.MAX_STAGE:
        raise argparse.ArgumentTypeError("transition must be one of 0to1, 1to2, 2to3, 3to4")
    return src


def batches(images: np.ndarray, batch_size: int, rng: np.random.Generator, rotate: bool = False):
    """Endless random batches in [-1, 1], with random horizontal flips (left/right eyes are mirror images).

    rotate: also rotate each image by a random angle, so the generator cannot anchor a change to a fixed position
    relative to the standard field (guided generators placed their dots at the same spot in every eye).
    """
    size = images.shape[1]
    while True:
        idx = rng.integers(len(images), size=batch_size)
        batch = images[idx]
        flip = rng.random(batch_size) < 0.5
        batch[flip] = batch[flip, :, ::-1]
        if rotate:
            for i in range(batch_size):
                m = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(-180, 180), 1.0)
                batch[i] = cv2.warpAffine(np.ascontiguousarray(batch[i]), m, (size, size), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)  # fmt: skip
        yield to_gan_range(batch)


def learning_rate(step: int, total: int, base: float) -> float:
    half = total // 2
    return base if step < half else base * max(0.0, (total - step) / (total - half))


def save_preview(generator: keras.Model, preview: np.ndarray, path) -> None:
    """Rows: input | simulated next stage | amplified absolute difference."""
    fake = from_gan_range(keras.ops.convert_to_numpy(generator(to_gan_range(preview), training=False)))
    diff = np.clip(np.abs(fake.astype(int) - preview.astype(int)) * 4, 0, 255).astype(np.uint8)
    rows = [np.hstack([p, f, d]) for p, f, d in zip(preview, fake, diff)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def train(args: argparse.Namespace, source: np.ndarray, target: np.ndarray, preview: np.ndarray) -> None:
    k = args.transition
    name = f"{k}to{k + 1}"
    run_dir = config.OUTPUTS_DIR / "gan" / name
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    guide = load_classifier(args.guide) if args.guide else None
    gan = CycleGAN(
        args.base_filters, args.lambda_cycle, args.lambda_identity, args.lr,
        guide=guide, source_stage=k, lambda_class=args.lambda_class, lambda_diversity=args.lambda_diversity,
    )  # fmt: skip
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    manager = tf.train.CheckpointManager(gan.checkpoint(step), run_dir / "checkpoints", max_to_keep=2)
    if manager.latest_checkpoint:
        manager.checkpoint.restore(manager.latest_checkpoint)
        print(f"resumed from {manager.latest_checkpoint} at step {int(step)}", flush=True)

    pool_x, pool_y = ImagePool(seed=args.seed), ImagePool(seed=args.seed + 1)
    x_batches = batches(source, args.batch_size, rng, args.rotate)
    y_batches = batches(target, args.batch_size, rng, args.rotate)
    history_path = run_dir / "history.csv"
    sums, count, tick = {}, 0, time.time()

    while int(step) < args.steps:
        s = int(step)
        gan.lr.assign(learning_rate(s, args.steps, args.lr))
        x, y = next(x_batches), next(y_batches)
        fake_y, fake_x, g_losses = gan.generator_step(x, y)
        d_losses = gan.discriminator_step(x, y, pool_x.query(fake_x.numpy()), pool_y.query(fake_y.numpy()))
        step.assign_add(1)

        for key, value in {**g_losses, **d_losses}.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
        if (s + 1) % args.log_every == 0 or s + 1 == args.steps:
            row = {"step": s + 1, "lr": float(gan.lr.numpy()), **{k_: v / count for k_, v in sums.items()}}
            row["steps_per_s"] = count / (time.time() - tick)
            new_file = not history_path.exists()
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            print(" ".join(f"{k_} {v}" if isinstance(v, int) else f"{k_} {v:.4g}" for k_, v in row.items()), flush=True)
            sums, count, tick = {}, 0, time.time()
        if (s + 1) % args.sample_every == 0 or s + 1 == args.steps:
            save_preview(gan.G, preview, run_dir / "samples" / f"step_{s + 1:06d}.jpg")
        if (s + 1) % args.checkpoint_every == 0:
            manager.save()

    manager.save()
    out_path = config.generator_path(k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gan.G.save(out_path)
    meta = {
        "transition": name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": args.steps,
        "guided_by": str(args.guide) if args.guide else None,
        "source_images": len(source),
        "target_images": len(target),
        "args": vars(args),
        "versions": {"keras": keras.__version__, "tensorflow": tf.__version__},
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transition", type=parse_transition, required=True, help="0to1, 1to2, 2to3 or 3to4")
    parser.add_argument("--steps", type=int, default=40_000, help="same budget for every transition")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-filters", type=int, default=32, help="32 gives a 39 MB generator; 64 is 157 MB")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--guide", help="frozen classifier (.keras) for classifier guidance; omit for plain CycleGAN")
    parser.add_argument("--lambda-class", type=float, default=1.0, help="weight of the classifier-guidance loss")
    parser.add_argument("--lambda-diversity", type=float, default=0.0,
                        help="weight of the penalty on change maps that repeat across eyes (anti-stamp)")
    parser.add_argument("--rotate", action="store_true", help="random-angle rotation augmentation of training images")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=2_000)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    k = args.transition
    source, _ = load_stage("train", k)
    target, _ = load_stage("train", k + 1)
    preview, _ = load_stage("validation", k)
    print(f"transition {k}->{k + 1}: {len(source)} source and {len(target)} target images", flush=True)
    train(args, source, target, preview[:PREVIEW_IMAGES])


if __name__ == "__main__":
    main()
