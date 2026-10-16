"""Sample figures from the trained generators (M4 wrap-up, paper figures).

    python -m gan.generate

outputs/eval/gan/samples_<k>to<k+1>.jpg   held-out test images of stage k | simulated stage k+1 | difference x4
outputs/eval/gan/chain.jpg                 stage-0 test images run through every generator, 0 -> 1 -> 2 -> 3 -> 4
"""

from __future__ import annotations

import argparse

import numpy as np

import config
from gan.data import load_stage
from gan.gan_metrics import generate
from gan.networks import load_generator
from utils.figures import header, row_of, save_jpeg, stack

OUT_DIR = config.EVAL_DIR / "gan"


def amplified_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(a.astype(int) - b.astype(int)) * 4, 0, 255).astype(np.uint8)


def transition_figure(k: int, n: int, rng: np.random.Generator) -> None:
    images, _ = load_stage("test", k)
    picks = images[np.sort(rng.choice(len(images), min(n, len(images)), replace=False))]
    fakes = generate(load_generator(config.generator_path(k)), picks)
    rows = [row_of([a, b, amplified_difference(a, b)], f"Test image {i + 1}") for i, (a, b) in enumerate(zip(picks, fakes))]
    body = stack(rows)
    titles = [f"Input (stage {k})", f"Simulated stage {k + 1}", "Difference x4"]
    save_jpeg(OUT_DIR / f"samples_{k}to{k + 1}.jpg", stack([header(titles, body.shape[1]), body]))


def chain_figure(n: int, rng: np.random.Generator) -> None:
    generators = [load_generator(config.generator_path(k)) for k in range(config.MAX_STAGE)]
    images, _ = load_stage("test", 0)
    current = images[np.sort(rng.choice(len(images), n, replace=False))]
    columns = [current]
    for g in generators:
        current = generate(g, current)
        columns.append(current)
    rows = [row_of([col[i] for col in columns], f"Test image {i + 1}") for i in range(n)]
    body = stack(rows)
    titles = ["Input (stage 0)"] + [f"Simulated {k}" for k in range(1, config.NUM_STAGES)]
    save_jpeg(OUT_DIR / "chain.jpg", stack([header(titles, body.shape[1]), body]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=6, help="images per figure")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    available = [k for k in range(config.MAX_STAGE) if config.generator_path(k).exists()]
    for k in available:
        transition_figure(k, args.n, rng)
    if len(available) == config.MAX_STAGE:
        chain_figure(min(args.n, 4), rng)


if __name__ == "__main__":
    main()
