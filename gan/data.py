"""GAN input arrays, one per split and stage (M4).

    python -m gan.data      # writes dataset/arrays/gan_<split>_<stage>.npy

Each image goes once through `prepare_gan` (the function the app uses: 512 -> 256 px, denoise, CLAHE) and is
stored as uint8; training rescales to [-1, 1]. Only images without quality warnings (`gan_ok`) are used, and
large stages are capped at a seeded random sample: a generator needs variety, not 18,000 healthy eyes.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process (see preprocessing/build_dataset.py)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from preprocessing.fundus import from_gan_range, prepare_gan, read_image

ARRAY_DIR = config.ARRAY_DIR
CAPS = {"train": 6000, "validation": 1000, "test": 1000}


def array_path(split: str, stage: int):
    return ARRAY_DIR / f"gan_{split}_{stage}.npy"


def meta_path(split: str, stage: int):
    return ARRAY_DIR / f"gan_{split}_{stage}.csv"


def _init_worker() -> None:
    cv2.setNumThreads(1)


def _prepare(path: str) -> np.ndarray:
    return from_gan_range(prepare_gan(read_image(config.ROOT / path)))


def select(split: str, stage: int, cap: int, seed: int = config.SEED) -> pd.DataFrame:
    manifest = pd.read_csv(config.MANIFEST_DIR / f"{split}.csv")
    rows = manifest[(manifest["stage"] == stage) & manifest["gan_ok"]]
    if len(rows) > cap:
        rows = rows.sample(cap, random_state=seed)
    return rows.sort_values("path")


def build_arrays(split: str, stage: int, workers: int) -> None:
    rows = select(split, stage, CAPS[split])
    if rows.empty:
        return
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    s = config.GAN_SIZE
    out = np.lib.format.open_memmap(array_path(split, stage), mode="w+", dtype=np.uint8, shape=(len(rows), s, s, 3))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        images = pool.map(_prepare, rows["path"], chunksize=16)
        for i, img in enumerate(tqdm(images, total=len(rows), desc=f"{split} stage {stage}")):
            out[i] = img
    out.flush()
    del out
    rows[["path", "stage", "source"]].to_csv(meta_path(split, stage), index=False)


def load_stage(split: str, stage: int) -> tuple[np.ndarray, np.ndarray]:
    """uint8 images (N, GAN_SIZE, GAN_SIZE, 3) and their paths; refuses arrays from an older manifest."""
    meta = pd.read_csv(meta_path(split, stage))
    expected = select(split, stage, CAPS[split])
    if not meta["path"].reset_index(drop=True).equals(expected["path"].reset_index(drop=True)):
        raise RuntimeError(f"{array_path(split, stage)} is out of date with the manifest; rerun `python -m gan.data`")
    return np.load(array_path(split, stage)), meta["path"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=list(CAPS), choices=CAPS)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    for split in args.splits:
        for stage in range(config.NUM_STAGES):
            build_arrays(split, stage, args.workers)


if __name__ == "__main__":
    main()
