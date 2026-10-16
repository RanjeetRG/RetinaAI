"""Preprocessed CNN input arrays (M3).

    python -m cnn.data                 # 224 px arrays for every split
    python -m cnn.data --size 384      # 384 px arrays (~12 GB for train)

Each split is run once through `prepare_cnn`, the same function the app calls, and stored as a uint8 array of
shape (N, size, size, 3) in manifest order, as <ARRAY_DIR>/cnn_<split>_<size>.npy. Training then only gathers rows;
augmentation runs on the GPU inside the training model. This module does not import TensorFlow, so the worker
processes stay light.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process (see preprocessing/build_dataset.py)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from preprocessing.fundus import prepare_cnn, read_image

SPLITS = ("train", "validation", "test", "external_test")


def array_path(split: str, size: int = config.CNN_SIZE) -> Path:
    return config.ARRAY_DIR / f"cnn_{split}_{size}.npy"


def meta_path(split: str, size: int = config.CNN_SIZE) -> Path:
    return config.ARRAY_DIR / f"cnn_{split}_{size}.csv"


def _init_worker() -> None:
    cv2.setNumThreads(1)


def _prepare(path: str, size: int) -> np.ndarray:
    return prepare_cnn(read_image(config.ROOT / path), size)


def build_arrays(split: str, size: int, workers: int) -> None:
    manifest = pd.read_csv(config.MANIFEST_DIR / f"{split}.csv")
    if manifest.empty:
        print(f"{split}: no images, skipped")
        return
    config.ARRAY_DIR.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        array_path(split, size), mode="w+", dtype=np.uint8, shape=(len(manifest), size, size, 3)
    )
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        images = pool.map(partial(_prepare, size=size), manifest["path"], chunksize=32)
        for i, img in enumerate(tqdm(images, total=len(manifest), desc=f"{split} {size}px")):
            out[i] = img
    out.flush()
    del out
    manifest[["path", "stage", "source"]].to_csv(meta_path(split, size), index=False)


@dataclass
class SplitArrays:
    images: np.ndarray  # (N, size, size, 3) uint8
    stages: np.ndarray  # (N,) int
    sources: np.ndarray  # (N,) str
    paths: np.ndarray  # (N,) str, relative to the repo root

    def __len__(self) -> int:
        return len(self.stages)


def load_split(split: str, size: int = config.CNN_SIZE, mmap: bool = False) -> SplitArrays:
    """Load a split's arrays, refusing ones built from an older version of the manifest."""
    meta = pd.read_csv(meta_path(split, size))
    manifest = pd.read_csv(config.MANIFEST_DIR / f"{split}.csv")
    if not meta["path"].equals(manifest["path"]):
        raise RuntimeError(f"{array_path(split, size)} is out of date with the manifest; rerun `python -m cnn.data`")
    images = np.load(array_path(split, size), mmap_mode="r" if mmap else None)
    return SplitArrays(images, meta["stage"].to_numpy(), meta["source"].to_numpy(), meta["path"].to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=config.CNN_SIZE, choices=config.CNN_SIZES)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    for split in args.splits:
        if (config.MANIFEST_DIR / f"{split}.csv").exists():
            build_arrays(split, args.size, args.workers)


if __name__ == "__main__":
    main()
