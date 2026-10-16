"""Build the cached, deduplicated, patient-level split dataset (M2).

    python -m preprocessing.build_dataset

1. Collect labelled images from dataset/raw/{aptos,eyepacs,messidor2}.
2. Cache each one at CACHE_SIZE with `load_fundus` (dataset/cache/<source>/<id>.jpg) and record quality
   metrics and a perceptual hash.
3. Drop unreadable and unusable images, and near-duplicates (APTOS is known to contain some, a few with
   conflicting labels: those groups are dropped entirely).
4. Split APTOS + EyePACS 70/15/15 by patient, stratified by source and stage; both eyes of a patient stay
   in one split. Messidor-2 becomes the external test set only.
5. Hard-link the cached files into dataset/<split>/<stage>/ and write manifests to dataset/manifests/.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process. By default every worker starts one OpenBLAS thread per CPU core,
# each with its own memory buffer, and 18 workers ran a 24 GB laptop out of memory. Must run before numpy
# is imported; spawned workers inherit the environment.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from preprocessing.fundus import load_fundus, read_image
from preprocessing.quality import check_quality

CACHE_DIR = config.DATASET_DIR / "cache"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
JPEG_QUALITY = 95
HASH_SIZE = 16  # 256-bit perceptual hash
DUPLICATE_MAX_DISTANCE = 20  # bits out of 256; provisional, check the distance histogram this script prints
TRAINING_SOURCES = ("aptos", "eyepacs")


# ---------------------------------------------------------------- collect


def index_images(folder: Path) -> dict[str, Path]:
    """Map lower-case file stem -> path for every image under folder."""
    return {p.stem.lower(): p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}


def collect_aptos() -> pd.DataFrame:
    root = config.RAW_DIR / "aptos"
    labels = pd.read_csv(root / "train.csv")
    images = index_images(root / "train_images")
    return pd.DataFrame(
        {
            "source": "aptos",
            "image_id": labels["id_code"],
            "raw_path": labels["id_code"].str.lower().map(images),
            "stage": labels["diagnosis"],
            "patient_id": "aptos_" + labels["id_code"],  # APTOS publishes no patient ids
        }
    )


def collect_eyepacs() -> pd.DataFrame:
    root = config.RAW_DIR / "eyepacs"
    labels = pd.read_csv(root / "trainLabels.csv")
    images = index_images(root)
    return pd.DataFrame(
        {
            "source": "eyepacs",
            "image_id": labels["image"],
            "raw_path": labels["image"].str.lower().map(images),
            "stage": labels["level"],
            "patient_id": "eyepacs_" + labels["image"].str.split("_").str[0],  # "10_left" -> patient 10
        }
    )


def collect_messidor() -> pd.DataFrame:
    root = config.RAW_DIR / "messidor2"
    labels = pd.read_csv(root / "messidor_data.csv")
    labels = labels[(labels["adjudicated_gradable"] == 1) & labels["adjudicated_dr_grade"].notna()]
    stems = labels["image_id"].map(lambda s: Path(s).stem)
    images = index_images(root / "images")
    return pd.DataFrame(
        {
            "source": "messidor2",
            "image_id": stems,
            "raw_path": stems.str.lower().map(images),
            "stage": labels["adjudicated_dr_grade"].astype(int),
            "patient_id": "messidor2_" + stems,
        }
    )


COLLECTORS = {"aptos": collect_aptos, "eyepacs": collect_eyepacs, "messidor2": collect_messidor}


def collect(sources: list[str]) -> pd.DataFrame:
    frames = []
    for name in sources:
        try:
            df = COLLECTORS[name]()
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")
            continue
        # The 512 px cache is all later steps read, so an image whose original was deleted to save disk space is
        # still usable as long as it is cached
        cached = [(CACHE_DIR / s / f"{i}.jpg").exists() for s, i in zip(df["source"], df["image_id"])]
        usable = df["raw_path"].notna() | pd.Series(cached, index=df.index)
        if (~usable).sum():
            print(f"{name}: {(~usable).sum()} labelled images have no file and are skipped")
        frames.append(df[usable])
        print(f"{name}: {len(frames[-1])} images")
    if not frames:
        raise SystemExit("No raw data found. Run `python -m scripts.download_data` first.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- cache + measure


def content_hash(img: np.ndarray) -> str:
    """Perceptual hash of the central 70% of the green channel.

    The black surround and circular edge look the same in every fundus photo, so hashing the whole frame
    would make unrelated images look like duplicates.
    """
    s = img.shape[0]
    m = int(s * 0.15)
    centre = img[m : s - m, m : s - m, 1]
    return str(imagehash.phash(Image.fromarray(centre), hash_size=HASH_SIZE))


def write_jpeg(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    # Write then rename, so a crash never leaves a truncated JPEG that a rerun would trust
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, path)


def process_one(row: dict) -> dict:
    cache_path = CACHE_DIR / row["source"] / f"{row['image_id']}.jpg"
    try:
        if not cache_path.exists():
            write_jpeg(cache_path, load_fundus(row["raw_path"]))
        # Measure the cached file itself, so reruns give identical numbers
        img = read_image(cache_path)
    except Exception as exc:  # unreadable or corrupt raw file
        return {**row, "error": str(exc)}
    report = check_quality(img)
    return {
        **row,
        "cache_path": cache_path,
        "phash": content_hash(img),
        "quality_issues": ";".join(report.issues),
        "unusable": report.unusable,
        "gan_ok": report.ok,  # GAN training uses only images without any quality issue
        **report.metrics,
    }


def _init_worker() -> None:
    cv2.setNumThreads(1)  # parallelism comes from the process pool


def cache_all(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    rows = df.to_dict("records")
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = list(tqdm(pool.map(process_one, rows, chunksize=16), total=len(rows), desc="Caching"))
    out = pd.DataFrame(results)
    if "error" not in out:
        out["error"] = np.nan
    return out


# ---------------------------------------------------------------- deduplicate


def duplicate_groups(hashes: list[str], max_distance: int) -> tuple[np.ndarray, np.ndarray]:
    """Union-find over all pairs within max_distance bits.

    Returns (group id per image, nearest-neighbour distance per image) for the histogram printed below.
    """
    bits = np.array([imagehash.hex_to_hash(h).hash.flatten() for h in hashes])
    packed = np.ascontiguousarray(np.packbits(bits, axis=1)).view(np.uint64)  # (N, HASH_SIZE**2 / 64)
    n = len(hashes)
    parent = np.arange(n)
    nearest = np.full(n, HASH_SIZE * HASH_SIZE)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    chunk = 64
    for start in range(0, n, chunk):
        block = packed[start : start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ packed[None, :, :]).sum(axis=-1)
        rows = np.arange(len(block))
        dist[rows, start + rows] = HASH_SIZE * HASH_SIZE  # ignore self-distance
        nearest[start : start + len(block)] = dist.min(axis=1)
        for i, j in zip(*np.nonzero(dist <= max_distance)):
            a, b = find(start + i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups = np.array([find(i) for i in range(n)])
    return groups, nearest


def mark_duplicates(df: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    """Within each near-duplicate group keep the sharpest image, or drop the group if labels disagree."""
    df = df.copy()
    df["exclude_reason"] = np.where(df["error"].notna(), "unreadable", np.where(df["unusable"] == True, "unusable", ""))  # noqa: E712
    candidates = df[(df["exclude_reason"] == "") & df["source"].isin(TRAINING_SOURCES)]
    if len(candidates) < 2:
        return df
    groups, nearest = duplicate_groups(candidates["phash"].tolist(), max_distance)
    df.loc[candidates.index, "dup_group"] = groups
    df.loc[candidates.index, "nearest_hash_distance"] = nearest

    hist = np.bincount(np.minimum(nearest, 40), minlength=41)
    print("Nearest-neighbour hash distance (bits): count")
    print("  " + ", ".join(f"{d}:{c}" for d, c in enumerate(hist) if c))

    for _, grp in df.loc[candidates.index].groupby("dup_group"):
        if len(grp) < 2:
            continue
        if grp["stage"].nunique() > 1:
            df.loc[grp.index, "exclude_reason"] = "duplicate_conflicting_labels"
        else:
            keep = grp["laplacian_var"].idxmax()
            df.loc[grp.index.drop(keep), "exclude_reason"] = "duplicate"
    return df


# ---------------------------------------------------------------- split


def assign_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    """Patient-level 70/15/15 split stratified by (source, worst stage of the patient)."""
    patients = df.groupby("patient_id").agg(stage=("stage", "max"), source=("source", "first"))
    strata = (patients["source"] + "_" + patients["stage"].astype(str)).to_numpy(dtype=object)
    fr = config.SPLIT_FRACTIONS
    holdout = fr["validation"] + fr["test"]
    # Split integer positions: scikit-learn cannot index pandas 3's Arrow-backed string index
    positions = np.arange(len(patients))
    _, rest = train_test_split(positions, test_size=holdout, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=fr["test"] / holdout, stratify=strata[rest], random_state=seed)
    split_of = pd.Series("train", index=patients.index)
    split_of.iloc[val] = "validation"
    split_of.iloc[test] = "test"
    return df["patient_id"].map(split_of)


# ---------------------------------------------------------------- write


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-link cached images into dataset/<split>/<stage>/ (no extra disk space)."""
    for split_dir in config.SPLIT_DIRS.values():
        for stage in range(config.NUM_STAGES):
            shutil.rmtree(split_dir / str(stage), ignore_errors=True)
            (split_dir / str(stage)).mkdir(parents=True)
    paths = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Linking"):
        dst = config.SPLIT_DIRS[row.split] / str(row.stage) / f"{row.source}_{row.image_id}.jpg"
        link_or_copy(Path(row.cache_path), dst)
        paths.append(dst.relative_to(config.ROOT).as_posix())  # posix paths work from Windows and WSL
    df = df.copy()
    df["path"] = paths
    return df


QUALITY_COLUMNS = ["fundus_fraction", "corner_lit", "mean_brightness", "laplacian_var", "red_blue_ratio"]
MANIFEST_COLUMNS = [
    "path", "split", "stage", "source", "image_id", "patient_id", "gan_ok", "quality_issues", *QUALITY_COLUMNS,
]  # fmt: skip
AUDIT_COLUMNS = [
    "source", "image_id", "patient_id", "stage", "split", "exclude_reason", "phash", "dup_group",
    "nearest_hash_distance", "quality_issues", *QUALITY_COLUMNS, "error", "raw_path",
]  # fmt: skip


def write_outputs(kept: pd.DataFrame, excluded: pd.DataFrame) -> None:
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for split in config.SPLIT_DIRS:
        part = kept[kept["split"] == split]
        part[MANIFEST_COLUMNS].sort_values("path").to_csv(config.MANIFEST_DIR / f"{split}.csv", index=False)
    # Every collected image with its split or exclusion reason, hash and duplicate group, for auditing
    audit = pd.concat([kept, excluded]).reindex(columns=AUDIT_COLUMNS).sort_values(["source", "image_id"])
    audit.to_csv(config.MANIFEST_DIR / "all_images.csv", index=False)
    # Compact record of the exact split, committed to Git so the paper's results can be reproduced
    audit[["source", "image_id", "patient_id", "stage", "split", "exclude_reason"]].to_csv(
        config.MANIFEST_DIR / "splits.csv", index=False
    )

    counts = pd.crosstab([kept["split"], kept["source"]], kept["stage"], margins=True, margins_name="total")
    counts.columns = [config.STAGE_NAMES[c] if isinstance(c, (int, np.integer)) else c for c in counts.columns]
    out_dir = config.EVAL_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "class_counts.csv")
    kept[QUALITY_COLUMNS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75]).to_csv(out_dir / "quality_stats.csv")

    print("\nImages per split, source and stage:")
    print(counts.to_string())
    print("\nExcluded:", excluded["exclude_reason"].value_counts().to_dict())


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", default=list(COLLECTORS), choices=COLLECTORS)
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--dup-distance", type=int, default=DUPLICATE_MAX_DISTANCE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    df = cache_all(collect(args.sources), args.workers)
    df = mark_duplicates(df, args.dup_distance)

    excluded = df[df["exclude_reason"] != ""]
    kept = df[df["exclude_reason"] == ""].copy()
    train_pool = kept["source"].isin(TRAINING_SOURCES)
    kept.loc[train_pool, "split"] = assign_splits(kept[train_pool], args.seed)
    kept.loc[~train_pool, "split"] = "external_test"

    kept = materialize(kept)
    write_outputs(kept, excluded)


if __name__ == "__main__":
    main()
