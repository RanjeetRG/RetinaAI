"""Download the raw datasets into dataset/raw/ (M2).

    python -m scripts.download_data              # everything
    python -m scripts.download_data --only aptos

Needs a Kaggle API token in ~/.kaggle/access_token (or a legacy ~/.kaggle/kaggle.json), and the APTOS 2019
competition rules accepted on its Kaggle page. Messidor-2 images are not on Kaggle under a clear license: request them from ADCIS
(https://www.adcis.net/en/third-party/messidor2/) and unzip them into dataset/raw/messidor2/images/.
Only the adjudicated grades are downloaded here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import config

APTOS_COMPETITION = "aptos2019-blindness-detection"
EYEPACS_DATASET = "tanlikesmath/diabetic-retinopathy-resized"  # 35,126 EyePACS train images, max 1024 px
MESSIDOR_GRADES_DATASET = "google-brain/messidor2-dr-grades"  # Krause et al. 2018 adjudicated ICDR grades

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def kaggle_cli() -> list[str]:
    # `python -m kaggle` rather than the kaggle.exe launcher: on Windows the launcher runs a child process
    # that would survive a stall-restart and keep the half-written file open
    if importlib.util.find_spec("kaggle") is None:
        sys.exit("The kaggle package is not installed in this environment (pip install kaggle).")
    return [sys.executable, "-m", "kaggle"]


def check_credentials() -> None:
    """Accept a new-style API token (env var or ~/.kaggle/access_token) or a legacy kaggle.json."""
    kaggle_dir = Path.home() / ".kaggle"
    if os.environ.get("KAGGLE_API_TOKEN") or any(
        (kaggle_dir / name).exists() for name in ("access_token", "kaggle.json")
    ):
        return
    sys.exit(
        f"No Kaggle credentials found.\n"
        "Generate a token at kaggle.com -> Settings -> API Tokens and save it, as plain text with no extension,\n"
        f"to {kaggle_dir / 'access_token'}"
    )


def run(args: list[str], watch: Path | None = None, stall_seconds: int = 120, attempts: int = 30) -> None:
    """Run a kaggle command, restarting it if `watch` stops growing (the CLI hangs on dead connections).

    Kaggle resumes partial downloads, so a restart continues where the last attempt stopped.
    """
    for attempt in range(1, attempts + 1):
        print(f"$ kaggle {' '.join(args)}  (attempt {attempt})", flush=True)
        proc = subprocess.Popen([*kaggle_cli(), *args])
        last_size, last_change = -1, time.monotonic()
        while proc.poll() is None:
            time.sleep(10)
            size = watch.stat().st_size if watch is not None and watch.exists() else 0
            if size != last_size:
                last_size, last_change = size, time.monotonic()
            elif watch is not None and time.monotonic() - last_change > stall_seconds:
                print(f"No progress for {stall_seconds} s; restarting the download", flush=True)
                proc.kill()
                proc.wait()
                break
        else:
            if proc.returncode == 0:
                return
            print(f"kaggle exited with code {proc.returncode}; retrying", flush=True)
        time.sleep(5)
    sys.exit(f"Download failed after {attempts} attempts: kaggle {' '.join(args)}")


def extract(zip_path: Path, dest: Path, members_prefix: tuple[str, ...] | None = None) -> None:
    """Unzip, optionally only members under the given prefixes, then delete the zip to save disk."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if members_prefix:
            wanted = [n for n in names if n.startswith(members_prefix)]
            if not wanted:
                print(f"Warning: nothing in {zip_path.name} matches {members_prefix}; extracting everything")
            names = wanted or names
        print(f"Extracting {len(names)} files from {zip_path.name}")
        zf.extractall(dest, members=names)
    zip_path.unlink()


def done_marker(dest: Path) -> Path:
    return dest / ".download_complete"


def download_aptos() -> None:
    dest = config.RAW_DIR / "aptos"
    if done_marker(dest).exists():
        print("APTOS already downloaded")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{APTOS_COMPETITION}.zip"
    run(["competitions", "download", "-c", APTOS_COMPETITION, "-p", str(dest)], watch=zip_path)
    # Only the labelled training set is needed; the competition test set has no public labels
    extract(zip_path, dest, members_prefix=("train.csv", "train_images/"))
    done_marker(dest).touch()


def download_eyepacs() -> None:
    dest = config.RAW_DIR / "eyepacs"
    if done_marker(dest).exists():
        print("EyePACS already downloaded")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{EYEPACS_DATASET.split('/')[1]}.zip"
    run(["datasets", "download", "-d", EYEPACS_DATASET, "-p", str(dest)], watch=zip_path)
    # The dataset also ships a pre-cropped copy; our own crop is used instead, so skip it
    extract(zip_path, dest, members_prefix=("resized_train/", "trainLabels.csv"))
    done_marker(dest).touch()


def download_messidor() -> None:
    dest = config.RAW_DIR / "messidor2"
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "messidor_data.csv").exists():
        run(["datasets", "download", "-d", MESSIDOR_GRADES_DATASET, "-p", str(dest)])
        extract(dest / f"{MESSIDOR_GRADES_DATASET.split('/')[1]}.zip", dest)
    images = dest / "images"
    n = sum(1 for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if images.exists() else 0
    if n == 0:
        print(
            "Messidor-2 grades are ready, but the images are missing.\n"
            "Request them from ADCIS (https://www.adcis.net/en/third-party/messidor2/) and unzip them into\n"
            f"  {images}"
        )
    else:
        print(f"Messidor-2: {n} images found")


SOURCES = {"aptos": download_aptos, "eyepacs": download_eyepacs, "messidor": download_messidor}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=SOURCES, action="append", help="download only this source")
    args = parser.parse_args()
    check_credentials()
    for name in args.only or SOURCES:
        SOURCES[name]()


if __name__ == "__main__":
    main()
