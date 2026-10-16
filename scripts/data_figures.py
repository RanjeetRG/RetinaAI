"""Figures describing the built dataset (M2), saved to outputs/eval/data/.

    python -m scripts.data_figures

- stage_samples.jpg: random training images per stage, as the CNN sees them (preprocessed, 224 px)
- preprocessing_steps.jpg: raw -> cached 512 px -> CNN input -> GAN input, one image per stage
"""

from __future__ import annotations

import pandas as pd

import config
from preprocessing import fundus as F
from utils.figures import header, row_of, save_jpeg, stack

OUT_DIR = config.EVAL_DIR / "data"


def raw_path(row) -> str:
    audit = pd.read_csv(config.MANIFEST_DIR / "all_images.csv", usecols=["source", "image_id", "raw_path"])
    match = audit[(audit["source"] == row.source) & (audit["image_id"] == row.image_id)]
    return match["raw_path"].iloc[0]


def main(per_stage: int = 6, seed: int = config.SEED) -> None:
    train = pd.read_csv(config.MANIFEST_DIR / "train.csv")
    train = train[train["gan_ok"]]  # show clean examples

    rows = []
    for stage, name in enumerate(config.STAGE_NAMES):
        sample = train[train["stage"] == stage].sample(per_stage, random_state=seed)
        imgs = [F.prepare_cnn(F.read_image(config.ROOT / p)) for p in sample["path"]]
        rows.append(row_of(imgs, f"{stage}: {name}"))
    save_jpeg(OUT_DIR / "stage_samples.jpg", stack(rows))

    rows = []
    for stage in range(config.NUM_STAGES):
        r = train[(train["stage"] == stage) & (train["source"] == "aptos")].sample(1, random_state=seed).iloc[0]
        cached = F.read_image(config.ROOT / r.path)
        raw = F.pad_to_square(F.read_image(raw_path(r)))
        steps = [raw, cached, F.prepare_cnn(cached), F.from_gan_range(F.prepare_gan(cached))]
        rows.append(row_of(steps, f"Stage {stage}"))
    body = stack(rows)
    titles = ["Raw (padded)", f"Cached {config.CACHE_SIZE} px", f"CNN input {config.CNN_SIZE} px", f"GAN input {config.GAN_SIZE} px"]
    save_jpeg(OUT_DIR / "preprocessing_steps.jpg", stack([header(titles, body.shape[1]), body]))


if __name__ == "__main__":
    main()
