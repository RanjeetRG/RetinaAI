import numpy as np
import pandas as pd

from preprocessing import build_dataset as B
from preprocessing.fundus import load_fundus, read_image
from tests.conftest import encode, make_fundus


def _hash(img):
    return B.content_hash(load_fundus(img))


def test_reencoded_copy_is_a_duplicate_but_other_images_are_not():
    a, b = make_fundus(1), make_fundus(2)
    a_jpeg = read_image(encode(a, ".jpg"))  # same photo, recompressed
    groups, nearest = B.duplicate_groups([_hash(a), _hash(a_jpeg), _hash(b)], B.DUPLICATE_MAX_DISTANCE)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_conflicting_duplicates_are_dropped_and_consistent_ones_kept_once():
    a, b = make_fundus(1), make_fundus(2)
    ha, hb = _hash(a), _hash(b)
    df = pd.DataFrame(
        {
            "source": ["aptos"] * 5,
            "image_id": list("abcde"),
            "stage": [2, 2, 0, 3, 1],
            "phash": [ha, ha, hb, hb, _hash(make_fundus(3))],
            "error": np.nan,
            "unusable": False,
            "laplacian_var": [10.0, 20.0, 5.0, 5.0, 5.0],
        }
    )
    out = B.mark_duplicates(df, B.DUPLICATE_MAX_DISTANCE)
    reasons = dict(zip(out["image_id"], out["exclude_reason"]))
    assert reasons["a"] == "duplicate" and reasons["b"] == ""  # sharper copy kept
    assert reasons["c"] == reasons["d"] == "duplicate_conflicting_labels"
    assert reasons["e"] == ""


def test_split_is_patient_level_and_stratified():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(600):
        stage = int(rng.choice(5, p=[0.5, 0.1, 0.2, 0.1, 0.1]))
        for eye in ("left", "right"):
            rows.append({"patient_id": f"eyepacs_{p}", "source": "eyepacs", "stage": stage, "image_id": f"{p}_{eye}"})
    df = pd.DataFrame(rows)
    df["split"] = B.assign_splits(df, seed=42)

    assert df.groupby("patient_id")["split"].nunique().max() == 1
    shares = df["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.02
    assert abs(shares["validation"] - 0.15) < 0.02
    # Every split contains every stage
    assert (pd.crosstab(df["split"], df["stage"]) > 0).all().all()


def test_split_is_reproducible():
    df = pd.DataFrame(
        {"patient_id": [f"aptos_{i}" for i in range(200)], "source": "aptos", "stage": np.arange(200) % 5}
    )
    assert B.assign_splits(df, 7).equals(B.assign_splits(df, 7))
