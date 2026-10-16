"""Checks on the built dataset (skipped until `python -m preprocessing.build_dataset` has run)."""

import pandas as pd
import pytest

import config

SPLITS = ("train", "validation", "test")
pytestmark = pytest.mark.skipif(
    not all((config.MANIFEST_DIR / f"{s}.csv").exists() for s in SPLITS), reason="dataset not built"
)


@pytest.fixture(scope="module")
def manifests() -> pd.DataFrame:
    return pd.concat([pd.read_csv(config.MANIFEST_DIR / f"{s}.csv") for s in SPLITS], ignore_index=True)


def test_no_patient_in_two_splits(manifests):
    splits_per_patient = manifests.groupby("patient_id")["split"].nunique()
    assert splits_per_patient.max() == 1, splits_per_patient[splits_per_patient > 1].index[:5].tolist()


def test_no_image_listed_twice(manifests):
    assert not manifests.duplicated(["source", "image_id"]).any()
    assert not manifests["path"].duplicated().any()


def test_every_split_has_every_stage(manifests):
    counts = pd.crosstab(manifests["split"], manifests["stage"])
    assert list(counts.columns) == list(range(config.NUM_STAGES))
    assert (counts > 0).all().all()


def test_split_fractions(manifests):
    shares = manifests["split"].value_counts(normalize=True)
    for split, target in config.SPLIT_FRACTIONS.items():
        assert abs(shares[split] - target) < 0.01


def test_files_exist_and_match_folder(manifests):
    for row in manifests.sample(min(500, len(manifests)), random_state=0).itertuples():
        path = config.ROOT / row.path
        assert path.exists(), row.path
        assert path.parent.name == str(row.stage) and path.parent.parent.name == row.split


def test_excluded_images_are_not_in_splits(manifests):
    audit = pd.read_csv(config.MANIFEST_DIR / "all_images.csv")
    excluded = audit[audit["exclude_reason"].notna()]
    keys = set(zip(manifests["source"], manifests["image_id"]))
    assert not any(k in keys for k in zip(excluded["source"], excluded["image_id"]))
