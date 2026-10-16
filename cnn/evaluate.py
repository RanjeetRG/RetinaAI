"""Evaluate trained classifiers (M3).

    python -m cnn.evaluate                                     # every models/cnn_<backbone>_<size>.keras
    python -m cnn.evaluate --models efficientnetb0_384 --splits test

Each model's input size is read from the model itself, and the matching arrays are used. For each model and split,
writes to outputs/eval/cnn/<backbone>_<size>/:
    <split>_metrics.json      metrics overall and per source (APTOS vs EyePACS exposes the camera confound)
    <split>_predictions.csv   per-image probabilities
and merges one row per model x split x subset into outputs/eval/cnn/summary.csv.

Referable-DR sensitivity is reported twice: at the argmax grade, and at a screening operating point, i.e. the
cut-off on P(stage >= 2) that reaches 90% sensitivity on the validation split, applied unchanged to test.
Runs on the CPU or GPU.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from cnn.data import array_path, load_split
from cnn.metrics import classification_report, referable_threshold, sensitivity_specificity_at
from cnn.model import BACKBONES, input_size, load_classifier
from cnn.train import BatchLoader

EVAL_SPLITS = ("test", "external_test")
SCREENING_SENSITIVITY = 0.90
SUMMARY_COLUMNS = [
    "n", "qwk", "accuracy", "macro_f1", "macro_recall", "referable_sensitivity", "referable_specificity",
    "referable_auc", "screening_sensitivity", "screening_specificity", "recall_severe", "recall_proliferative",
]  # fmt: skip
# Proposed targets from the PRD (to be confirmed for the paper)
TARGETS = {"qwk": 0.85, "macro_f1": 0.70, "screening_sensitivity": 0.90}


def model_path(name: str):
    return config.MODELS_DIR / f"cnn_{name}.keras"


def trained_models() -> list[str]:
    """Names like 'efficientnetb0_384' for every models/cnn_<backbone>_<size>.keras on disk."""
    names = [p.stem.removeprefix("cnn_") for p in config.MODELS_DIR.glob("cnn_*_*.keras")]
    return sorted(n for n in names if n.rsplit("_", 1)[0] in BACKBONES)


def summary_row(report: dict) -> dict:
    row = {k: report[k] for k in SUMMARY_COLUMNS if k in report}
    row["recall_severe"] = report["per_class"]["Severe"]["recall"]
    row["recall_proliferative"] = report["per_class"]["Proliferative"]["recall"]
    return row


def predict(model, split: str):
    data = load_split(split, input_size(model), mmap=True)
    return data, model.predict(BatchLoader(data, 64), verbose=0)


def report_with_operating_point(stages, probs, threshold: float) -> dict:
    report = classification_report(stages, probs)
    sens, spec = sensitivity_specificity_at(stages, probs, threshold)
    report.update(screening_threshold=threshold, screening_sensitivity=sens, screening_specificity=spec)
    return report


def evaluate(name: str, split: str) -> list[dict]:
    model = load_classifier(model_path(name))
    val, val_probs = predict(model, "validation")
    threshold = referable_threshold(val.stages, val_probs, SCREENING_SENSITIVITY)
    data, probs = predict(model, split)

    reports = {"all": report_with_operating_point(data.stages, probs, threshold)}
    for source in sorted(set(data.sources)):
        mask = data.sources == source
        reports[source] = report_with_operating_point(data.stages[mask], probs[mask], threshold)

    out_dir = config.EVAL_DIR / "cnn" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(reports, indent=2))
    preds = pd.DataFrame(probs, columns=[f"p{k}" for k in range(config.NUM_STAGES)])
    preds.insert(0, "pred", probs.argmax(axis=1))
    preds.insert(0, "stage", data.stages)
    preds.insert(0, "source", data.sources)
    preds.insert(0, "path", data.paths)
    preds.round(5).to_csv(out_dir / f"{split}_predictions.csv", index=False)

    size = input_size(model)
    return [{"model": name, "size": size, "split": split, "subset": subset, **summary_row(r)} for subset, r in reports.items()]


def merge_summary(rows: list[dict]) -> pd.DataFrame:
    path = config.EVAL_DIR / "cnn" / "summary.csv"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["model"], new["split"]))
        old = old[[k not in keys for k in zip(old["model"], old["split"])]]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["split", "subset", "model"]).round(4)
    new.to_csv(path, index=False)
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="e.g. efficientnetb0_384 (default: every trained model)")
    parser.add_argument("--splits", nargs="+", choices=EVAL_SPLITS)
    args = parser.parse_args()

    names = args.models or trained_models()
    if not names:
        raise SystemExit("No trained models in models/. Run `python -m cnn.train` first.")
    sizes = {int(n.rsplit("_", 1)[1]) for n in names}
    splits = args.splits or [s for s in EVAL_SPLITS if all(array_path(s, size).exists() for size in sizes)]

    rows = []
    for name in names:
        for split in splits:
            print(f"evaluating {name} on {split}", flush=True)
            rows += evaluate(name, split)
    summary = merge_summary(rows)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[summary["model"].isin(names)].to_string(index=False))
    for row in rows:
        if row["subset"] == "all":
            verdicts = [f"{k} {row[k]:.3f} {'meets' if row[k] >= t else 'BELOW'} {t}" for k, t in TARGETS.items()]
            print(f"{row['model']} / {row['split']}: " + "; ".join(verdicts))


if __name__ == "__main__":
    main()
