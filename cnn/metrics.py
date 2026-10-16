"""Classification metrics for the paper (PRD "Success metrics and evaluation")."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import config

LABELS = list(range(config.NUM_STAGES))


def quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic weighted kappa: the standard DR grading metric, penalizing far-off grades more."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS))


def referable_score(probs: np.ndarray) -> np.ndarray:
    """Probability of referable DR: P(stage >= 2)."""
    return probs[:, config.REFERABLE_STAGE :].sum(axis=1)


def referable_threshold(y_true: np.ndarray, probs: np.ndarray, target_sensitivity: float = 0.90) -> float:
    """Highest cut-off on the referable score that still reaches `target_sensitivity` on this data.

    Choose it on the validation split and apply it to test: the standard way DR screening papers fix an
    operating point without looking at the test set.
    """
    positives = np.sort(referable_score(probs)[np.asarray(y_true) >= config.REFERABLE_STAGE])
    # Keeping the top `needed` positive scores at or above the cut-off gives the target sensitivity
    needed = int(np.ceil(target_sensitivity * len(positives)))
    return float(positives[len(positives) - needed])


def sensitivity_specificity_at(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[float, float]:
    positive = np.asarray(y_true) >= config.REFERABLE_STAGE
    flagged = referable_score(probs) >= threshold
    return float((flagged & positive).sum() / positive.sum()), float((~flagged & ~positive).sum() / (~positive).sum())


def _auc(y_true_binary: np.ndarray, score: np.ndarray) -> float:
    # AUC is undefined when only one class is present (e.g. a small per-source subset)
    return float(roc_auc_score(y_true_binary, score)) if 0 < y_true_binary.sum() < len(y_true_binary) else float("nan")


def classification_report(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """All grading metrics from true stages and predicted class probabilities (N, NUM_STAGES)."""
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)

    referable_true = y_true >= config.REFERABLE_STAGE
    referable_pred = y_pred >= config.REFERABLE_STAGE
    positives, negatives = referable_true.sum(), (~referable_true).sum()

    return {
        "n": int(len(y_true)),
        "qwk": quadratic_kappa(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "referable_sensitivity": float((referable_pred & referable_true).sum() / positives) if positives else float("nan"),
        "referable_specificity": float((~referable_pred & ~referable_true).sum() / negatives) if negatives else float("nan"),
        "referable_auc": _auc(referable_true.astype(int), referable_score(probs)),
        "per_class": {
            config.STAGE_NAMES[k]: {
                "support": int(support[k]),
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "auc_ovr": _auc((y_true == k).astype(int), probs[:, k]),
            }
            for k in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),  # rows true, cols predicted
    }
