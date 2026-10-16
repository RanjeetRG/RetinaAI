import numpy as np
import pytest

from cnn.metrics import classification_report, quadratic_kappa, referable_threshold, sensitivity_specificity_at


def one_hot(stages, n=5):
    return np.eye(n)[stages]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0, 2])
    report = classification_report(y, one_hot(y))
    assert report["qwk"] == pytest.approx(1.0)
    assert report["accuracy"] == 1.0 and report["macro_f1"] == 1.0
    assert report["referable_sensitivity"] == 1.0 and report["referable_specificity"] == 1.0
    assert np.trace(report["confusion_matrix"]) == len(y)


def test_kappa_penalizes_far_errors_more():
    y = np.array([0, 0, 1, 2, 3, 4] * 5)
    near = np.clip(y + 1, 0, 4)
    far = np.where(y < 2, 4, 0)
    assert quadratic_kappa(y, near) > quadratic_kappa(y, far)


def test_referable_metrics():
    y = np.array([0, 1, 2, 3, 4])
    pred = np.array([0, 2, 1, 3, 4])  # one false positive (1 -> 2), one missed referable case (2 -> 1)
    report = classification_report(y, one_hot(pred))
    assert report["referable_sensitivity"] == pytest.approx(2 / 3)
    assert report["referable_specificity"] == pytest.approx(1 / 2)


def test_auc_is_nan_when_a_class_is_absent():
    y = np.array([0, 0, 1, 1])
    report = classification_report(y, one_hot(y))
    assert np.isnan(report["per_class"]["Severe"]["auc_ovr"])
    assert np.isnan(report["referable_auc"])


def test_referable_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    # Noisy referable scores: positives tend to score higher
    referable = (y >= 2).astype(float)
    score = np.clip(0.5 * referable + rng.normal(0.25, 0.2, len(y)), 0.001, 0.999)
    probs = np.zeros((len(y), 5))
    probs[:, 2] = score
    probs[:, 0] = 1 - score

    t = referable_threshold(y, probs, target_sensitivity=0.90)
    sens, spec = sensitivity_specificity_at(y, probs, t)
    assert 0.90 <= sens < 0.91
    assert 0 < spec < 1
    # A stricter target needs a lower cut-off
    assert referable_threshold(y, probs, target_sensitivity=0.97) < t
