import numpy as np
import pytest

from adfd.eval import metrics


def test_eer_separable_is_zero():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    eer, _ = metrics.equal_error_rate(scores, labels)
    assert eer == 0.0


def test_auc_perfect():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert metrics.area_under_curve(scores, labels) == 1.0


def test_relative_degradation():
    assert metrics.relative_degradation(0.1, 0.2) == pytest.approx(1.0)


def test_attack_success_rate():
    labels = [1, 1, 0]
    clean = [1, 1, 0]
    attacked = [0, 0, 0]
    assert metrics.attack_success_rate(labels, clean, attacked) == pytest.approx(2 / 3)
