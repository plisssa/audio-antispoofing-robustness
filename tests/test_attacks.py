import numpy as np
import pytest

from adfd.eval import metrics


def test_cllr_separable_small():
    scores = np.array([0.01, 0.02, 0.98, 0.99])
    labels = np.array([0, 0, 1, 1])
    assert metrics.cllr(scores, labels) < 0.1
    assert metrics.min_cllr(scores, labels) <= metrics.cllr(scores, labels) + 1e-6


def test_cllr_random_is_uninformative():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=400)
    labels = rng.integers(0, 2, size=400)
    assert metrics.min_cllr(scores, labels) <= metrics.cllr(scores, labels) + 1e-6
    assert 0.9 < metrics.min_cllr(scores, labels) < 1.1


def test_pgd_perturbs_scores(tmp_path):
    pytest.importorskip("torch")
    from adfd.datasets.bootstrap import make_bootstrap
    from adfd.datasets.manifest import label_to_int, read_manifest, split_view
    from adfd.eval.attack import run_attack
    from adfd.eval.runner import fit_detector
    from adfd.eval.sweep import clean_scores_map
    from adfd.models.base import build_detector

    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=16, seed=3)
    manifest = read_manifest(out / "manifest.csv")
    detector = build_detector("torch_reference", epochs=6)
    detector.load()
    fit_detector(detector, manifest)
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    clean = clean_scores_map(detector, evaluation)
    clean_report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets)
    table = run_attack(detector, evaluation, "pgd", [0.005], clean, clean_report["threshold"], {"steps": 10, "alpha_ratio": 0.3})
    success = metrics.attack_success_rate(table["target"], table["prediction_clean"], table["prediction_distorted"])
    assert 0.0 <= success <= 1.0
    assert table["delta_score"].abs().mean() > 0.0
