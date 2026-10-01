import numpy as np
import pytest

from adfd.eval import metrics


def test_epsilon_for_snr_matches_target_power():
    from adfd.eval.attack import epsilon_for_snr

    power = 0.25
    epsilon = epsilon_for_snr(power, 20.0)
    achieved = 10.0 * np.log10(power / (epsilon ** 2))
    assert abs(achieved - 20.0) < 1e-6


def test_attack_label_follows_spoof_index():
    from adfd.eval.attack import attack_label

    assert attack_label(1, 1) == 1 and attack_label(0, 1) == 0
    assert attack_label(1, 0) == 0 and attack_label(0, 0) == 1


def test_bootstrap_delta_eer_ci_brackets_zero_for_identical_scores():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=200)
    labels = rng.integers(0, 2, size=200)
    low, high = metrics.bootstrap_delta_eer_ci(scores, scores, labels, n_boot=300, seed=1)
    assert low <= 0.0 <= high


def test_snr_budget_reports_achieved_snr_and_prepared_consistency(tmp_path):
    pytest.importorskip("torch")
    from adfd.datasets.bootstrap import make_bootstrap
    from adfd.datasets.manifest import label_to_int, read_manifest, split_view
    from adfd.eval.attack import run_attack
    from adfd.eval.runner import fit_detector
    from adfd.eval.sweep import clean_scores_map
    from adfd.models.base import build_detector

    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=16, seed=5)
    manifest = read_manifest(out / "manifest.csv")
    detector = build_detector("torch_reference", epochs=6)
    detector.load()
    fit_detector(detector, manifest)
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    clean = clean_scores_map(detector, evaluation)
    report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets)
    strong = run_attack(detector, evaluation, "pgd", [30, 10], clean, report["threshold"], {"budget": "snr", "steps": 8}, budget="snr")
    at30 = strong[strong["level"] == 30]["strength_value"].mean()
    at10 = strong[strong["level"] == 10]["strength_value"].mean()
    assert at30 > at10
    assert set(strong["severity"].unique()) == {-30.0, -10.0}
    assert strong["delta_score"].abs().mean() > 0.0


def test_transfer_scoring_populates_column(tmp_path):
    pytest.importorskip("torch")
    from adfd.datasets.bootstrap import make_bootstrap
    from adfd.datasets.manifest import label_to_int, read_manifest, split_view
    from adfd.eval.attack import run_attack, transfer_aggregate
    from adfd.eval.runner import fit_detector
    from adfd.eval.sweep import clean_scores_map
    from adfd.models.base import build_detector

    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=16, seed=6)
    manifest = read_manifest(out / "manifest.csv")
    detector = build_detector("torch_reference", epochs=6)
    detector.load()
    fit_detector(detector, manifest)
    other = build_detector("reference")
    other.load()
    fit_detector(other, manifest)
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    clean = clean_scores_map(detector, evaluation)
    report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets)
    table = run_attack(detector, evaluation, "fgsm", [20], clean, report["threshold"], {"budget": "snr"}, budget="snr", transfer=[other])
    assert f"transfer_{other.name}" in table.columns
    aggregate = transfer_aggregate(table, [other.name], {other.name: report["eer"]})
    assert len(aggregate) == 1
