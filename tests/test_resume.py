import pandas as pd

from adfd.datasets.bootstrap import make_bootstrap
from adfd.datasets.manifest import read_manifest, split_view
from adfd.eval.runner import fit_detector
from adfd.eval.sweep import clean_scores_map, run_sweep
from adfd.models.base import build_detector

DISTORTIONS = {
    "noise": {
        "kind": "noise",
        "param": "snr_db",
        "family": "natural_distortions",
        "levels": [20, 5],
        "options": {"color": "white"},
    }
}


def test_sweep_resumes_without_duplication(tmp_path):
    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=8, seed=2)
    manifest = read_manifest(out / "manifest.csv")
    detector = build_detector("reference")
    detector.load()
    fit_detector(detector, manifest)
    evaluation = split_view(manifest, "eval")
    clean = clean_scores_map(detector, evaluation)
    partial = str(tmp_path / "sweep.partial.csv")

    run_sweep(detector, evaluation.iloc[:2], DISTORTIONS, clean, 0.5, partial_path=partial)
    interrupted = pd.read_csv(partial)
    full = run_sweep(detector, evaluation, DISTORTIONS, clean, 0.5, partial_path=partial)

    assert len(interrupted) < len(full)
    assert full["file_id"].nunique() == len(evaluation)
    assert len(full) == len(evaluation) * 2
    assert full.duplicated(subset=["file_id", "distortion_type", "level"]).sum() == 0
