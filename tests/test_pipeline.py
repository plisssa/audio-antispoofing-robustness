from adfd.datasets.bootstrap import make_bootstrap
from adfd.datasets.manifest import read_manifest, split_view
from adfd.eval.runner import fit_detector
from adfd.eval.sweep import aggregate_sweep, clean_scores_map, run_sweep
from adfd.models.base import build_detector


def test_pipeline_smoke(tmp_path):
    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=12, seed=7)
    manifest = read_manifest(out / "manifest.csv")
    detector = build_detector("reference")
    detector.load()
    fit_detector(detector, manifest)
    evaluation = split_view(manifest, "eval")
    clean = clean_scores_map(detector, evaluation)
    distortions = {
        "noise": {
            "kind": "noise",
            "param": "snr_db",
            "family": "natural_distortions",
            "levels": [20, 5],
            "options": {"color": "white"},
        }
    }
    sweep = run_sweep(detector, evaluation, distortions, clean, 0.5)
    assert {"distortion_type", "severity", "distorted_score", "error_type"}.issubset(sweep.columns)
    aggregate = aggregate_sweep(sweep, 0.0)
    assert len(aggregate) == 2
