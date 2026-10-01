import pandas as pd

from adfd.datasets.bootstrap import make_bootstrap
from adfd.datasets.manifest import label_to_int, read_manifest, split_view
from adfd.eval import compare, metrics
from adfd.eval.error_table import build_error_table, file_features
from adfd.eval.clustering import cluster_errors
from adfd.eval.runner import fit_detector
from adfd.eval.sweep import aggregate_sweep, clean_scores_map, run_sweep
from adfd.models.base import available_detectors, build_detector

DISTORTIONS = {
    "noise": {
        "kind": "noise",
        "param": "snr_db",
        "family": "natural_distortions",
        "levels": [20, 5],
        "options": {"color": "white"},
    }
}


def test_registry_lists_multiple_models():
    names = set(available_detectors())
    assert {
        "reference",
        "melspec",
        "aasist",
        "rawnet2",
        "aasist3",
        "ssl_aasist",
        "xlsr_mamba",
        "nes2net",
    }.issubset(names)


def test_multimodel_comparison(tmp_path):
    out = tmp_path / "bootstrap"
    make_bootstrap(out, n_per_class=12, seed=5)
    manifest = read_manifest(out / "manifest.csv")
    evaluation = split_view(manifest, "eval")
    targets = evaluation["label"].map(label_to_int).to_numpy()
    features = file_features(evaluation)
    aggregates, error_tables = [], []
    for name in ["reference", "melspec"]:
        detector = build_detector(name)
        detector.load()
        fit_detector(detector, manifest)
        clean = clean_scores_map(detector, evaluation)
        clean_report = metrics.summary(evaluation["file_id"].map(clean).to_numpy(), targets)
        sweep = run_sweep(detector, evaluation, DISTORTIONS, clean, clean_report["threshold"])
        sweep["model_name"] = detector.name
        aggregate = aggregate_sweep(sweep, clean_report["eer"])
        aggregate["model_name"] = detector.name
        aggregates.append(aggregate)
        error_tables.append(build_error_table(sweep, features, detector.metadata()))
    aggregate_all = pd.concat(aggregates, ignore_index=True)
    error_all = pd.concat(error_tables, ignore_index=True)
    summary = compare.model_distortion_summary(aggregate_all)
    assert set(summary["model_name"]) == {"reference_lfcc_logreg", "melspec_logreg"}
    clustered = cluster_errors(error_all, n_clusters=4)
    table, pivot = compare.model_cluster_error(clustered)
    assert "reference_lfcc_logreg" in pivot.columns
    assert "melspec_logreg" in pivot.columns
