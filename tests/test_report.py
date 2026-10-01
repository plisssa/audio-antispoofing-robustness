import json

import pandas as pd

from adfd.eval import report


def _make_sweep_dir(path, model, dataset):
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps({"dataset": dataset, "detectors": [{"name": model}]}))
    (path / "clean_metrics.json").write_text(json.dumps({model: {"eer": 0.1, "auc": 0.9, "cllr": 0.3, "min_dcf": 0.2, "n": 100}}))
    pd.DataFrame([
        {"distortion_type": "noise", "level": 0.0, "severity": 0.0, "model_name": model, "eer": 0.4, "delta_eer": 0.3, "delta_eer_ci_low": 0.2, "delta_eer_ci_high": 0.4},
        {"distortion_type": "noise", "level": 20.0, "severity": -20.0, "model_name": model, "eer": 0.15, "delta_eer": 0.05, "delta_eer_ci_low": 0.0, "delta_eer_ci_high": 0.1},
    ]).to_csv(path / "aggregate.csv", index=False)


def _make_attack_dir(path, model, dataset):
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps({"dataset": dataset, "detector": {"name": model}, "budget": "snr"}))
    pd.DataFrame([
        {"distortion_type": "pgd", "level": 30.0, "severity": -30.0, "model_name": model, "eer": 0.6, "delta_eer": 0.5, "attack_success_rate": 0.7},
        {"distortion_type": "pgd", "level": 10.0, "severity": -10.0, "model_name": model, "eer": 0.9, "delta_eer": 0.8, "attack_success_rate": 0.95},
    ]).to_csv(path / "attack_aggregate.csv", index=False)


def test_build_report_consolidates_and_shifts(tmp_path):
    _make_sweep_dir(tmp_path / "aasist_asv19", "aasist", "asvspoof2019_la")
    _make_sweep_dir(tmp_path / "aasist_itw", "aasist", "in_the_wild")
    _make_attack_dir(tmp_path / "aasist_attack", "aasist", "asvspoof2019_la")

    results = report.build_report(
        [str(tmp_path / "aasist_asv19"), str(tmp_path / "aasist_itw")],
        [str(tmp_path / "aasist_attack")],
        tmp_path / "report",
    )
    assert len(results["clean"]) == 2
    shift = results["dataset_shift"]
    assert "asvspoof2019_la" in shift.columns and "in_the_wild" in shift.columns
    assert not results["worst"].empty
    assert set(results["attacks"]["attack_success_rate"]) == {0.7, 0.95}
    assert (tmp_path / "report" / "report.md").exists()
    assert (tmp_path / "report" / "dataset_shift.csv").exists()


def test_build_report_survives_missing_dirs(tmp_path):
    results = report.build_report([str(tmp_path / "nope")], [str(tmp_path / "gone")], tmp_path / "out")
    assert results["clean"].empty
    assert (tmp_path / "out" / "clean_summary.csv").exists()
