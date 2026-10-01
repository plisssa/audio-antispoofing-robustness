import numpy as np
import pandas as pd

from adfd.eval import combine


def make_table(model, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(80):
        file_id = f"f{i}"
        label = "bona_fide" if i % 2 == 0 else "spoof"
        target = 0 if label == "bona_fide" else 1
        clean_score = rng.uniform(0.0, 0.3) if target == 0 else rng.uniform(0.7, 1.0)
        clean_pred = int(clean_score >= 0.5)
        for distortion in ["noise", "codec"]:
            for severity in [5.0, 15.0]:
                score = float(np.clip(clean_score + rng.normal(0, 0.05 * severity / 5.0), 0, 1))
                prediction = int(score >= 0.5)
                error = "correct" if prediction == target else ("false_negative" if target == 1 else "false_positive")
                rows.append({
                    "file_id": file_id, "dataset": "d", "label": label, "model_name": model, "model_version": "v",
                    "clean_score": clean_score, "distorted_score": score, "prediction_clean": clean_pred,
                    "prediction_distorted": prediction, "error_type": error, "flipped": int(prediction != clean_pred),
                    "distortion_type": distortion, "family": "f", "distortion_strength": severity, "severity": -severity,
                    "duration_s": rng.uniform(1, 5), "rms": rng.uniform(0, 0.3), "silence_ratio": rng.uniform(0, 0.5),
                    "clipping_ratio": rng.uniform(0, 0.1), "spectral_centroid": rng.uniform(500, 4000),
                    "spectral_bandwidth": rng.uniform(500, 3000), "snr_estimate_db": rng.uniform(0, 40),
                })
    return pd.DataFrame(rows)


def test_combine_cross_model(tmp_path):
    p1 = tmp_path / "m1.csv"
    p2 = tmp_path / "m2.csv"
    make_table("m1", 0).to_csv(p1, index=False)
    make_table("m2", 1).to_csv(p2, index=False)
    results = combine.combine([str(p1), str(p2)], n_clusters=4)
    assert set(results["clean_eer_by_model"]["model_name"]) == {"m1", "m2"}
    assert len(results["cross_model_conditions"]) > 0
    pivot = results["model_distortion_pivot"]
    assert "m1" in pivot.columns and "m2" in pivot.columns
    assert set(results["inverse_by_model"]["model_name"]) == {"m1", "m2"}
    assert "m1" in results["model_cluster_pivot"].columns
