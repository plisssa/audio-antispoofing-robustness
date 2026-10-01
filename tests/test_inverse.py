import numpy as np
import pandas as pd

from adfd.eval import inverse


def make_error_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    severity = rng.uniform(0.0, 1.0, n)
    error = (severity + 0.1 * rng.standard_normal(n) > 0.6).astype(int)
    return pd.DataFrame({
        "error_type": np.where(error == 1, "false_negative", "correct"),
        "model_name": "m",
        "distortion_type": rng.choice(["noise", "codec"], n),
        "severity": severity,
        "distortion_strength": severity * 10.0,
        "duration_s": rng.uniform(1.0, 5.0, n),
        "rms": rng.uniform(0.0, 0.3, n),
        "silence_ratio": rng.uniform(0.0, 0.5, n),
        "clipping_ratio": rng.uniform(0.0, 0.1, n),
        "spectral_centroid": rng.uniform(500.0, 4000.0, n),
        "spectral_bandwidth": rng.uniform(500.0, 3000.0, n),
        "snr_estimate_db": rng.uniform(0.0, 40.0, n),
    })


def test_inverse_predicts_error_from_params():
    table = make_error_table()
    matrix, target, names = inverse.build_dataset(table)
    model, auc = inverse.fit_inverse(matrix, target, seed=1)
    importance = inverse.feature_importance(model, names)
    assert auc > 0.7
    assert len(importance) == len(names)
    assert importance.iloc[0]["feature"] in {"severity", "distortion_strength"}


def test_failure_by_distortion_shape():
    table = make_error_table()
    summary = inverse.failure_by_distortion(table)
    assert set(summary["distortion_type"]) == {"noise", "codec"}
    assert (summary["error_rate"] >= 0).all() and (summary["error_rate"] <= 1).all()
