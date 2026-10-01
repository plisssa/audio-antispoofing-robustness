import numpy as np
import pandas as pd

from adfd.eval import eda


def test_feature_summary_finds_driver():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        duration = rng.uniform(0.5, 5.0)
        is_error = duration < 2.0
        rows.append({
            "model_name": "m",
            "error_type": "false_negative" if is_error else "correct",
            "duration_s": duration,
            "rms": rng.uniform(0.0, 0.3),
            "silence_ratio": rng.uniform(0.0, 0.5),
            "clipping_ratio": rng.uniform(0.0, 0.1),
            "spectral_centroid": rng.uniform(500.0, 4000.0),
            "spectral_bandwidth": rng.uniform(500.0, 3000.0),
            "snr_estimate_db": rng.uniform(0.0, 40.0),
            "severity": rng.uniform(0.0, 1.0),
        })
    summary = eda.feature_summary(pd.DataFrame(rows))
    assert summary.iloc[0]["feature"] == "duration_s"
    assert abs(summary.iloc[0]["std_diff"]) > abs(summary.iloc[-1]["std_diff"])
