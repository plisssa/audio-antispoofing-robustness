import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def monotonicity(x, y):
    if np.std(y) < 1e-12:
        return 0.0
    correlation = spearmanr(x, y).correlation
    return float(correlation) if correlation == correlation else 0.0


def linear_interp_expected(x, y):
    expected = y.astype(float).copy()
    for index in range(1, len(y) - 1):
        left, middle, right = x[index - 1], x[index], x[index + 1]
        span = right - left
        if span == 0:
            expected[index] = y[index]
        else:
            expected[index] = y[index - 1] + (y[index + 1] - y[index - 1]) * ((middle - left) / span)
    return expected


def non_monotonic(y, tolerance=0.05):
    steps = np.diff(y)
    signs = np.sign(steps[np.abs(steps) > tolerance])
    return len(set(signs.tolist())) > 1


def curve_metrics(severity, scores):
    order = np.argsort(severity)
    x = np.asarray(severity, dtype=float)[order]
    y = np.asarray(scores, dtype=float)[order]
    expected = linear_interp_expected(x, y)
    residual = np.abs(y - expected)
    steps = np.abs(np.diff(y))
    return {
        "monotonicity": monotonicity(x, y),
        "max_abs_step": float(steps.max()) if len(steps) else 0.0,
        "max_interp_residual": float(residual.max()) if len(residual) else 0.0,
        "mean_interp_residual": float(residual.mean()) if len(residual) else 0.0,
        "non_monotonic": int(non_monotonic(y)),
    }


def per_file_response(sweep):
    rows = []
    for (model, file_id, distortion), group in sweep.groupby(["model_name", "file_id", "distortion_type"]):
        metrics = curve_metrics(group["severity"].to_numpy(), group["distorted_score"].to_numpy())
        metrics.update({
            "model_name": model,
            "file_id": file_id,
            "distortion_type": distortion,
            "label": group["label"].iloc[0],
        })
        rows.append(metrics)
    return pd.DataFrame(rows)


def per_distortion_response(per_file):
    return (
        per_file.groupby(["model_name", "distortion_type"])
        .agg(
            monotonicity=("monotonicity", "mean"),
            max_abs_step=("max_abs_step", "mean"),
            mean_interp_residual=("mean_interp_residual", "mean"),
            non_monotonic_rate=("non_monotonic", "mean"),
        )
        .reset_index()
        .sort_values("non_monotonic_rate", ascending=False)
        .reset_index(drop=True)
    )
