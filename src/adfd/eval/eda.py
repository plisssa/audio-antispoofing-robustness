import pandas as pd

FEATURES = [
    "duration_s",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "spectral_centroid",
    "spectral_bandwidth",
    "snr_estimate_db",
    "severity",
]


def feature_summary(error_table):
    table = error_table.copy()
    table["is_error"] = table["error_type"] != "correct"
    rows = []
    for model, group in table.groupby("model_name"):
        errors = group[group["is_error"]]
        correct = group[~group["is_error"]]
        for feature in FEATURES:
            if feature not in group.columns:
                continue
            spread = float(group[feature].std()) + 1e-9
            mean_error = float(errors[feature].mean())
            mean_correct = float(correct[feature].mean())
            rows.append({
                "model_name": model,
                "feature": feature,
                "mean_error": mean_error,
                "mean_correct": mean_correct,
                "std_diff": (mean_error - mean_correct) / spread,
                "n_error": int(len(errors)),
                "n_correct": int(len(correct)),
            })
    frame = pd.DataFrame(rows)
    frame["abs_std_diff"] = frame["std_diff"].abs()
    return (
        frame.sort_values(["model_name", "abs_std_diff"], ascending=[True, False])
        .drop(columns="abs_std_diff")
        .reset_index(drop=True)
    )
