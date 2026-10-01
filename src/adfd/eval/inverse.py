import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

FEATURE_COLUMNS = [
    "severity",
    "distortion_strength",
    "duration_s",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "spectral_centroid",
    "spectral_bandwidth",
    "snr_estimate_db",
]


def build_dataset(error_table):
    table = error_table.copy()
    table["error"] = (table["error_type"] != "correct").astype(int)
    numeric = table[[column for column in FEATURE_COLUMNS if column in table.columns]].copy()
    distortion = pd.get_dummies(table["distortion_type"], prefix="dist")
    matrix = pd.concat([numeric, distortion], axis=1).fillna(0.0)
    return matrix, table["error"].to_numpy(), list(matrix.columns)


def fit_inverse(matrix, target, seed=1337):
    model = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    classes = np.unique(target)
    if len(classes) < 2:
        model.fit(matrix, target)
        return model, float("nan")
    folds = min(5, int(np.bincount(target).min()))
    if folds < 2:
        model.fit(matrix, target)
        return model, float("nan")
    auc = float(np.mean(cross_val_score(model, matrix, target, cv=folds, scoring="roc_auc")))
    model.fit(matrix, target)
    return model, auc


def feature_importance(model, feature_names):
    frame = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def failure_by_distortion(error_table):
    table = error_table.copy()
    table["error"] = (table["error_type"] != "correct").astype(int)
    grouped = table.groupby(["model_name", "distortion_type"]).agg(
        error_rate=("error", "mean"),
        n=("error", "size"),
    ).reset_index()
    return grouped.sort_values(["model_name", "error_rate"], ascending=[True, False]).reset_index(drop=True)
