import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "duration_s",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "spectral_centroid",
    "spectral_bandwidth",
    "snr_estimate_db",
    "severity",
]


def cluster_errors(error_table, n_clusters=6, feature_columns=FEATURE_COLUMNS, seed=1337):
    data = error_table.dropna(subset=feature_columns).copy()
    matrix = StandardScaler().fit_transform(data[feature_columns].to_numpy())
    clusters = min(n_clusters, len(data))
    model = KMeans(n_clusters=clusters, n_init=10, random_state=seed)
    data["cluster"] = model.fit_predict(matrix)
    return data


def cluster_profile(clustered):
    rows = []
    for cluster, group in clustered.groupby("cluster"):
        rows.append({
            "cluster": int(cluster),
            "n": int(len(group)),
            "error_rate": float((group["error_type"] != "correct").mean()),
            "false_negative_rate": float((group["error_type"] == "false_negative").mean()),
            "false_positive_rate": float((group["error_type"] == "false_positive").mean()),
            "top_distortion": group["distortion_type"].mode().iloc[0],
            "mean_severity": float(group["severity"].mean()),
            "mean_snr_estimate_db": float(group["snr_estimate_db"].mean()),
            "mean_duration_s": float(group["duration_s"].mean()),
            "mean_silence_ratio": float(group["silence_ratio"].mean()),
            "mean_spectral_centroid": float(group["spectral_centroid"].mean()),
        })
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)
