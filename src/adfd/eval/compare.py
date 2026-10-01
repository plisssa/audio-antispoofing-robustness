import pandas as pd


def model_distortion_summary(aggregate):
    summary = (
        aggregate.groupby(["model_name", "distortion_type"])
        .agg(
            mean_eer=("eer", "mean"),
            mean_delta_eer=("delta_eer", "mean"),
            max_delta_eer=("delta_eer", "max"),
            mean_flip_rate=("flip_rate", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["distortion_type", "mean_delta_eer"], ascending=[True, False]).reset_index(drop=True)


def model_distortion_pivot(aggregate, value="delta_eer"):
    reduced = aggregate.groupby(["distortion_type", "model_name"])[value].max().reset_index()
    return reduced.pivot(index="distortion_type", columns="model_name", values=value).reset_index()


def model_cluster_error(clustered):
    rows = []
    for (cluster, model), group in clustered.groupby(["cluster", "model_name"]):
        rows.append({
            "cluster": int(cluster),
            "model_name": model,
            "n": int(len(group)),
            "error_rate": float((group["error_type"] != "correct").mean()),
        })
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="cluster", columns="model_name", values="error_rate").reset_index()
    return table, pivot
