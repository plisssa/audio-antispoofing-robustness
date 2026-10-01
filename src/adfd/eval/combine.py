import pandas as pd

from ..datasets.manifest import label_to_int
from . import clustering, compare, inverse
from .metrics import area_under_curve, equal_error_rate

KEY = ["model_name", "dataset", "file_id", "distortion_type", "severity"]


def load_error_tables(paths):
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return table.drop_duplicates(subset=KEY).reset_index(drop=True)


def clean_eer_by_model(table):
    rows = []
    for (model, dataset), group in table.groupby(["model_name", "dataset"]):
        files = group.drop_duplicates(subset="file_id")
        target = files["label"].map(label_to_int).to_numpy()
        if target.min() == target.max():
            rows.append({"model_name": model, "dataset": dataset, "clean_eer": float("nan")})
            continue
        eer, _ = equal_error_rate(files["clean_score"].to_numpy(), target)
        rows.append({"model_name": model, "dataset": dataset, "clean_eer": eer})
    return pd.DataFrame(rows)


def condition_metrics(table):
    baseline = clean_eer_by_model(table)
    clean = {(row.model_name, row.dataset): row.clean_eer for row in baseline.itertuples()}
    rows = []
    for (model, dataset, distortion, severity), group in table.groupby(["model_name", "dataset", "distortion_type", "severity"]):
        target = group["label"].map(label_to_int).to_numpy()
        if target.min() == target.max():
            continue
        scores = group["distorted_score"].to_numpy()
        eer, _ = equal_error_rate(scores, target)
        rows.append({
            "model_name": model,
            "dataset": dataset,
            "distortion_type": distortion,
            "severity": severity,
            "eer": eer,
            "auc": area_under_curve(scores, target),
            "delta_eer": eer - float(clean.get((model, dataset), float("nan"))),
            "error_rate": float((group["error_type"] != "correct").mean()),
            "n": int(len(group)),
        })
    return pd.DataFrame(rows).sort_values(["model_name", "dataset", "distortion_type", "severity"]).reset_index(drop=True)


def worst_delta_pivot(conditions):
    worst = conditions.groupby(["distortion_type", "model_name"])["delta_eer"].max().reset_index()
    return worst.pivot(index="distortion_type", columns="model_name", values="delta_eer").reset_index()


def worst_delta_pivot_by_dataset(conditions):
    worst = conditions.groupby(["dataset", "distortion_type", "model_name"])["delta_eer"].max().reset_index()
    return worst.pivot(index=["dataset", "distortion_type"], columns="model_name", values="delta_eer").reset_index()


def inverse_by_model(table, seed=1337):
    rows = []
    for model, group in table.groupby("model_name"):
        matrix, target, _ = inverse.build_dataset(group)
        _, auc = inverse.fit_inverse(matrix, target, seed=seed)
        rows.append({"model_name": model, "predict_error_auc": auc, "error_rate": float(target.mean()), "n": int(len(target))})
    return pd.DataFrame(rows)


def combine(paths, n_clusters=6, seed=1337):
    table = load_error_tables(paths)
    clustered = clustering.cluster_errors(table, n_clusters=n_clusters, seed=seed)
    cluster_table, cluster_pivot = compare.model_cluster_error(clustered)
    conditions = condition_metrics(table)
    return {
        "error_table_combined": table,
        "clean_eer_by_model": clean_eer_by_model(table),
        "cross_model_conditions": conditions,
        "model_distortion_pivot": worst_delta_pivot(conditions),
        "model_distortion_pivot_by_dataset": worst_delta_pivot_by_dataset(conditions),
        "failure_by_distortion": inverse.failure_by_distortion(table),
        "model_cluster_error": cluster_table,
        "model_cluster_pivot": cluster_pivot,
        "cluster_profile": clustering.cluster_profile(clustered),
        "inverse_by_model": inverse_by_model(table, seed=seed),
    }
