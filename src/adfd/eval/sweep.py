import pandas as pd

from ..audio.io import load_audio
from ..datasets.manifest import label_to_int
from ..distortions import apply_distortion, file_salt
from ..repro.checkpoint import append_rows, done_ids, load_partial
from .metrics import bootstrap_delta_eer_ci, relative_degradation, summary


def severity_of(spec, level):
    if spec.get("kind") == "gain":
        return abs(float(level))
    return float(spec.get("severity_sign", 1.0)) * float(level)


def strength_value(strength, param):
    for key in ("achieved_snr_db", "achieved_loss_rate"):
        if key in strength:
            return float(strength[key])
    if param in strength:
        return float(strength[param])
    return float("nan")


def error_type(target, prediction):
    if prediction == target:
        return "correct"
    return "false_negative" if target == 1 else "false_positive"


def clean_scores_map(detector, manifest, partial_path=None):
    scores = {}
    previous = load_partial(partial_path) if partial_path else None
    if previous is not None:
        scores = dict(zip(previous["file_id"], previous["score"]))
    for row in manifest.itertuples():
        if row.file_id in scores:
            continue
        value = detector.score_file(row.path)
        scores[row.file_id] = value
        if partial_path:
            append_rows(partial_path, [{"file_id": row.file_id, "score": value}])
    return scores


def run_sweep(detector, manifest, distortions, clean_scores, threshold, sample_rate=16000, partial_path=None):
    done = done_ids(partial_path) if partial_path else set()
    collected = []
    previous = load_partial(partial_path) if partial_path else None
    if previous is not None:
        collected.append(previous)
    for row in manifest.itertuples():
        if row.file_id in done:
            continue
        signal, sr = load_audio(row.path, sample_rate)
        target = label_to_int(row.label)
        clean_score = clean_scores[row.file_id]
        clean_prediction = int(clean_score >= threshold)
        salt = file_salt(row.file_id)
        file_rows = []
        for name, spec in distortions.items():
            options = spec.get("options") or {}
            param = spec["param"]
            family = spec.get("family", spec["kind"])
            for level in spec["levels"]:
                distorted, strength = apply_distortion(spec["kind"], signal, sr, level, options, salt=salt)
                score = detector.score_signal(distorted, sr)
                prediction = int(score >= threshold)
                file_rows.append({
                    "file_id": row.file_id,
                    "dataset": row.dataset,
                    "label": row.label,
                    "target": target,
                    "distortion_type": name,
                    "kind": spec["kind"],
                    "family": family,
                    "param": param,
                    "level": float(level),
                    "severity": severity_of(spec, level),
                    "strength_value": strength_value(strength, param),
                    "clean_score": clean_score,
                    "distorted_score": score,
                    "delta_score": score - clean_score,
                    "prediction_clean": clean_prediction,
                    "prediction_distorted": prediction,
                    "error_type": error_type(target, prediction),
                    "flipped": int(prediction != clean_prediction),
                })
        if partial_path:
            append_rows(partial_path, file_rows)
        collected.append(pd.DataFrame(file_rows))
    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def aggregate_sweep(sweep, clean_eer, dcf=None, n_boot=1000, seed=1337):
    records = []
    for (distortion, level), group in sweep.groupby(["distortion_type", "level"]):
        report = summary(group["distorted_score"].to_numpy(), group["target"].to_numpy(), dcf=dcf)
        ci_low, ci_high = bootstrap_delta_eer_ci(
            group["clean_score"].to_numpy(),
            group["distorted_score"].to_numpy(),
            group["target"].to_numpy(),
            n_boot=n_boot,
            seed=seed,
        )
        records.append({
            "distortion_type": distortion,
            "level": level,
            "severity": float(group["severity"].iloc[0]),
            "strength_value": float(group["strength_value"].mean()),
            "eer": report["eer"],
            "auc": report["auc"],
            "min_dcf": report["min_dcf"],
            "cllr": report["cllr"],
            "delta_eer": report["eer"] - clean_eer,
            "delta_eer_ci_low": ci_low,
            "delta_eer_ci_high": ci_high,
            "relative_degradation": relative_degradation(clean_eer, report["eer"]),
            "mean_abs_delta_score": float(group["delta_score"].abs().mean()),
            "flip_rate": float(group["flipped"].mean()),
            "false_negative": int((group["error_type"] == "false_negative").sum()),
            "false_positive": int((group["error_type"] == "false_positive").sum()),
        })
    return pd.DataFrame(records).sort_values(["distortion_type", "severity"]).reset_index(drop=True)
