import numpy as np
import pandas as pd

from ..attacks.base import get_attack
from ..audio.io import load_audio
from ..datasets.manifest import label_to_int
from ..distortions.base import apply_distortion, file_salt
from ..repro.checkpoint import append_rows, done_ids, load_partial
from .metrics import attack_success_rate, summary
from .sweep import aggregate_sweep, error_type


def epsilon_for_snr(power, snr_db):
    return float(np.sqrt(power / (10.0 ** (float(snr_db) / 10.0))))


def attack_label(target, spoof_index):
    return spoof_index if target == 1 else 1 - spoof_index


def apply_defence(detector, signal, sr, spec, salt):
    raw = detector.to_raw(signal)
    defended, _ = apply_distortion(spec["kind"], raw, sr, spec["level"], spec.get("options"), salt)
    return np.asarray(detector.prepare(np.asarray(defended, dtype=np.float32)), dtype=np.float32)


def run_attack(detector, manifest, attack_name, levels, clean_scores, threshold, options=None, sample_rate=16000, partial_path=None, budget="snr", transfer=None, defences=None):
    torch = detector.torch
    attack = get_attack(attack_name)
    options = dict(options or {})
    budget = options.pop("budget", budget)
    transfer = transfer or []
    defences = defences or {}
    spoof_index = getattr(detector, "spoof_index", 1)
    done = done_ids(partial_path) if partial_path else set()
    collected = []
    previous = load_partial(partial_path) if partial_path else None
    if previous is not None:
        collected.append(previous)
    for row in manifest.itertuples():
        if row.file_id in done:
            continue
        signal, sr = load_audio(row.path, sample_rate)
        prepared = np.asarray(detector.prepare(signal), dtype=np.float32)
        power = float(np.mean(np.square(prepared))) + 1e-12
        target = label_to_int(row.label)
        clean_score = clean_scores[row.file_id]
        clean_prediction = int(clean_score >= threshold)
        x = torch.tensor(prepared, dtype=torch.float32, device=detector.device)[None, :]
        label = torch.tensor([attack_label(target, spoof_index)], dtype=torch.long, device=detector.device)
        salt = file_salt(row.file_id)
        defence_clean = {}
        for name, spec in defences.items():
            defended_clean = apply_defence(detector, prepared, sr, spec, salt)
            defence_clean[name] = detector.score_prepared(defended_clean)
        file_rows = []
        for level in levels:
            if budget == "snr":
                epsilon = epsilon_for_snr(power, level)
                target_snr = float(level)
            else:
                epsilon = float(level)
                target_snr = float("nan")
            adversarial = attack(detector, x, label, epsilon, **options)
            adversarial_signal = adversarial[0].detach().cpu().numpy()
            perturbation_power = float(np.mean(np.square(adversarial_signal - prepared))) + 1e-12
            achieved_snr = 10.0 * np.log10(power / perturbation_power)
            score = detector.score_prepared(adversarial_signal)
            prediction = int(score >= threshold)
            record = {
                "file_id": row.file_id,
                "dataset": row.dataset,
                "label": row.label,
                "target": target,
                "distortion_type": attack_name,
                "kind": attack_name,
                "family": "adversarial_attacks",
                "param": "snr_db" if budget == "snr" else "epsilon",
                "level": float(level),
                "severity": -target_snr if budget == "snr" else float(level),
                "strength_value": float(achieved_snr) if budget == "snr" else float(np.max(np.abs(adversarial_signal - prepared))),
                "epsilon": epsilon,
                "clean_score": clean_score,
                "distorted_score": score,
                "delta_score": score - clean_score,
                "prediction_clean": clean_prediction,
                "prediction_distorted": prediction,
                "error_type": error_type(target, prediction),
                "flipped": int(prediction != clean_prediction),
            }
            raw_adversarial = detector.to_raw(adversarial_signal)
            for other in transfer:
                record[f"transfer_{other.name}"] = other.score_signal(raw_adversarial, sr)
            for name, spec in defences.items():
                defended = apply_defence(detector, adversarial_signal, sr, spec, salt)
                defended_score = detector.score_prepared(defended)
                record[f"defence_{name}"] = defended_score
                record[f"defence_pred_{name}"] = int(defended_score >= threshold)
                record[f"defence_clean_{name}"] = defence_clean[name]
                record[f"defence_clean_pred_{name}"] = int(defence_clean[name] >= threshold)
            file_rows.append(record)
        if partial_path:
            append_rows(partial_path, file_rows)
        collected.append(pd.DataFrame(file_rows))
    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def attack_aggregate(attack_table, clean_eer, dcf=None, n_boot=1000, seed=1337):
    aggregate = aggregate_sweep(attack_table, clean_eer, dcf=dcf, n_boot=n_boot, seed=seed)
    rates = []
    for (distortion, level), group in attack_table.groupby(["distortion_type", "level"]):
        rates.append({
            "distortion_type": distortion,
            "level": level,
            "attack_success_rate": attack_success_rate(
                group["target"], group["prediction_clean"], group["prediction_distorted"]
            ),
        })
    return aggregate.merge(pd.DataFrame(rates), on=["distortion_type", "level"], how="left")


def defence_aggregate(attack_table, defence_names, clean_eer):
    records = []
    for name in defence_names:
        column = f"defence_{name}"
        if column not in attack_table.columns:
            continue
        for (attack, level), group in attack_table.groupby(["distortion_type", "level"]):
            target = group["target"].to_numpy()
            defended = summary(group[column].to_numpy(), target)
            row = {
                "defence": name,
                "distortion_type": attack,
                "level": level,
                "eer_attacked": summary(group["distorted_score"].to_numpy(), target)["eer"],
                "asr_attacked": attack_success_rate(group["target"], group["prediction_clean"], group["prediction_distorted"]),
                "eer_defended": defended["eer"],
                "asr_defended": attack_success_rate(group["target"], group["prediction_clean"], group[f"defence_pred_{name}"]),
                "cllr_defended": defended["cllr"],
                "delta_eer_defended": defended["eer"] - clean_eer,
            }
            clean_column = f"defence_clean_{name}"
            if clean_column in attack_table.columns:
                clean_defended = summary(group[clean_column].to_numpy(), target)
                row["eer_clean_defended"] = clean_defended["eer"]
                row["defence_cost_eer"] = clean_defended["eer"] - clean_eer
            records.append(row)
    return pd.DataFrame(records)


def transfer_aggregate(attack_table, transfer_names, clean_eers):
    records = []
    for name in transfer_names:
        column = f"transfer_{name}"
        if column not in attack_table.columns:
            continue
        clean_eer = clean_eers.get(name, float("nan"))
        for (distortion, level), group in attack_table.groupby(["distortion_type", "level"]):
            report = summary(group[column].to_numpy(), group["target"].to_numpy())
            records.append({
                "transfer_model": name,
                "distortion_type": distortion,
                "level": level,
                "eer": report["eer"],
                "delta_eer": report["eer"] - clean_eer,
                "auc": report["auc"],
                "cllr": report["cllr"],
                "min_cllr": report["min_cllr"],
                "calibration_loss": report["cllr"] - report["min_cllr"],
            })
    return pd.DataFrame(records)
