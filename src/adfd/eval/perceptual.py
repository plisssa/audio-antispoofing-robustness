from pathlib import Path

import numpy as np
import pandas as pd

from ..attacks.base import get_attack
from ..audio.io import load_audio, save_audio
from ..datasets.manifest import label_to_int
from .attack import attack_label, epsilon_for_snr


def quality(reference, degraded, sample_rate=16000):
    from pesq import pesq
    from pystoi import stoi

    try:
        pesq_wb = float(pesq(sample_rate, reference, degraded, "wb"))
    except Exception:
        pesq_wb = float("nan")
    return {"pesq_wb": pesq_wb, "stoi": float(stoi(reference, degraded, sample_rate, extended=False))}


def matched_noise(perturbation, generator):
    power = float(np.mean(np.square(perturbation)))
    noise = generator.standard_normal(len(perturbation)).astype(np.float32)
    return noise * np.float32(np.sqrt(power / (float(np.mean(np.square(noise))) + 1e-12)))


def pcm16(signal):
    return (np.round(np.clip(signal, -1.0, 1.0) * 32767.0) / 32767.0).astype(np.float32)


def snr_db(reference, perturbation):
    return float(10.0 * np.log10((float(np.mean(np.square(reference))) + 1e-12) / (float(np.mean(np.square(perturbation))) + 1e-12)))


def perceptual_rows(detector, evaluation, attacks, sample_rate=16000, seed=1337, save_dir=None, save_n=0, measure=quality):
    torch = detector.torch
    spoof_index = getattr(detector, "spoof_index", 1)
    generator = np.random.default_rng(int(seed))
    rows = []
    for position, row in enumerate(evaluation.itertuples()):
        signal, _ = load_audio(row.path, sample_rate)
        prepared = np.asarray(detector.prepare(signal), dtype=np.float32)
        reference = np.asarray(detector.to_raw(prepared), dtype=np.float32)
        power = float(np.mean(np.square(prepared))) + 1e-12
        x = torch.tensor(prepared, dtype=torch.float32, device=detector.device)[None, :]
        label = torch.tensor([attack_label(label_to_int(row.label), spoof_index)], dtype=torch.long, device=detector.device)
        clean_score = detector.score_prepared(prepared)
        keep = save_dir is not None and position < int(save_n)
        if keep:
            folder = Path(save_dir) / detector.name
            folder.mkdir(parents=True, exist_ok=True)
            save_audio(folder / f"{row.file_id}_clean.wav", reference, sample_rate, subtype="FLOAT")
        for name, spec in attacks.items():
            attack = get_attack(spec["attack"])
            options = {key: value for key, value in (spec.get("options") or {}).items() if key != "budget"}
            for level in spec["levels"]:
                epsilon = epsilon_for_snr(power, level)
                adversarial = attack(detector, x, label, epsilon, **options)[0].detach().cpu().numpy()
                degraded = np.asarray(detector.to_raw(adversarial), dtype=np.float32)
                perturbation = degraded - reference
                noisy = np.clip(reference + matched_noise(perturbation, generator), -1.0, 1.0).astype(np.float32)
                achieved = snr_db(reference, perturbation)
                quantized = pcm16(degraded)
                scores = {
                    "clean_score": clean_score,
                    "attack_score": detector.score_prepared(adversarial),
                    "pcm16_score": detector.score_prepared(np.asarray(detector.prepare(quantized), dtype=np.float32)),
                    "pcm16_kept": float(np.mean(np.square(quantized - pcm16(reference)))) / (float(np.mean(np.square(perturbation))) + 1e-12),
                }
                for kind, candidate in (("attack", degraded), ("noise", noisy)):
                    record = {"model": detector.name, "file_id": row.file_id, "label": row.label, "attack": name,
                              "level": float(level), "kind": kind, "snr": achieved}
                    record.update(measure(reference, candidate, sample_rate))
                    if kind == "attack":
                        record.update(scores)
                    rows.append(record)
                if keep:
                    save_audio(folder / f"{row.file_id}_{name}_{int(level)}dB.wav", degraded, sample_rate, subtype="FLOAT")
                    save_audio(folder / f"{row.file_id}_noise_{int(level)}dB.wav", noisy, sample_rate, subtype="FLOAT")
    return pd.DataFrame(rows)


def perceptual_summary(rows):
    aggregations = {
        "n": ("file_id", "size"),
        "snr": ("snr", "median"),
        "pesq_wb": ("pesq_wb", "median"),
        "pesq_wb_p10": ("pesq_wb", lambda s: float(s.quantile(0.1))),
        "stoi": ("stoi", "median"),
        "stoi_p10": ("stoi", lambda s: float(s.quantile(0.1))),
    }
    if "pcm16_kept" in rows.columns:
        aggregations["pcm16_kept"] = ("pcm16_kept", "median")
    summary = rows.groupby(["model", "attack", "level", "kind"]).agg(**aggregations)
    return summary.reset_index().sort_values(["model", "attack", "kind", "level"], ascending=[True, True, True, False])
