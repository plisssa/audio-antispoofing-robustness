import numpy as np
import pandas as pd

from ..attacks.gradient import cross_entropy_gradient
from ..audio.io import load_audio
from ..datasets.manifest import label_to_int
from .attack import attack_label


def balanced_sample(evaluation, n, seed):
    per_class = max(n // 2, 1)
    parts = []
    for _, group in evaluation.groupby("label"):
        parts.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    return pd.concat(parts).sort_values("file_id").reset_index(drop=True)


def input_gradients(detector, evaluation, sample_rate=16000):
    torch = detector.torch
    spoof_index = getattr(detector, "spoof_index", 1)
    gradients, support = {}, {}
    for row in evaluation.itertuples():
        signal, _ = load_audio(row.path, sample_rate)
        prepared = np.asarray(detector.prepare(signal), dtype=np.float32)
        x = torch.tensor(prepared, dtype=torch.float32, device=detector.device)[None, :]
        label = torch.tensor([attack_label(label_to_int(row.label), spoof_index)], dtype=torch.long, device=detector.device)
        gradient = cross_entropy_gradient(detector, x, label)[0].float().cpu().numpy()
        gradients[row.file_id] = np.asarray(detector.raw_gradient(gradient), dtype=np.float32)
        support[row.file_id] = min(len(signal), len(prepared))
    return gradients, support


def alignment(source, victim):
    norm = float(np.linalg.norm(source) * np.linalg.norm(victim))
    cosine = float(np.dot(source, victim) / norm) if norm > 0 else float("nan")
    scale = float(np.sum(np.abs(victim)))
    sign = float(np.dot(np.sign(source), victim) / scale) if scale > 0 else float("nan")
    return cosine, sign


def pair_alignment(gradients, support):
    names = sorted(gradients)
    if len(names) < 2:
        return pd.DataFrame(), pd.DataFrame()
    files = sorted(set.intersection(*(set(gradients[name]) for name in names)))
    summary, per_file = [], []
    for source in names:
        for victim in names:
            if source == victim:
                continue
            cosines, signs = [], []
            for file_id in files:
                length = min(support[file_id], len(gradients[source][file_id]), len(gradients[victim][file_id]))
                cosine, sign = alignment(gradients[source][file_id][:length], gradients[victim][file_id][:length])
                cosines.append(cosine)
                signs.append(sign)
                per_file.append({"source": source, "victim": victim, "file_id": file_id,
                                 "cosine": cosine, "sign_alignment": sign, "length": length})
            summary.append({
                "source": source,
                "victim": victim,
                "cosine_mean": float(np.nanmean(cosines)),
                "cosine_median": float(np.nanmedian(cosines)),
                "sign_alignment_mean": float(np.nanmean(signs)),
                "sign_alignment_median": float(np.nanmedian(signs)),
                "n": len(files),
            })
    return pd.DataFrame(summary), pd.DataFrame(per_file)
