import os

import numpy as np
import pandas as pd

from ..attacks.gradient import LOSS_GRADIENTS, cross_entropy_gradient, logit_margin, margin_gradient
from ..audio.io import load_audio
from ..datasets.manifest import label_to_int
from .attack import attack_label, epsilon_for_snr


def training_modules(detector):
    model = getattr(detector, "model", None)
    if model is None or not hasattr(model, "modules"):
        return 0
    return sum(int(module.training) for module in model.modules())


def checkpoint_mismatch(detector):
    torch = detector.torch
    path = getattr(detector, "checkpoint", None)
    model = getattr(detector, "model", None)
    if not path or not os.path.isfile(str(path)) or model is None:
        return {"missing_keys": None, "unexpected_keys": None, "missing_examples": [], "unexpected_examples": []}
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
    remap = getattr(detector, "remap_state", None)
    if remap is not None:
        state = remap(state, model)
    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    return {"missing_keys": len(missing), "unexpected_keys": len(unexpected), "missing_examples": missing[:15], "unexpected_examples": unexpected[:15]}


def objective(detector, x, label):
    return -float(logit_margin(detector, x.clamp(-1.0, 1.0), label)[0])


def cosine(a, b):
    norm = float(a.norm() * b.norm())
    return float((a * b).sum()) / norm if norm > 0 else float("nan")


def directional_agreement(detector, x, label, gradient, step, directions, generator):
    torch = detector.torch
    base = objective(detector, x, label)
    predicted, actual = [], []
    for _ in range(int(directions)):
        v = (torch.randint(0, 2, tuple(x.shape), generator=generator) * 2 - 1).to(device=x.device, dtype=x.dtype)
        actual.append((objective(detector, x + step * v, label) - objective(detector, x - step * v, label)) / 2.0)
        predicted.append(step * float((gradient * v).sum()))
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    sign_match = float(np.mean(np.sign(predicted) == np.sign(actual)))
    ratio = float(np.median(actual / np.where(predicted == 0, np.nan, predicted)))
    signed = objective(detector, x + step * gradient.sign(), label) - base
    return sign_match, ratio, signed, step * float(gradient.abs().sum())


def file_checks(detector, x, label, fine_step, attack_step, directions, generator):
    torch = detector.torch
    first = logit_margin(detector, x, label)
    second = logit_margin(detector, x, label)
    gradient = margin_gradient(detector, x, label)
    ce = cross_entropy_gradient(detector, x, label)
    record = {
        "margin": float(first[0]),
        "repeat_abs_diff": float((first - second).abs().max()),
        "grad_norm": float(gradient.norm()),
        "grad_zero_frac": float((gradient == 0).float().mean()),
        "grad_finite": bool(torch.isfinite(gradient).all()),
        "ce_grad_norm": float(ce.norm()),
        "ce_margin_cosine": cosine(ce, gradient),
    }
    for tag, step in (("fine", fine_step), ("attack", attack_step)):
        sign_match, ratio, gained, predicted = directional_agreement(detector, x, label, gradient, step, directions, generator)
        record[f"{tag}_sign_match"] = sign_match
        record[f"{tag}_ratio"] = ratio
        record[f"{tag}_sign_step_gain"] = gained
        record[f"{tag}_sign_step_predicted"] = predicted
    return record


def trajectory(detector, x, label, epsilon, steps, alpha_ratio, loss, generator):
    torch = detector.torch
    gradient_of = LOSS_GRADIENTS[loss]
    alpha = max(epsilon * alpha_ratio, epsilon / steps)
    original = x.clone().detach()
    start = (torch.rand(tuple(x.shape), generator=generator) * 2 - 1).to(device=x.device, dtype=x.dtype)
    adversarial = (original + epsilon * start).clamp(-1.0, 1.0).detach()
    values = [objective(detector, adversarial, label)]
    for _ in range(int(steps)):
        adversarial = adversarial + alpha * gradient_of(detector, adversarial, label).sign()
        adversarial = torch.min(torch.max(adversarial, original - epsilon), original + epsilon).clamp(-1.0, 1.0).detach()
        values.append(objective(detector, adversarial, label))
    return values


def gradient_check(detector, evaluation, sample_rate=16000, fine_snr=80.0, attack_snr=50.0, directions=8,
                   trajectory_files=10, trajectory_snr=45.0, trajectory_steps=100, alpha_ratio=0.025, seed=1337):
    torch = detector.torch
    spoof_index = getattr(detector, "spoof_index", 1)
    generator = torch.Generator().manual_seed(int(seed))
    per_file, paths = [], []
    for position, row in enumerate(evaluation.itertuples()):
        signal, _ = load_audio(row.path, sample_rate)
        prepared = np.asarray(detector.prepare(signal), dtype=np.float32)
        power = float(np.mean(np.square(prepared))) + 1e-12
        x = torch.tensor(prepared, dtype=torch.float32, device=detector.device)[None, :]
        label = torch.tensor([attack_label(label_to_int(row.label), spoof_index)], dtype=torch.long, device=detector.device)
        record = {"model": detector.name, "file_id": row.file_id, "label": row.label}
        record.update(file_checks(detector, x, label, epsilon_for_snr(power, fine_snr), epsilon_for_snr(power, attack_snr), directions, generator))
        per_file.append(record)
        if position < int(trajectory_files):
            epsilon = epsilon_for_snr(power, trajectory_snr)
            for loss in ("ce", "margin"):
                values = trajectory(detector, x, label, epsilon, trajectory_steps, alpha_ratio, loss, generator)
                for step, value in enumerate(values):
                    paths.append({"model": detector.name, "file_id": row.file_id, "label": row.label, "loss": loss, "step": step, "objective": value})
    return pd.DataFrame(per_file), pd.DataFrame(paths)


def gradient_summary(per_file, paths):
    rows = []
    for model, group in per_file.groupby("model"):
        row = {"model": model, "n": len(group)}
        for column in ["repeat_abs_diff", "grad_zero_frac", "ce_margin_cosine", "fine_sign_match", "fine_ratio",
                       "attack_sign_match", "attack_ratio"]:
            row[column] = float(group[column].median())
        row["repeat_abs_diff_max"] = float(group["repeat_abs_diff"].max())
        row["grad_finite_all"] = bool(group["grad_finite"].all())
        realised = group["attack_sign_step_gain"] / group["attack_sign_step_predicted"].replace(0, np.nan)
        row["attack_sign_step_realised"] = float(realised.median())
        if not paths.empty:
            trace = paths[paths.model == model]
            for loss, lossgroup in trace.groupby("loss"):
                final = lossgroup[lossgroup.step == lossgroup.step.max()].set_index("file_id").objective
                peak = lossgroup.groupby("file_id").objective.max()
                start = lossgroup[lossgroup.step == 0].set_index("file_id").objective
                row[f"{loss}_final_minus_start"] = float((final - start).median())
                row[f"{loss}_peak_minus_final"] = float((peak - final).median())
                row[f"{loss}_crossed"] = float((final > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)
