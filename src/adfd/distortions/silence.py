import numpy as np

from .base import register


def _samples(sr, milliseconds):
    return int(round(float(milliseconds) * sr / 1000.0))


@register("trim_lead")
def trim_lead(signal, sr, level):
    cut = _samples(sr, level)
    if cut <= 0:
        return signal.astype(np.float32), {"trimmed_ms": 0.0, "kept_fraction": 1.0}
    if cut >= len(signal):
        cut = max(len(signal) - sr // 10, 0)
    out = signal[cut:]
    return out.astype(np.float32), {"trimmed_ms": float(cut * 1000.0 / sr), "kept_fraction": float(len(out) / max(len(signal), 1))}


@register("pad_lead")
def pad_lead(signal, sr, level):
    pad = _samples(sr, level)
    if pad <= 0:
        return signal.astype(np.float32), {"padded_ms": 0.0}
    out = np.concatenate([np.zeros(pad, dtype=np.float32), signal.astype(np.float32)])
    return out, {"padded_ms": float(pad * 1000.0 / sr)}


@register("lead_energy")
def lead_energy(signal, sr, level, window_ms=25.0):
    window = max(_samples(sr, window_ms), 1)
    if len(signal) <= window:
        return signal.astype(np.float32), {"detected_lead_ms": 0.0}
    threshold = 10.0 ** (-abs(float(level)) / 20.0) * float(np.max(np.abs(signal)) + 1e-12)
    index = 0
    while index + window <= len(signal):
        if float(np.sqrt(np.mean(signal[index: index + window] ** 2))) > threshold:
            break
        index += window
    out = signal[index:] if index < len(signal) else signal
    return out.astype(np.float32), {"detected_lead_ms": float(index * 1000.0 / sr)}
