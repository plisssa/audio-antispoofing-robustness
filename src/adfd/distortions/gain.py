import numpy as np

from .base import register


@register("gain")
def apply_gain(signal, sr, level):
    factor = 10.0 ** (float(level) / 20.0)
    amplified = signal * factor
    out = np.clip(amplified, -1.0, 1.0)
    clipped_fraction = float(np.mean(np.abs(amplified) > 1.0))
    return out.astype(np.float32), {"gain_db": float(level), "clipped_fraction": clipped_fraction}
