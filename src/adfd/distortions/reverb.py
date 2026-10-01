import numpy as np

from .base import register, rng_for
from .corpora import load_sample


def synthetic_impulse(sr, rt60, rng):
    length = max(2, int(sr * rt60))
    time = np.arange(length) / sr
    impulse = rng.standard_normal(length) * np.exp(-6.908 * time / rt60)
    impulse[0] = 1.0
    return impulse


def real_impulse(sr, rir_dir, rng, max_seconds=1.0):
    impulse = load_sample(rir_dir, sr, rng)
    limit = max(2, int(sr * max_seconds))
    peak = int(np.argmax(np.abs(impulse)))
    return impulse[peak:peak + limit]


@register("reverb")
def reverberate(signal, sr, level, rir_dir=None, salt=0):
    rng = rng_for(level, len(signal), salt)
    if rir_dir:
        impulse = real_impulse(sr, rir_dir, rng)
        rt60 = float(level)
        real = True
    else:
        rt60 = max(0.05, float(level))
        impulse = synthetic_impulse(sr, rt60, rng)
        real = False
    impulse = impulse / (np.sqrt(np.sum(np.square(impulse))) + 1e-9)
    out = np.convolve(signal, impulse)[: len(signal)]
    return out.astype(np.float32), {"rt60_s": rt60, "real_rir": real}
