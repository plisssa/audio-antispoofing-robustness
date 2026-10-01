import numpy as np

from .base import register
from .codec import transcode
from .noise import add_noise
from .reverb import reverberate


def _resample_roundtrip(signal, sr, target_hz):
    if not target_hz or int(target_hz) >= sr:
        return signal, 0
    from scipy.signal import resample_poly

    down = int(round(sr / float(target_hz)))
    if down < 2:
        return signal, 0
    reduced = resample_poly(signal, 1, down)
    restored = resample_poly(reduced, down, 1)
    if len(restored) < len(signal):
        restored = np.pad(restored, (0, len(signal) - len(restored)))
    return restored[: len(signal)].astype(np.float32), int(target_hz)


@register("laundering")
def launder(signal, sr, level, rt60_s=0.3, color="pink", resample_hz=8000, bitrate=64, codec="mp3", rir_dir=None, salt=0):
    stages = []
    out = signal.astype(np.float32)

    out, _ = reverberate(out, sr, rt60_s, rir_dir=rir_dir, salt=salt)
    stages.append(f"reverb:{rt60_s}")

    out, _ = add_noise(out, sr, level, color=color, salt=salt)
    stages.append(f"noise:{color}@{level}dB")

    out, applied_hz = _resample_roundtrip(out, sr, resample_hz)
    if applied_hz:
        stages.append(f"resample:{applied_hz}")

    out, _ = transcode(out, sr, bitrate, codec=codec)
    stages.append(f"{codec}:{bitrate}k")

    return out.astype(np.float32), {"snr_db": float(level), "chain": "|".join(stages)}
