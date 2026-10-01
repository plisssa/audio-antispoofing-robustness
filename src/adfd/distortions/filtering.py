import numpy as np
from scipy.signal import butter, sosfilt

from .base import register


@register("bandpass")
def bandpass(signal, sr, level, low_hz=100):
    nyquist = sr / 2.0
    high_hz = min(float(level), nyquist * 0.99)
    low = max(10.0, float(low_hz)) / nyquist
    high = high_hz / nyquist
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    out = sosfilt(sos, signal)
    return out.astype(np.float32), {"low_hz": float(low_hz), "high_hz": float(high_hz)}
