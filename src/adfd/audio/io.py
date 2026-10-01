import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def load_audio(path, target_sr=TARGET_SR):
    signal, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != target_sr:
        signal = resample(signal, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(signal, dtype=np.float32), sr


def resample(signal, sr, target_sr):
    if sr == target_sr:
        return signal.astype(np.float32)
    divisor = np.gcd(int(sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(sr) // divisor
    return resample_poly(signal, up, down).astype(np.float32)


def save_audio(path, signal, sr=TARGET_SR, subtype=None):
    sf.write(str(path), np.asarray(signal, dtype=np.float32), int(sr), subtype=subtype)


def peak_normalize(signal, peak=0.95):
    scale = float(np.max(np.abs(signal))) + 1e-9
    return (signal / scale) * peak
