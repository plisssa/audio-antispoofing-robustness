import numpy as np
from scipy.signal import get_window

FRAME = 400
HOP = 160
N_FFT = 512


def frame_signal(signal, frame=FRAME, hop=HOP):
    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    count = 1 + (len(signal) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return signal[index]


def magnitude_spectrogram(signal, sr, n_fft=N_FFT, frame=FRAME, hop=HOP):
    frames = frame_signal(signal, frame, hop)
    window = get_window("hann", frame, fftbins=True)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return np.abs(spectrum), freqs


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal)) + 1e-12))


def clipping_ratio(signal, threshold=0.99):
    return float(np.mean(np.abs(signal) >= threshold))


def silence_ratio(signal, frame=FRAME, hop=HOP, relative_db=-40.0):
    frames = frame_signal(signal, frame, hop)
    energy = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    threshold = float(np.max(energy)) * (10.0 ** (relative_db / 20.0))
    return float(np.mean(energy < threshold))


def spectral_centroid_bandwidth(signal, sr):
    magnitude, freqs = magnitude_spectrogram(signal, sr)
    weights = magnitude + 1e-12
    total = np.sum(weights, axis=1)
    centroid = np.sum(freqs[None, :] * weights, axis=1) / total
    spread = np.sqrt(np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * weights, axis=1) / total)
    return float(np.mean(centroid)), float(np.mean(spread))


def snr_estimate_db(signal, frame=FRAME, hop=HOP):
    frames = frame_signal(signal, frame, hop)
    energy = np.mean(np.square(frames), axis=1) + 1e-12
    low = float(np.quantile(energy, 0.1))
    high = float(np.quantile(energy, 0.9))
    return float(10.0 * np.log10(high / low))


def describe(signal, sr):
    centroid, bandwidth = spectral_centroid_bandwidth(signal, sr)
    return {
        "duration_s": len(signal) / sr,
        "rms": rms(signal),
        "silence_ratio": silence_ratio(signal),
        "clipping_ratio": clipping_ratio(signal),
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "snr_estimate_db": snr_estimate_db(signal),
    }


def signal_distance(clean, distorted, sr):
    length = min(len(clean), len(distorted))
    reference = clean[:length]
    altered = distorted[:length]
    l2 = float(np.sqrt(np.mean(np.square(reference - altered)) + 1e-12))
    magnitude_reference, _ = magnitude_spectrogram(reference, sr)
    magnitude_altered, _ = magnitude_spectrogram(altered, sr)
    frames = min(magnitude_reference.shape[0], magnitude_altered.shape[0])
    log_reference = np.log(magnitude_reference[:frames] + 1e-6)
    log_altered = np.log(magnitude_altered[:frames] + 1e-6)
    log_spectral = float(np.mean(np.sqrt(np.mean(np.square(log_reference - log_altered), axis=1))))
    return {"l2": l2, "log_spectral_distance": log_spectral}
