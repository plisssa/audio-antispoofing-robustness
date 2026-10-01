import numpy as np

from ..audio.features import magnitude_spectrogram
from .base import register_detector
from .feature_detector import FeatureLogRegDetector


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(freqs, n_filters):
    low, high = hz_to_mel(float(freqs[0])), hz_to_mel(float(freqs[-1]))
    points = mel_to_hz(np.linspace(low, high, n_filters + 2))
    bank = np.zeros((n_filters, len(freqs)))
    for index in range(n_filters):
        left, center, right = points[index], points[index + 1], points[index + 2]
        rising = (freqs - left) / (center - left + 1e-9)
        falling = (right - freqs) / (right - center + 1e-9)
        bank[index] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


@register_detector("melspec")
class MelSpecDetector(FeatureLogRegDetector):
    name = "melspec_logreg"
    version = "0.1.0"

    def __init__(self, sample_rate=16000, n_filters=40):
        super().__init__(sample_rate)
        self.n_filters = n_filters

    def embed(self, signal, sr):
        magnitude, freqs = magnitude_spectrogram(signal, sr)
        log_energies = np.log(np.square(magnitude) @ mel_filterbank(freqs, self.n_filters).T + 1e-8)
        return np.concatenate([log_energies.mean(axis=0), log_energies.std(axis=0)])
