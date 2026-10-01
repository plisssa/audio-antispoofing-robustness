import numpy as np
from scipy.fft import dct

from ..audio.features import magnitude_spectrogram
from .base import register_detector
from .feature_detector import FeatureLogRegDetector


def triangular_filterbank(freqs, n_filters):
    low, high = float(freqs[0]), float(freqs[-1])
    points = np.linspace(low, high, n_filters + 2)
    bank = np.zeros((n_filters, len(freqs)))
    for index in range(n_filters):
        left, center, right = points[index], points[index + 1], points[index + 2]
        rising = (freqs - left) / (center - left + 1e-9)
        falling = (right - freqs) / (right - center + 1e-9)
        bank[index] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def lfcc(signal, sr, n_filters=20, n_ceps=20):
    magnitude, freqs = magnitude_spectrogram(signal, sr)
    power = np.square(magnitude)
    bank = triangular_filterbank(freqs, n_filters)
    energies = np.log(power @ bank.T + 1e-8)
    return dct(energies, type=2, axis=1, norm="ortho")[:, :n_ceps]


@register_detector("reference")
class ReferenceDetector(FeatureLogRegDetector):
    name = "reference_lfcc_logreg"
    version = "0.1.0"

    def __init__(self, sample_rate=16000, n_filters=20, n_ceps=20):
        super().__init__(sample_rate)
        self.n_filters = n_filters
        self.n_ceps = n_ceps

    def embed(self, signal, sr):
        coefficients = lfcc(signal, sr, self.n_filters, self.n_ceps)
        return np.concatenate([coefficients.mean(axis=0), coefficients.std(axis=0)])
