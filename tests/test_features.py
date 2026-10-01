import numpy as np

from adfd.audio import features


def test_describe_keys_and_ranges():
    signal = (0.3 * np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)).astype(np.float32)
    described = features.describe(signal, 16000)
    for key in [
        "duration_s",
        "rms",
        "silence_ratio",
        "clipping_ratio",
        "spectral_centroid",
        "spectral_bandwidth",
        "snr_estimate_db",
    ]:
        assert key in described
    assert 0.0 <= described["silence_ratio"] <= 1.0
    assert 0.0 <= described["clipping_ratio"] <= 1.0


def test_signal_distance_zero_for_identical():
    signal = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    distance = features.signal_distance(signal, signal, 16000)
    assert distance["l2"] < 1e-6
