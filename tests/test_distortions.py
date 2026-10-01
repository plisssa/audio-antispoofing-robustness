import shutil

import numpy as np
import pytest

from adfd.distortions import apply_distortion, available_distortions


def make_signal(sr=16000, seconds=1.0):
    time = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def test_registry_has_expected():
    for name in ["noise", "codec", "transmission", "reverb", "gain", "bandpass"]:
        assert name in available_distortions()


def test_noise_reaches_target_snr():
    signal = make_signal()
    out, strength = apply_distortion("noise", signal, 16000, 10, {"color": "white"})
    assert out.shape == signal.shape
    assert abs(strength["achieved_snr_db"] - 10) < 1.0


def test_transmission_loss_rate_bounded():
    signal = make_signal()
    out, strength = apply_distortion("transmission", signal, 16000, 0.2, {"packet_ms": 20})
    assert out.shape == signal.shape
    assert 0.0 <= strength["achieved_loss_rate"] <= 1.0


def test_bandpass_reports_band():
    signal = make_signal()
    out, strength = apply_distortion("bandpass", signal, 16000, 2000, {"low_hz": 100})
    assert out.shape == signal.shape
    assert strength["high_hz"] == 2000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_codec_applied():
    signal = make_signal()
    out, strength = apply_distortion("codec", signal, 16000, 32, {"codec": "mp3"})
    assert strength["applied"]
    assert out.shape == signal.shape
