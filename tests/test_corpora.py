import numpy as np
import soundfile as sf

from adfd.distortions import apply_distortion, file_salt


def make_signal(sr=16000, seconds=1.0):
    time = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def write_corpus(directory, count=4, sr=16000):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        sf.write(str(directory / f"noise_{index}.wav"), rng.standard_normal(sr).astype(np.float32), sr)
    return str(directory)


def test_real_noise_reaches_target_snr(tmp_path):
    corpus = write_corpus(tmp_path / "musan")
    signal = make_signal()
    out, strength = apply_distortion("noise", signal, 16000, 10, {"color": "file", "corpus": corpus}, salt=7)
    assert out.shape == signal.shape
    assert abs(strength["achieved_snr_db"] - 10) < 1.0


def test_real_rir_convolves(tmp_path):
    rir_dir = tmp_path / "rirs"
    rir_dir.mkdir(parents=True, exist_ok=True)
    impulse = np.zeros(2000, dtype=np.float32)
    impulse[0] = 1.0
    impulse[500] = 0.5
    sf.write(str(rir_dir / "room.wav"), impulse, 16000)
    signal = make_signal()
    out, strength = apply_distortion("reverb", signal, 16000, 0, {"rir_dir": str(rir_dir)}, salt=1)
    assert out.shape == signal.shape
    assert strength["real_rir"] is True


def test_salt_changes_realization_but_is_reproducible():
    signal = make_signal()
    a, _ = apply_distortion("noise", signal, 16000, 10, {"color": "pink"}, salt=file_salt("a"))
    b, _ = apply_distortion("noise", signal, 16000, 10, {"color": "pink"}, salt=file_salt("b"))
    a_again, _ = apply_distortion("noise", signal, 16000, 10, {"color": "pink"}, salt=file_salt("a"))
    assert not np.allclose(a, b)
    assert np.allclose(a, a_again)


def test_codec_raises_on_unknown_codec():
    signal = make_signal()
    try:
        apply_distortion("codec", signal, 16000, 32, {"codec": "does_not_exist"})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown codec")
