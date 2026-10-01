from pathlib import Path

import numpy as np

from ..audio.io import peak_normalize, save_audio
from ..distortions.noise import pink_noise
from .manifest import write_manifest


def harmonic_voice(length, sr, rng, n_harmonics, decay):
    time = np.arange(length) / sr
    f0 = rng.uniform(110, 210)
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * rng.uniform(4, 7) * time)
    phase = 2 * np.pi * f0 * np.cumsum(vibrato) / sr
    voice = np.zeros(length)
    for harmonic in range(1, n_harmonics + 1):
        voice += (harmonic ** (-decay)) * np.sin(harmonic * phase + rng.uniform(0, 2 * np.pi))
    modulation = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(2, 5) * time + rng.uniform(0, 2 * np.pi))
    return voice * modulation


def make_bona_fide(length, sr, rng):
    voice = harmonic_voice(length, sr, rng, n_harmonics=int(rng.integers(10, 16)), decay=1.0)
    return voice + 0.03 * pink_noise(length, rng)


def make_spoof(length, sr, rng):
    voice = harmonic_voice(length, sr, rng, n_harmonics=int(rng.integers(12, 18)), decay=0.8)
    time = np.arange(length) / sr
    buzz = 0.012 * np.sin(2 * np.pi * 7000 * time)
    return voice + buzz + 0.03 * pink_noise(length, rng)


def make_bootstrap(out_dir, n_per_class=80, sr=16000, seed=1337, train_ratio=0.6):
    rng = np.random.default_rng(seed)
    audio_dir = Path(out_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    train_count = int(n_per_class * train_ratio)
    for index in range(n_per_class):
        split = "train" if index < train_count else "eval"
        for label in ("bona_fide", "spoof"):
            length = int(sr * rng.uniform(1.2, 2.5))
            signal = make_bona_fide(length, sr, rng) if label == "bona_fide" else make_spoof(length, sr, rng)
            signal = peak_normalize(signal.astype(np.float32))
            file_id = f"{label}_{index:03d}"
            path = audio_dir / f"{file_id}.wav"
            save_audio(path, signal, sr)
            rows.append({
                "file_id": file_id,
                "path": str(path),
                "label": label,
                "split": split,
                "dataset": "bootstrap",
                "source": "synthetic",
            })
    manifest_path = Path(out_dir) / "manifest.csv"
    write_manifest(manifest_path, rows)
    return manifest_path
