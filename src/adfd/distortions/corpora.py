from pathlib import Path

import numpy as np

from ..audio.io import load_audio

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aac", ".m4a"}
_BANKS = {}


def list_audio(directory):
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {directory}")
    files = sorted(str(path) for path in root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no audio files under corpus directory: {directory}")
    return files


def bank(directory):
    if directory not in _BANKS:
        _BANKS[directory] = list_audio(directory)
    return _BANKS[directory]


def pick_file(directory, rng):
    files = bank(directory)
    return files[int(rng.integers(0, len(files)))]


def load_sample(directory, sr, rng):
    path = pick_file(directory, rng)
    sample, _ = load_audio(path, sr)
    return np.asarray(sample, dtype=np.float32)


def fit_length(sample, length, rng):
    if len(sample) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(sample) > length:
        start = int(rng.integers(0, len(sample) - length + 1))
        return sample[start:start + length]
    if len(sample) == length:
        return sample
    repeats = int(np.ceil(length / len(sample)))
    return np.tile(sample, repeats)[:length]
