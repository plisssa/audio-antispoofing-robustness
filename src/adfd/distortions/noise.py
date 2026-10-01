import numpy as np

from .base import register, rng_for
from .corpora import fit_length, load_sample


def pink_noise(length, rng):
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    shaped = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(shaped, n=length)
    return pink / (np.std(pink) + 1e-9)


def chirp_bed(length, sr, rng):
    bed = np.zeros(length)
    count = max(1, length // sr * 6)
    for _ in range(count):
        start = rng.integers(0, max(1, length - sr // 4))
        span = rng.integers(sr // 20, sr // 4)
        end = min(length, start + span)
        time = np.arange(end - start) / sr
        f0 = rng.uniform(2000, 6000)
        f1 = f0 + rng.uniform(-1500, 1500)
        sweep = np.sin(2 * np.pi * (f0 * time + (f1 - f0) / (2 * (time[-1] + 1e-6)) * time ** 2))
        envelope = np.hanning(len(time))
        bed[start:end] += sweep * envelope
    return bed / (np.std(bed) + 1e-9)


def real_noise(length, sr, rng, corpus):
    sample = load_sample(corpus, sr, rng)
    fitted = fit_length(sample, length, rng)
    return fitted / (np.std(fitted) + 1e-9)


@register("noise")
def add_noise(signal, sr, level, color="white", corpus=None, salt=0):
    rng = rng_for(level, len(signal), salt)
    length = len(signal)
    if color == "pink":
        noise = pink_noise(length, rng)
    elif color == "chirp":
        noise = chirp_bed(length, sr, rng)
    elif color == "file":
        if not corpus:
            raise ValueError("noise color='file' requires a corpus directory option")
        noise = real_noise(length, sr, rng, corpus)
    else:
        noise = rng.standard_normal(length)
    signal_power = float(np.mean(np.square(signal))) + 1e-12
    noise_power = float(np.mean(np.square(noise))) + 1e-12
    target_noise_power = signal_power / (10.0 ** (float(level) / 10.0))
    scaled = noise * np.sqrt(target_noise_power / noise_power)
    out = signal + scaled
    achieved = 10.0 * np.log10(signal_power / (float(np.mean(np.square(scaled))) + 1e-12))
    return out.astype(np.float32), {
        "snr_db": float(level),
        "achieved_snr_db": float(achieved),
        "color": color,
    }
