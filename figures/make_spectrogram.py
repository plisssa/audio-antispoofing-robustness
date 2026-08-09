import argparse
import os
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly, stft

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=None)
parser.add_argument("--out", default="figures/out")
parser.add_argument("--sr", type=int, default=16000)
args = parser.parse_args()

SR = args.sr
os.makedirs(args.out, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "font.family": "DejaVu Sans"})

def load_audio(path, sr):
    fs, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    if fs != sr:
        from math import gcd
        g = gcd(fs, sr)
        x = resample_poly(x, sr // g, fs // g)
    return x

def tts_fallback(sr):
    tmp = tempfile.mktemp(suffix=".wav")
    subprocess.run(["espeak-ng", "-v", "en", "-s", "150", "-w", tmp,
                    "Audio deepfake detectors must remain reliable in real conditions."],
                   check=True)
    return load_audio(tmp, sr)

def add_noise(x, snr_db, seed=7):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x))
    ps = np.mean(x ** 2)
    pn = ps / (10 ** (snr_db / 10))
    return x + n * np.sqrt(pn / np.mean(n ** 2))

def g711_ulaw(x, sr):
    a = tempfile.mktemp(suffix=".wav")
    b = tempfile.mktemp(suffix=".wav")
    c = tempfile.mktemp(suffix=".wav")
    wavfile.write(a, sr, (x * 32767 * 0.9).astype(np.int16))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a,
                    "-ar", "8000", "-acodec", "pcm_mulaw", b], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", b,
                    "-ar", str(sr), "-acodec", "pcm_s16le", c], check=True)
    return load_audio(c, sr)

def reverb(x, sr, rt60=0.5, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rt60 * sr)) / sr
    h = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    h[0] = 1.0
    h = h / np.sqrt(np.sum(h ** 2))
    y = fftconvolve(x, h)[: len(x)]
    return y / np.max(np.abs(y))

def mel_filterbank(n_mels, n_fft, sr, fmax):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)
    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(0, hz2mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, ce, hi = bins[i], bins[i + 1], bins[i + 2]
        if ce == lo:
            ce += 1
        if hi <= ce:
            hi = ce + 1
        fb[i, lo:ce] = (np.arange(lo, ce) - lo) / (ce - lo)
        fb[i, ce:hi] = (hi - np.arange(ce, hi)) / (hi - ce)
    return fb, pts[1:-1]

def melspec(x, sr, n_fft=512, hop=160, n_mels=80):
    f, tt, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, padded=False)
    S = np.abs(Z) ** 2
    fb, centers = mel_filterbank(n_mels, n_fft, sr, sr / 2)
    M = fb @ S
    return 10 * np.log10(np.maximum(M, 1e-10)), tt, centers

L = {"ru": dict(t="Мел-спектрограммы одной фразы под четырьмя условиями",
                a="(а) исходный сигнал", b="(б) шум, SNR 10 дБ",
                c="(в) телефония G.711 μ-law, 8 кГц", d="(г) реверберация, RT60 0,5 с",
                x="время, с", y="частота, кГц", cb="дБ"),
     "en": dict(t="Mel-spectrograms of one utterance under four conditions",
                a="(a) original signal", b="(b) noise, SNR 10 dB",
                c="(c) G.711 μ-law telephony, 8 kHz", d="(d) reverberation, RT60 0.5 s",
                x="time, s", y="frequency, kHz", cb="dB")}

x = load_audio(args.input, SR) if args.input else tts_fallback(SR)
x = x[: int(3.2 * SR)]
variants = [x, add_noise(x, 10), g711_ulaw(x, SR), reverb(x, SR)]

specs = [melspec(v, SR) for v in variants]
vmax = max(S.max() for S, _, _ in specs)
vmin = vmax - 70

for lang in ["ru", "en"]:
    t = L[lang]
    keys = ["a", "b", "c", "d"]
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True, sharey=True)
    for ax, (S, tt, centers), k in zip(axs.ravel(), specs, keys):
        im = ax.imshow(S, aspect="auto", origin="lower", cmap="magma",
                       vmin=vmin, vmax=vmax,
                       extent=[tt[0], tt[-1], 0, S.shape[0]])
        ax.set_title(t[k], fontsize=10)
        hz_ticks = [500, 1000, 2000, 4000, 8000]
        idx = [int(np.argmin(np.abs(centers - h))) for h in hz_ticks]
        ax.set_yticks(idx)
        ax.set_yticklabels([f"{h/1000:g}" for h in hz_ticks])
    for ax in axs[1]:
        ax.set_xlabel(t["x"])
    for ax in axs[:, 0]:
        ax.set_ylabel(t["y"])
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)
    cb.set_label(t["cb"])
    fig.suptitle(t["t"], fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{args.out}/fig_spectrogram_{lang}.{ext}",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

print("ok")
