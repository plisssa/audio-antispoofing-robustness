import numpy as np

from ..audio.io import load_audio
from .base import Detector, register_detector

try:
    import torch
    from torch import nn

    class ReferenceNet(nn.Module):
        def __init__(self, n_fft=512, win_length=400, hop_length=160, hidden=64):
            super().__init__()
            self.n_fft = n_fft
            self.win_length = win_length
            self.hop_length = hop_length
            self.register_buffer("window", torch.hann_window(win_length))
            bins = n_fft // 2 + 1
            self.classifier = nn.Sequential(nn.Linear(2 * bins, hidden), nn.ReLU(), nn.Linear(hidden, 2))

        def features(self, waveform):
            spectrum = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=True,
                return_complex=True,
            )
            power = torch.log(spectrum.abs() ** 2 + 1e-6)
            return torch.cat([power.mean(dim=-1), power.std(dim=-1)], dim=-1)

        def forward(self, waveform):
            return self.classifier(self.features(waveform))

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@register_detector("torch_reference")
class TorchReferenceDetector(Detector):
    name = "torch_reference_stft"
    version = "0.1.0"
    trainable = True
    differentiable = True

    def __init__(self, sample_rate=16000, crop=64000, epochs=8, batch_size=16, lr=1e-3, device="cpu", seed=1337):
        self.sample_rate = sample_rate
        self.crop = crop
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed
        self.model = None
        self.torch = None

    def available(self):
        return TORCH_AVAILABLE

    def load(self):
        self.torch = torch
        self.model = ReferenceNet().to(self.device)
        return self

    def prepare(self, signal):
        if len(signal) >= self.crop:
            return signal[: self.crop]
        return np.pad(signal, (0, self.crop - len(signal)))

    def batch(self, signals):
        stacked = np.stack([self.prepare(signal) for signal in signals])
        return self.torch.tensor(stacked, dtype=self.torch.float32, device=self.device)

    def fit(self, items):
        if self.model is None:
            self.load()
        signals, labels = [], []
        for path, label in items:
            signal, _ = load_audio(path, self.sample_rate)
            signals.append(signal)
            labels.append(label)
        inputs = self.batch(signals)
        targets = self.torch.tensor(labels, dtype=self.torch.long, device=self.device)
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = self.torch.nn.CrossEntropyLoss()
        generator = self.torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            order = self.torch.randperm(len(targets), generator=generator)
            for start in range(0, len(targets), self.batch_size):
                index = order[start: start + self.batch_size]
                optimizer.zero_grad()
                loss_fn(self.model(inputs[index]), targets[index]).backward()
                optimizer.step()
        self.model.eval()
        return self

    def logits_tensor(self, x):
        return self.model(x)

    def score_prepared(self, prepared):
        inputs = self.torch.tensor(np.asarray(prepared, dtype=np.float32)[None, :], dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            probability = self.torch.softmax(self.model(inputs), dim=-1)[0, 1]
        return float(probability)

    def score_signal(self, signal, sr):
        return self.score_prepared(self.prepare(signal))

    def score_file(self, path):
        signal, sr = load_audio(path, self.sample_rate)
        return self.score_signal(signal, sr)

    def metadata(self):
        return {
            "name": self.name,
            "version": self.version,
            "crop": self.crop,
            "epochs": self.epochs,
            "differentiable": True,
            "trainable": True,
        }
