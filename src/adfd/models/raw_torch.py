import sys
from pathlib import Path

import numpy as np

from ..audio.io import load_audio
from .base import Detector, logits_to_spoof_probability


class RawWaveformTorchDetector(Detector):
    trainable = False
    differentiable = True

    def __init__(self, sample_rate=16000, repo_path=None, config=None, checkpoint=None, device=None, spoof_index=0, crop=64600, extra=None):
        self.sample_rate = sample_rate
        self.repo_path = repo_path
        self.config = config
        self.checkpoint = checkpoint
        self.device = device
        self.spoof_index = spoof_index
        self.crop = crop
        self.extra = extra or {}
        self.model = None
        self.torch = None

    def available(self):
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return bool(self.repo_path and self.checkpoint)

    def build_model(self, torch):
        raise NotImplementedError

    def remap_state(self, state, model):
        return state

    def load(self):
        import torch

        if self.repo_path:
            repo = str(Path(self.repo_path).resolve())
            if repo in sys.path:
                sys.path.remove(repo)
            sys.path.insert(0, repo)
        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.build_model(torch)
        state = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}
        state = self.remap_state(state, model)
        model.load_state_dict(state, strict=False)
        self.model = model.to(self.device).eval()
        return self

    def prepare(self, signal):
        if not self.crop:
            return signal
        if len(signal) >= self.crop:
            return signal[: self.crop]
        repeats = int(np.ceil(self.crop / len(signal)))
        return np.tile(signal, repeats)[: self.crop]

    def logits_tensor(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        return output

    def score_prepared(self, prepared):
        tensor = self.torch.from_numpy(np.asarray(prepared, dtype=np.float32))[None, :].to(self.device)
        with self.torch.no_grad():
            output = self.logits_tensor(tensor)
        return logits_to_spoof_probability(output.detach().float().cpu().numpy(), self.spoof_index)

    def score_signal(self, signal, sr):
        return self.score_prepared(self.prepare(signal))

    def score_file(self, path):
        signal, sr = load_audio(path, self.sample_rate)
        return self.score_signal(signal, sr)

    def metadata(self):
        return {
            "name": self.name,
            "version": self.version,
            "repo_path": self.repo_path,
            "checkpoint": self.checkpoint,
            "spoof_index": self.spoof_index,
            "trainable": False,
        }
