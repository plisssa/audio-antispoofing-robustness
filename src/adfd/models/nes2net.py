import sys
from pathlib import Path
from importlib import import_module
from types import SimpleNamespace

import numpy as np

from ..audio.io import load_audio
from .base import Detector, register_detector

NES2NET_CLASSES = {
    "WavLM_Nes2Net": ("models.WavLM_Nes2Net", "WavLM_Nes2Net_noRes"),
    "WavLM_Nes2Net_X": ("models.WavLM_Nes2Net_X", "WavLM_Nes2Net_noRes_w_allT"),
    "WavLM_Nes2Net_X_SeLU": ("models.WavLM_Nes2Net_X_SeLU", "WavLM_Nes2Net_SE_cat_SeLU"),
}


@register_detector("nes2net")
class Nes2NetDetector(Detector):
    name = "nes2net"
    version = "Liu-Tianchi/Nes2Net"
    trainable = False
    differentiable = True

    def __init__(self, sample_rate=16000, repo_path=None, checkpoint=None, model_name="WavLM_Nes2Net_X", device=None, crop=64000, extra=None):
        self.sample_rate = sample_rate
        self.repo_path = repo_path
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.device = device
        self.crop = crop
        self.extra = extra or {}
        self.model = None
        self.torch = None

    def available(self):
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return bool(self.repo_path and self.checkpoint and self.model_name in NES2NET_CLASSES)

    def load(self):
        import torch

        if self.repo_path:
            repo = str(Path(self.repo_path).resolve())
            if repo in sys.path:
                sys.path.remove(repo)
            sys.path.insert(0, repo)
        sys.modules.pop("models", None)
        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        module_name, class_name = NES2NET_CLASSES[self.model_name]
        builder = getattr(import_module(module_name), class_name)
        model = builder(SimpleNamespace(**self.extra), self.device)
        saved = torch.load(self.checkpoint, map_location=self.device)
        if isinstance(saved, dict) and "state_dict" in saved:
            saved = saved["state_dict"]
        target = model.state_dict()
        target.update({key: value for key, value in saved.items() if key in target})
        model.load_state_dict(target)
        self.model = model.to(self.device).eval()
        unfrozen = self.enable_input_gradients()
        if unfrozen:
            print(f"nes2net: feature_grad_mult 0 -> 1.0 в {unfrozen} модулях (нужно для градиентных атак; forward не меняется)", file=sys.stderr)
        return self

    def prepare(self, signal):
        if len(signal) >= self.crop:
            return signal[: self.crop]
        return np.pad(signal, (0, self.crop - len(signal)))

    def enable_input_gradients(self):
        changed = 0
        for module in self.model.modules():
            if getattr(module, "feature_grad_mult", None) == 0:
                module.feature_grad_mult = 1.0
                changed += 1
            config = getattr(module, "cfg", None)
            if config is not None and getattr(config, "feature_grad_mult", None) == 0:
                config.feature_grad_mult = 1.0
                changed += 1
        return changed

    def logits_tensor(self, x):
        output = self.model(x)
        value = output.reshape(output.shape[0], -1)[:, :1]
        return self.torch.cat([self.torch.zeros_like(value), -value], dim=-1)

    def score_prepared(self, prepared):
        tensor = self.torch.tensor(np.asarray(prepared, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            prediction = self.model(tensor)
        value = float(prediction.reshape(-1)[0].detach().cpu())
        return float(1.0 / (1.0 + np.exp(value)))

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
            "model_name": self.model_name,
            "trainable": False,
        }
