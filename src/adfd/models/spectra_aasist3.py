import sys
from pathlib import Path

from .base import register_detector
from .raw_torch import RawWaveformTorchDetector


@register_detector("spectra_aasist3")
class SpectraAASIST3Detector(RawWaveformTorchDetector):
    name = "spectra_aasist3"
    version = "lab260/Spectra-AASIST3"

    def __init__(self, sample_rate=16000, repo_path="third_party/Spectra-AASIST3", repo=None, device=None, spoof_index=1, crop=64600, preemphasis=0.97):
        super().__init__(sample_rate=sample_rate, repo_path=repo_path, checkpoint=repo or repo_path, device=device, spoof_index=spoof_index, crop=crop)
        self.repo = repo
        self.preemphasis = preemphasis

    def available(self):
        try:
            import torch  # noqa: F401
        except Exception:
            return False
        return bool(self.repo_path)

    def load(self):
        import torch

        if self.repo_path:
            repo = str(Path(self.repo_path).resolve())
            if repo in sys.path:
                sys.path.remove(repo)
            sys.path.insert(0, repo)
        sys.modules.pop("model", None)
        from model import SpectraAASIST3

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpectraAASIST3.from_pretrained(self.repo or self.repo_path).to(self.device).eval()
        return self

    def logits_tensor(self, x):
        if self.preemphasis:
            x = self.torch.cat([x[:, :1], x[:, 1:] - self.preemphasis * x[:, :-1]], dim=1)
        return super().logits_tensor(x)

    def metadata(self):
        data = super().metadata()
        data.update({"name": self.name, "version": self.version, "checkpoint": self.repo or self.repo_path, "crop": self.crop})
        return data
