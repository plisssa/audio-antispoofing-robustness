import sys
from pathlib import Path

from .base import register_detector
from .raw_torch import RawWaveformTorchDetector


@register_detector("aasist3")
class AASIST3Detector(RawWaveformTorchDetector):
    name = "aasist3_kan"
    version = "MTUCI/AASIST3"

    def __init__(self, sample_rate=16000, repo_path=None, repo="MTUCI/AASIST3", w2v_cache_dir=None, device=None, spoof_index=1, crop=64600):
        super().__init__(sample_rate=sample_rate, repo_path=repo_path, checkpoint=repo, device=device, spoof_index=spoof_index, crop=crop)
        self.repo = repo
        self.w2v_cache_dir = w2v_cache_dir

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
        from model import aasist3

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {}
        if self.w2v_cache_dir:
            kwargs["w2v_cache_dir"] = self.w2v_cache_dir
        self.model = aasist3.from_pretrained(self.repo, **kwargs).to(self.device).eval()
        return self

    def metadata(self):
        data = super().metadata()
        data.update({"name": self.name, "version": self.version, "repo": self.repo, "checkpoint": self.repo, "crop": self.crop})
        return data
