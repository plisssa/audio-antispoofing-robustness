import numpy as np

from ..audio.io import load_audio
from .base import Detector, logits_to_spoof_probability, register_detector
from .xlsr_hf import HF_MODEL_ID


@register_detector("xlsr_linear")
class XLSRLinearDetector(Detector):
    name = "xlsr_linear"
    version = "0.1.0"
    trainable = True
    differentiable = True
    spoof_index = 1
    default_model_id = HF_MODEL_ID

    def __init__(self, sample_rate=16000, crop=64600, device=None, epochs=40, batch_size=32,
                 feature_batch=8, lr=1e-3, seed=1337, model_id=None, head="linear", hidden=256,
                 label=None, **_):
        self.name = label or type(self).name
        self.head_kind = head
        self.hidden = hidden
        self.sample_rate = sample_rate
        self.crop = crop
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_batch = feature_batch
        self.lr = lr
        self.seed = seed
        self.model_id = model_id or type(self).default_model_id
        self.frontend = None
        self.head = None
        self.torch = None

    def available(self):
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    def load(self):
        import torch
        from torch import nn
        from transformers import AutoModel

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        frontend = AutoModel.from_pretrained(self.model_id)
        frontend.config.feature_grad_mult = 1.0
        for parameter in frontend.parameters():
            parameter.requires_grad_(False)
        self.frontend = frontend.to(self.device).eval()
        torch.manual_seed(self.seed)
        width = frontend.config.hidden_size
        if self.head_kind == "mlp":
            head = nn.Sequential(nn.Linear(width, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 2))
        else:
            head = nn.Linear(width, 2)
        self.head = head.to(self.device)
        return self

    def prepare(self, signal):
        if len(signal) >= self.crop:
            return signal[: self.crop]
        repeats = int(np.ceil(self.crop / len(signal)))
        return np.tile(signal, repeats)[: self.crop]

    def embed(self, x):
        return self.frontend(x).last_hidden_state.mean(dim=1)

    def logits_tensor(self, x):
        return self.head(self.embed(x))

    def fit(self, items):
        if self.head is None:
            self.load()
        torch = self.torch
        signals = []
        labels = []
        for path, label in items:
            signal, _ = load_audio(path, self.sample_rate)
            signals.append(self.prepare(signal))
            labels.append(label)
        features = []
        for start in range(0, len(signals), self.feature_batch):
            chunk = np.stack(signals[start: start + self.feature_batch])
            tensor = torch.from_numpy(chunk.astype(np.float32)).to(self.device)
            with torch.no_grad():
                features.append(self.embed(tensor).float().cpu())
        inputs = torch.cat(features).to(self.device)
        targets = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(self.seed)
        self.head.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(targets), generator=generator)
            for start in range(0, len(targets), self.batch_size):
                index = order[start: start + self.batch_size].to(self.device)
                optimizer.zero_grad()
                loss_fn(self.head(inputs[index]), targets[index]).backward()
                optimizer.step()
        self.head.eval()
        return self

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
            "model_id": self.model_id,
            "frontend": "frozen",
            "head": self.head_kind,
            "epochs": self.epochs,
            "spoof_index": self.spoof_index,
            "differentiable": True,
            "trainable": True,
        }


@register_detector("wavlm_linear")
class WavLMLinearDetector(XLSRLinearDetector):
    name = "wavlm_linear"
    version = "0.1.0"
    default_model_id = "microsoft/wavlm-large"


PROBES = ("hubert_linear", "hubert_mlp", "w2v2lv60_linear", "w2v2lv60_mlp", "wavlmbase_linear", "wavlmbase_mlp")
SEEDED = tuple(f"{base}_s{seed}" for base in ("xlsr_linear", "wavlm_linear") for seed in (2, 3, 4))

for probe in PROBES + SEEDED:
    register_detector(probe)(XLSRLinearDetector)
