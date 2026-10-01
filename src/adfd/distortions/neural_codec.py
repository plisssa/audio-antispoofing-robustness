import numpy as np

from ..audio.io import resample
from .base import register

try:
    import torch
    from encodec import EncodecModel

    ENCODEC_AVAILABLE = True
except Exception:
    ENCODEC_AVAILABLE = False

MODEL_SR = 24000
_MODELS = {}


def _model(bandwidth):
    if not ENCODEC_AVAILABLE:
        raise RuntimeError("encodec not installed; neural_codec distortion requires the encodec package")
    if "24k" not in _MODELS:
        model = EncodecModel.encodec_model_24khz()
        model.eval()
        _MODELS["24k"] = model
    model = _MODELS["24k"]
    model.set_target_bandwidth(float(bandwidth))
    return model


@register("neural_codec")
def encodec_roundtrip(signal, sr, level):
    model = _model(level)
    wide = resample(np.asarray(signal, dtype=np.float32), sr, MODEL_SR)
    tensor = torch.tensor(wide, dtype=torch.float32)[None, None, :]
    with torch.no_grad():
        encoded = model.encode(tensor)
        decoded = model.decode(encoded)
    out = decoded[0, 0].detach().cpu().numpy()
    out = resample(out.astype(np.float32), MODEL_SR, sr)
    if len(out) >= len(signal):
        out = out[: len(signal)]
    else:
        out = np.pad(out, (0, len(signal) - len(out)))
    return out.astype(np.float32), {"bandwidth_kbps": float(level), "codec": "encodec_24k", "applied": True}
