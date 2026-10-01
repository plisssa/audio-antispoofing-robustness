import re
import sys
import types

import torch
from torch import nn

HF_MODEL_ID = "facebook/wav2vec2-xls-r-300m"


def _build_hf():
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    config = Wav2Vec2Config.from_pretrained(HF_MODEL_ID)
    return Wav2Vec2Model(config)


class HFXLSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.hf = _build_hf()

    def _last_hidden(self, source):
        if source.dim() == 3:
            source = source[:, :, 0]
        return self.hf(source).last_hidden_state

    def forward(self, source, mask=False, features_only=True, **kwargs):
        return {"x": self._last_hidden(source), "layer_results": None, "padding_mask": None}

    def extract_features(self, source, padding_mask=None, mask=False, **kwargs):
        return {"x": self._last_hidden(source), "padding_mask": None}


def install_fairseq_shim():
    existing = sys.modules.get("fairseq")
    if existing is not None and getattr(existing, "_adfd_shim", False):
        return
    fairseq = types.ModuleType("fairseq")
    fairseq._adfd_shim = True
    checkpoint_utils = types.ModuleType("fairseq.checkpoint_utils")

    def load_model_ensemble_and_task(paths, *args, **kwargs):
        return [HFXLSR()], types.SimpleNamespace(), types.SimpleNamespace()

    checkpoint_utils.load_model_ensemble_and_task = load_model_ensemble_and_task
    fairseq.checkpoint_utils = checkpoint_utils
    sys.modules["fairseq"] = fairseq
    sys.modules["fairseq.checkpoint_utils"] = checkpoint_utils


def _fairseq_to_hf_key(name, pos_conv_parametrized):
    name = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.0\.", r"feature_extractor.conv_layers.\1.conv.", name)
    name = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.", r"feature_extractor.conv_layers.\1.layer_norm.", name)
    name = re.sub(r"^feature_extractor\.conv_layers\.(\d+)\.2\.", r"feature_extractor.conv_layers.\1.layer_norm.", name)
    name = re.sub(r"^mask_emb$", "masked_spec_embed", name)
    name = re.sub(r"^post_extract_proj\.", "feature_projection.projection.", name)
    name = re.sub(r"^layer_norm\.", "feature_projection.layer_norm.", name)
    name = re.sub(r"^encoder\.layers\.(\d+)\.self_attn\.(k|q|v|out)_proj\.", r"encoder.layers.\1.attention.\2_proj.", name)
    name = re.sub(r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"encoder.layers.\1.layer_norm.", name)
    name = re.sub(r"^encoder\.layers\.(\d+)\.fc1\.", r"encoder.layers.\1.feed_forward.intermediate_dense.", name)
    name = re.sub(r"^encoder\.layers\.(\d+)\.fc2\.", r"encoder.layers.\1.feed_forward.output_dense.", name)
    if name.startswith("encoder.pos_conv.0."):
        tail = name[len("encoder.pos_conv.0."):]
        if pos_conv_parametrized:
            tail = {"weight_g": "parametrizations.weight.original0", "weight_v": "parametrizations.weight.original1"}.get(tail, tail)
        name = "encoder.pos_conv_embed.conv." + tail
    return name


def remap_checkpoint(state, model):
    marker = "feature_extractor.conv_layers.0.0.weight"
    prefix = None
    for key in state:
        if key.endswith(marker):
            prefix = key[: -len(marker)]
            break
    if prefix is None:
        return state
    target_keys = set(model.state_dict().keys())
    pos_conv_parametrized = any("pos_conv_embed.conv.parametrizations.weight.original0" in k for k in target_keys)
    remapped = {}
    for key, value in state.items():
        if key.startswith(prefix) and not key.startswith(prefix + "hf."):
            hf_key = _fairseq_to_hf_key(key[len(prefix):], pos_conv_parametrized)
            remapped[prefix + "hf." + hf_key] = value
        else:
            remapped[key] = value
    matched = sum(1 for k in remapped if k in target_keys)
    frontend_total = sum(1 for k in target_keys if k.startswith(prefix + "hf."))
    frontend_matched = sum(1 for k in remapped if k in target_keys and k.startswith(prefix + "hf."))
    print(f"xlsr_hf remap: frontend keys {frontend_matched}/{frontend_total} matched; total loaded {matched}/{len(target_keys)}", file=sys.stderr)
    return remapped
