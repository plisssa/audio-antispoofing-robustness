import os
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from .base import register_detector
from .raw_torch import RawWaveformTorchDetector


class SSLXLSRDetector(RawWaveformTorchDetector):
    module_name = "model"
    class_name = "Model"

    def __init__(self, *args, module_name=None, class_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_name = module_name or type(self).module_name
        self.class_name = class_name or type(self).class_name

    def _clear_modules(self):
        roots = {"model", "mamba_blocks", self.module_name, self.module_name.split(".")[0]}
        for name in list(sys.modules):
            if name in roots or any(name.startswith(f"{root}.") for root in roots):
                sys.modules.pop(name, None)

    def build_model(self, torch):
        from . import xlsr_hf

        xlsr_hf.install_fairseq_shim()
        self._clear_modules()
        repo = str(Path(self.repo_path).resolve()) if self.repo_path else None
        cwd = os.getcwd()
        try:
            if repo:
                if repo in sys.path:
                    sys.path.remove(repo)
                sys.path.insert(0, repo)
                os.chdir(repo)
            factory = getattr(import_module(self.module_name), self.class_name)
            return factory(SimpleNamespace(**self.extra), self.device)
        finally:
            os.chdir(cwd)

    def remap_state(self, state, model):
        from . import xlsr_hf

        return xlsr_hf.remap_checkpoint(state, model)


@register_detector("ssl_aasist")
class SSLAASISTDetector(SSLXLSRDetector):
    name = "ssl_aasist"
    version = "TakHemlata/SSL_Anti-spoofing"


@register_detector("xlsr_mamba")
class XLSRMambaDetector(SSLXLSRDetector):
    name = "xlsr_mamba"
    version = "swagshaw/XLSR-Mamba"


@register_detector("nes2net_v2")
class Nes2NetV2Detector(SSLXLSRDetector):
    name = "nes2net_v2"
    version = "Liu-Tianchi/Nes2Net_ASVspoof_ITW"
    module_name = "model_scripts.wav2vec2_Nes2Net_X"
    class_name = "wav2vec2_Nes2Net_no_Res_w_allT"
