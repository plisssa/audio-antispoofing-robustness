import json
import sys
from importlib import import_module

from .base import register_detector
from .raw_torch import RawWaveformTorchDetector


@register_detector("aasist")
class AASISTDetector(RawWaveformTorchDetector):
    name = "aasist"
    version = "clovaai/aasist"

    def build_model(self, torch):
        sys.modules.pop("models", None)
        sys.modules.pop("models.AASIST", None)
        with open(self.config) as handle:
            args = json.load(handle)["model_config"]
        return import_module("models.AASIST").Model(args)


@register_detector("aasist_asv5")
class AASISTASV5Detector(AASISTDetector):
    name = "aasist_asv5"
    version = "asvspoof-challenge/asvspoof5 Baseline-AASIST (B02)"
