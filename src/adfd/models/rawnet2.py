import sys
from importlib import import_module

import yaml

from .base import register_detector
from .raw_torch import RawWaveformTorchDetector


@register_detector("rawnet2")
class RawNet2Detector(RawWaveformTorchDetector):
    name = "rawnet2"
    version = "asvspoof2021/Baseline-RawNet2"

    def build_model(self, torch):
        sys.modules.pop("model", None)
        with open(self.config) as handle:
            spec = yaml.safe_load(handle)
        return import_module("model").RawNet(spec["model"], self.device)
