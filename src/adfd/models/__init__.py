from . import (
    aasist,
    aasist3,
    melspec,
    nes2net,
    rawnet2,
    reference,
    spectra_aasist3,
    ssl_xlsr,
    torch_reference,
    xlsr_linear,
)
from .base import (
    Detector,
    available_detectors,
    build_detector,
    logits_to_spoof_probability,
    register_detector,
)
