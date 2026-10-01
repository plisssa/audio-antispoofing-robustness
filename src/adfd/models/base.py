import numpy as np

REGISTRY = {}


def logits_to_spoof_probability(output, spoof_index):
    array = np.asarray(output, dtype=float).reshape(-1)
    if array.size == 1:
        return float(1.0 / (1.0 + np.exp(-array[0])))
    shifted = array - array.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    return float(probabilities[spoof_index])


def register_detector(name):
    def wrap(factory):
        REGISTRY[name] = factory
        return factory

    return wrap


def build_detector(name, **kwargs):
    return REGISTRY[name](**kwargs)


def available_detectors():
    return sorted(REGISTRY)


class Detector:
    name = "detector"
    version = "0.0.0"
    trainable = False

    def available(self):
        return True

    def load(self):
        return self

    def fit(self, items):
        return self

    def score_file(self, path):
        raise NotImplementedError

    def raw_gradient(self, gradient):
        return gradient

    def to_raw(self, prepared):
        return prepared

    def metadata(self):
        return {"name": self.name, "version": self.version, "trainable": self.trainable}
