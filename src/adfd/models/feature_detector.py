import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..audio.io import load_audio
from .base import Detector


class FeatureLogRegDetector(Detector):
    trainable = True

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.scaler = None
        self.model = None

    def embed(self, signal, sr):
        raise NotImplementedError

    def fit(self, items):
        features = []
        targets = []
        for path, label in items:
            signal, sr = load_audio(path, self.sample_rate)
            features.append(self.embed(signal, sr))
            targets.append(label)
        matrix = np.asarray(features)
        self.scaler = StandardScaler().fit(matrix)
        self.model = LogisticRegression(max_iter=1000).fit(self.scaler.transform(matrix), np.asarray(targets))
        return self

    def score_signal(self, signal, sr):
        vector = self.scaler.transform(self.embed(signal, sr)[None, :])
        return float(self.model.predict_proba(vector)[0, 1])

    def score_file(self, path):
        signal, sr = load_audio(path, self.sample_rate)
        return self.score_signal(signal, sr)

    def metadata(self):
        return {"name": self.name, "version": self.version, "checkpoint": "fitted_in_run", "trainable": True}
