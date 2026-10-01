import pandas as pd

from ..audio.features import describe
from ..audio.io import load_audio

COLUMNS = [
    "file_id",
    "dataset",
    "label",
    "model_name",
    "model_version",
    "clean_score",
    "distorted_score",
    "prediction_clean",
    "prediction_distorted",
    "error_type",
    "flipped",
    "distortion_type",
    "family",
    "distortion_strength",
    "severity",
    "duration_s",
    "rms",
    "silence_ratio",
    "clipping_ratio",
    "spectral_centroid",
    "spectral_bandwidth",
    "snr_estimate_db",
]


def file_features(manifest, sample_rate=16000):
    rows = []
    for row in manifest.itertuples():
        signal, sr = load_audio(row.path, sample_rate)
        features = describe(signal, sr)
        features["file_id"] = row.file_id
        rows.append(features)
    return pd.DataFrame(rows)


def build_error_table(sweep, features, detector_meta):
    table = sweep.merge(features, on="file_id", how="left")
    table["model_name"] = detector_meta.get("name")
    table["model_version"] = detector_meta.get("version")
    table = table.rename(columns={"strength_value": "distortion_strength"})
    return table[COLUMNS]
