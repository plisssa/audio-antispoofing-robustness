from pathlib import Path

import pandas as pd

COLUMNS = ["file_id", "path", "label", "split", "dataset", "source"]
LABEL_TO_INT = {"bona_fide": 0, "spoof": 1}


def label_to_int(label):
    return LABEL_TO_INT[label]


def write_manifest(path, rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def read_manifest(path):
    return pd.read_csv(path)


def split_view(manifest, split):
    return manifest[manifest["split"] == split].reset_index(drop=True)
