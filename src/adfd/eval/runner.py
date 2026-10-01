import pandas as pd

from ..datasets.manifest import label_to_int, split_view
from ..repro.checkpoint import append_rows, done_ids, load_partial


def fit_detector(detector, manifest, train_split="train"):
    if not getattr(detector, "trainable", False):
        return detector
    train = split_view(manifest, train_split)
    items = [(row.path, label_to_int(row.label)) for row in train.itertuples()]
    detector.fit(items)
    return detector


def score_manifest(detector, manifest, partial_path=None):
    done = done_ids(partial_path) if partial_path else set()
    collected = []
    previous = load_partial(partial_path) if partial_path else None
    if previous is not None:
        collected.append(previous)
    for row in manifest.itertuples():
        if row.file_id in done:
            continue
        record = {
            "file_id": row.file_id,
            "path": row.path,
            "label": row.label,
            "split": row.split,
            "dataset": row.dataset,
            "source": getattr(row, "source", None),
            "score": detector.score_file(row.path),
        }
        if partial_path:
            append_rows(partial_path, [record])
        collected.append(pd.DataFrame([record]))
    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()


def predictions_from_scores(scores, threshold):
    frame = scores.copy()
    frame["target"] = frame["label"].map(label_to_int)
    frame["prediction"] = (frame["score"] >= threshold).astype(int)
    frame["correct"] = (frame["prediction"] == frame["target"]).astype(int)
    return frame
