from pathlib import Path

import pandas as pd


def load_partial(path):
    target = Path(path)
    if target.exists() and target.stat().st_size > 0:
        return pd.read_csv(target)
    return None


def done_ids(path, column="file_id"):
    frame = load_partial(path)
    return set(frame[column]) if frame is not None else set()


def append_rows(path, rows):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not Path(path).exists(), index=False)
