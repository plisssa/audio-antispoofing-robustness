from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import COLUMNS, write_manifest

ASV2019_PROTOCOL = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def build_asvspoof2019_la(root, out, split="eval", train_ratio=0.0, **_):
    root = Path(root)
    protocol = root / "ASVspoof2019_LA_cm_protocols" / ASV2019_PROTOCOL[split]
    audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
    rows = []
    for line in protocol.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        file_id = parts[1]
        label = "bona_fide" if parts[4] == "bonafide" else "spoof"
        rows.append({
            "file_id": file_id,
            "path": str(audio_dir / f"{file_id}.flac"),
            "label": label,
            "split": split,
            "dataset": "asvspoof2019_la",
            "source": parts[3],
        })
    return write_manifest(out, rows)


def _label_token(parts):
    for token in parts:
        if token == "bonafide":
            return "bona_fide"
        if token == "spoof":
            return "spoof"
    return None


def _build_asvspoof2021(root, out, track, dataset, keys, audio_subdir):
    root = Path(root)
    metadata = root / keys
    audio_dir = root / audio_subdir / "flac"
    rows = []
    for line in metadata.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        label = _label_token(parts)
        if label is None:
            continue
        file_id = parts[1]
        source = parts[4] if len(parts) > 4 else "unknown"
        rows.append({
            "file_id": file_id,
            "path": str(audio_dir / f"{file_id}.flac"),
            "label": label,
            "split": "eval",
            "dataset": dataset,
            "source": source,
        })
    return write_manifest(out, rows)


def build_asvspoof2021_la(root, out, split="eval", train_ratio=0.0, **_):
    return _build_asvspoof2021(root, out, "LA", "asvspoof2021_la", "keys/LA/CM/trial_metadata.txt", "ASVspoof2021_LA_eval")


def build_asvspoof2021_df(root, out, split="eval", train_ratio=0.0, **_):
    return _build_asvspoof2021(root, out, "DF", "asvspoof2021_df", "keys/DF/CM/trial_metadata.txt", "ASVspoof2021_DF_eval")


def build_in_the_wild(root, out, split="eval", train_ratio=0.0, seed=1337, **_):
    root = Path(root)
    meta = pd.read_csv(root / "meta.csv")
    rows = []
    for record in meta.itertuples():
        label = "bona_fide" if str(record.label).lower().startswith("bona") else "spoof"
        rows.append({
            "file_id": Path(record.file).stem,
            "path": str(root / record.file),
            "label": label,
            "split": split,
            "dataset": "in_the_wild",
            "source": getattr(record, "speaker", "unknown"),
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if train_ratio > 0.0:
        rng = np.random.default_rng(seed)
        for _, group in frame.groupby("label"):
            index = group.index.to_numpy().copy()
            rng.shuffle(index)
            cut = int(len(index) * train_ratio)
            frame.loc[index[:cut], "split"] = "train"
            frame.loc[index[cut:], "split"] = "eval"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


BUILDERS = {
    "asvspoof2019_la": build_asvspoof2019_la,
    "asvspoof2021_la": build_asvspoof2021_la,
    "asvspoof2021_df": build_asvspoof2021_df,
    "in_the_wild": build_in_the_wild,
}
