import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--out", default="results/wave2")
args = parser.parse_args()

root = Path(args.root)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def run_dirs():
    for group in sorted(root.glob("runs*")):
        for run in sorted((group / "pipeline").glob("*")):
            if run.is_dir():
                yield group.name, run


def detector_name(meta):
    detector = meta.get("detector")
    if isinstance(detector, dict):
        return detector.get("name")
    return detector


def read_meta(run):
    path = run / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def tagged(frame, group, run, meta):
    frame.insert(0, "dataset", meta.get("dataset"))
    frame.insert(0, "run", run.name)
    frame.insert(0, "run_group", group)
    return frame


def eer(scores, labels):
    order = np.argsort(-scores)
    labels = labels[order]
    positives = max(int(labels.sum()), 1)
    negatives = max(int(len(labels) - labels.sum()), 1)
    tpr = np.concatenate([[0.0], np.cumsum(labels) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(1 - labels) / negatives])
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[index] + fpr[index]) / 2.0)


tables = {name: [] for name in ["attack", "transfer", "defence", "sweep", "clean", "alignment", "gradcheck", "perceptual", "asv5_per_attack"]}
sources = {
    "attack": "attack_aggregate.csv",
    "transfer": "transfer_aggregate.csv",
    "defence": "defence_aggregate.csv",
    "sweep": "aggregate.csv",
    "alignment": "alignment_pairs.csv",
    "gradcheck": "gradcheck_summary.csv",
    "perceptual": "perceptual_summary.csv",
}

for group, run in run_dirs():
    meta = read_meta(run)
    model = detector_name(meta)
    for name, filename in sources.items():
        path = run / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if name == "transfer":
            frame.insert(0, "source_model", model)
        tables[name].append(tagged(frame, group, run, meta))
    for filename in ("clean_metrics.json", "metrics.json"):
        path = run / filename
        if path.exists():
            values = json.loads(path.read_text())
            row = {"run_group": group, "run": run.name, "dataset": meta.get("dataset"), "model_name": model}
            row.update({key: value for key, value in values.items() if not isinstance(value, (dict, list))})
            tables["clean"].append(pd.DataFrame([row]))
    scores_path = run / "scores.csv"
    if scores_path.exists() and meta.get("dataset") == "asvspoof5":
        scores = pd.read_csv(scores_path)
        scores = scores[scores["split"] == "eval"]
        labels = (scores["label"] == "spoof").to_numpy(int)
        bona = scores[scores["label"] == "bona_fide"]
        rows = [{"attack": "all", "n_spoof": int(labels.sum()), "eer": eer(scores["score"].to_numpy(float), labels)}]
        for attack, spoof in scores[scores["label"] == "spoof"].groupby("source"):
            subset = pd.concat([bona, spoof])
            rows.append({"attack": attack, "n_spoof": len(spoof), "eer": eer(subset["score"].to_numpy(float), (subset["label"] == "spoof").to_numpy(int))})
        frame = pd.DataFrame(rows)
        frame.insert(0, "model_name", model)
        tables["asv5_per_attack"].append(tagged(frame, group, run, meta))

for name, frames in tables.items():
    if frames:
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows")
