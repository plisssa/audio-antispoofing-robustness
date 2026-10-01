from pathlib import Path

import soundfile as sf


def dataset_report(manifest, min_duration=0.3):
    missing = 0
    unreadable = 0
    short = 0
    durations = []
    for row in manifest.itertuples():
        path = Path(row.path)
        if not path.exists():
            missing += 1
            continue
        try:
            info = sf.info(str(path))
            duration = info.frames / info.samplerate
        except Exception:
            unreadable += 1
            continue
        durations.append(duration)
        if duration < min_duration:
            short += 1
    return {
        "total": int(len(manifest)),
        "by_label": {str(key): int(value) for key, value in manifest["label"].value_counts().items()},
        "by_split": {str(key): int(value) for key, value in manifest["split"].value_counts().items()},
        "missing_files": missing,
        "unreadable_files": unreadable,
        "short_files": short,
        "total_hours": round(sum(durations) / 3600.0, 3) if durations else 0.0,
        "duration_min": round(min(durations), 3) if durations else None,
        "duration_mean": round(sum(durations) / len(durations), 3) if durations else None,
        "duration_max": round(max(durations), 3) if durations else None,
    }
