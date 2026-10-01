import argparse
import csv
import os
import random
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

REPO = "jungjee/asvspoof5"
EVAL_TARS = [f"flac_E_a{c}.tar" for c in "abcdefghij"]


def log(message):
    print(f"[{time.strftime('%d.%m %H:%M:%S')}] {message}", flush=True)


def select_subset(protocol, per_attack, bona_fide, seed, codec):
    groups = defaultdict(list)
    with open(protocol) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9:
                continue
            if codec is not None and parts[3] != codec:
                continue
            groups[parts[7]].append(parts[1])
    rng = random.Random(seed)
    chosen = {}
    for label in sorted(groups):
        names = sorted(groups[label])
        rng.shuffle(names)
        quota = bona_fide if label == "bonafide" else per_attack
        for name in names[:quota]:
            chosen[name] = label
    return chosen


def download_with_retry(name, tar_dir, attempts, pause):
    from huggingface_hub import hf_hub_download

    for attempt in range(1, attempts + 1):
        try:
            return hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(tar_dir))
        except Exception as error:
            log(f"{name}: попытка {attempt}/{attempts} оборвалась: {type(error).__name__}: {str(error)[:120]}")
            if attempt == attempts:
                raise
            time.sleep(pause)


def process_tar(tar_path, wanted, audio_dir):
    extracted = 0
    with tarfile.open(tar_path, "r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            if stem not in wanted:
                continue
            target = audio_dir / f"{stem}.flac"
            if target.exists() and target.stat().st_size == member.size:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            tmp = target.with_suffix(".flac.part")
            with open(tmp, "wb") as out:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            tmp.rename(target)
            extracted += 1
    return extracted


def write_manifest(path, chosen, audio_dir, project_root):
    rows = []
    for name, label in sorted(chosen.items()):
        audio = audio_dir / f"{name}.flac"
        if not audio.exists():
            continue
        rows.append({
            "file_id": name,
            "path": os.path.relpath(audio, project_root),
            "label": "bona_fide" if label == "bonafide" else "spoof",
            "split": "eval",
            "dataset": "asvspoof5",
            "source": label,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_id", "path", "label", "split", "dataset", "source"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--per-attack", type=int, default=1500)
    parser.add_argument("--bona-fide", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--codec", default="-")
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--pause", type=int, default=60)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    project = Path(args.project)
    protocol = workspace / "ASVspoof5.eval.track_1.tsv"
    tar_dir = workspace / "tars"
    audio_dir = project / "data" / "raw" / "asvspoof5" / "flac_E_eval"
    done_file = workspace / "tars.done"
    tar_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    done_file.touch()

    chosen = select_subset(protocol, args.per_attack, args.bona_fide, args.seed, None if args.codec == "any" else args.codec)
    counts = defaultdict(int)
    for label in chosen.values():
        counts[label] += 1
    log(f"подвыборка: {len(chosen)} файлов, кодек={args.codec}, по меткам: {dict(sorted(counts.items()))}")
    wanted = set(chosen)

    finished = set(done_file.read_text().split())
    for name in EVAL_TARS:
        if name in finished:
            log(f"{name}: уже обработан, пропуск")
            continue
        log(f"{name}: загрузка")
        started = time.time()
        local = download_with_retry(name, tar_dir, args.attempts, args.pause)
        size = os.path.getsize(local) / 1e9
        log(f"{name}: скачан {size:.2f} ГБ за {time.time() - started:.0f} с, извлекаю")
        extracted = process_tar(local, wanted, audio_dir)
        present = sum(1 for n in wanted if (audio_dir / f"{n}.flac").exists())
        log(f"{name}: извлечено {extracted}, всего на месте {present}/{len(wanted)}")
        os.remove(local)
        with open(done_file, "a") as handle:
            handle.write(name + "\n")
        log(f"{name}: архив удалён")

    rows = write_manifest(project / "data" / "asvspoof5" / "manifest.csv", chosen, audio_dir, project)
    missing = len(chosen) - len(rows)
    log(f"манифест: {len(rows)} строк, не найдено {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
