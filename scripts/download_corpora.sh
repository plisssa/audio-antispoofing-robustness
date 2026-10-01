#!/usr/bin/env bash
set -uo pipefail

RAW="${1:-data/raw/corpora}"
DEST="${2:-data/corpora}"
mkdir -p "$RAW" "$DEST"

warn() { echo "WARN: $*"; }

fetch() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1 && curl -fSL --connect-timeout 60 --speed-time 120 --speed-limit 1024 --retry 10 --retry-delay 5 -C - "$url" -o "$out"; then
        return 0
    fi
    if command -v wget >/dev/null 2>&1 && wget -c --timeout=60 --tries=3 -O "$out" "$url"; then
        return 0
    fi
    return 1
}

valid_archive() {
    case "$1" in
        *.zip) unzip -l "$1" >/dev/null 2>&1 ;;
        *.tar.gz|*.tgz) gzip -t "$1" >/dev/null 2>&1 ;;
        *) [ -s "$1" ] ;;
    esac
}

get() {
    local out="$1"; shift
    if [ -f "$out" ]; then
        valid_archive "$out" && return 0
        echo "  removing corrupt $out"
        rm -f "$out"
    fi
    for url in "$@"; do
        echo "  fetching $url"
        if fetch "$url" "$out" && valid_archive "$out"; then
            return 0
        fi
        rm -f "$out"
    done
    return 1
}

MUSAN_URLS=("${MUSAN_URL:-https://www.openslr.org/resources/17/musan.tar.gz}" "https://openslr.elda.org/resources/17/musan.tar.gz" "https://us.openslr.org/resources/17/musan.tar.gz")
RIRS_URLS=("${RIRS_URL:-https://www.openslr.org/resources/28/rirs_noises.zip}" "https://openslr.elda.org/resources/28/rirs_noises.zip" "https://us.openslr.org/resources/28/rirs_noises.zip")
ESC50_URLS=("${ESC50_URL:-https://github.com/karoldvl/ESC-50/archive/master.zip}" "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip")

if [ ! -e "$DEST/musan_noise" ]; then
    echo "MUSAN (~11G)"
    if get "$RAW/musan.tar.gz" "${MUSAN_URLS[@]}"; then
        tar -xzf "$RAW/musan.tar.gz" -C "$RAW" && ln -sfn "$(cd "$RAW/musan/noise" && pwd)" "$DEST/musan_noise"
    else
        warn "MUSAN download failed (real_noise will skip)"
    fi
fi

if [ ! -e "$DEST/rirs" ]; then
    echo "RIRS_NOISES (~5G)"
    if get "$RAW/rirs_noises.zip" "${RIRS_URLS[@]}"; then
        unzip -q -o "$RAW/rirs_noises.zip" -d "$RAW" && ln -sfn "$(cd "$RAW/RIRS_NOISES/real_rirs_isotropic_noises" && pwd)" "$DEST/rirs"
    else
        warn "RIRS download failed (real_rir will skip)"
    fi
fi

if [ ! -e "$DEST/esc50_birds" ] || [ -z "$(ls -A "$DEST/esc50_birds" 2>/dev/null)" ]; then
    echo "ESC-50"
    if get "$RAW/esc50.zip" "${ESC50_URLS[@]}"; then
        unzip -q -o "$RAW/esc50.zip" -d "$RAW"
        ESC_ROOT="$(find "$RAW" -maxdepth 1 -type d -name 'ESC-50*' | head -1)"
        python - "$ESC_ROOT" "$DEST/esc50_birds" <<'PY'
import shutil
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta = pd.read_csv(root / "meta" / "esc50.csv")
birds = meta[meta["category"] == "chirping_birds"]["filename"]
for name in birds:
    shutil.copy(root / "audio" / name, out / name)
print(f"esc50 birds copied: {len(birds)} -> {out}")
PY
    else
        warn "ESC-50 download failed (real_bird will skip)"
    fi
fi

echo "corpora under $DEST:"
for corpus in musan_noise rirs esc50_birds; do
    echo "  $corpus: $(find -L "$DEST/$corpus" -type f 2>/dev/null | wc -l | tr -d ' ') files"
done
