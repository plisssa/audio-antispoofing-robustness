#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data/raw/asvspoof2019_la}"
URL="${ASV2019_LA_URL:-https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip}"

fetch() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1 && curl -fSL --connect-timeout 60 --speed-time 120 --speed-limit 1024 --retry 20 --retry-delay 5 -C - "$url" -o "$out"; then
        return 0
    fi
    if command -v wget >/dev/null 2>&1 && wget -c --timeout=60 --tries=3 -O "$out" "$url"; then
        return 0
    fi
    return 1
}

if [ -d "$DEST/LA" ]; then
    echo "ASVspoof 2019 LA already present at $DEST/LA, skipping"
    exit 0
fi

mkdir -p "$DEST"
echo "downloading ASVspoof 2019 LA to $DEST (resumable)"
fetch "$URL" "$DEST/LA.zip"
unzip -q -o "$DEST/LA.zip" -d "$DEST"
echo "done"
echo "build manifest: adfd manifest --dataset asvspoof2019_la --root $DEST/LA --split eval --out data/asvspoof2019_la/eval.csv"
