#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data/raw/in_the_wild}"
URL="${ITW_URL:-https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download}"

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

if [ -f "$DEST/release_in_the_wild/meta.csv" ]; then
    echo "In-the-Wild already present at $DEST/release_in_the_wild, skipping"
    exit 0
fi

mkdir -p "$DEST"
echo "downloading In-the-Wild to $DEST"
fetch "$URL" "$DEST/release_in_the_wild.zip"
unzip -q -o "$DEST/release_in_the_wild.zip" -d "$DEST"
echo "done"
echo "build manifest: adfd manifest --dataset in_the_wild --root $DEST/release_in_the_wild --out data/in_the_wild/manifest.csv --train-ratio 0.2"
