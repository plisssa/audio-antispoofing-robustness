#!/usr/bin/env bash
set -euo pipefail

TRACK="${1:-LA}"
DEST="${2:-data/raw/asvspoof2021_${TRACK,,}}"
mkdir -p "$DEST"

if [ "$TRACK" = "LA" ]; then
    AUDIO_URL="${ASV21_LA_AUDIO_URL:-https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz}"
    KEYS_URL="${ASV21_LA_KEYS_URL:-https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz}"
else
    AUDIO_URL="${ASV21_DF_AUDIO_URL:-https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz}"
    KEYS_URL="${ASV21_DF_KEYS_URL:-https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz}"
fi

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

if [ -d "$DEST/ASVspoof2021_${TRACK}_eval/flac" ] && [ -f "$DEST/keys/${TRACK}/CM/trial_metadata.txt" ]; then
    echo "ASVspoof2021 ${TRACK} already present at $DEST, skipping"
    exit 0
fi

echo "downloading ASVspoof2021 ${TRACK} audio"
[ -f "$DEST/audio.tar.gz" ] || fetch "$AUDIO_URL" "$DEST/audio.tar.gz"
tar -xzf "$DEST/audio.tar.gz" -C "$DEST"

echo "downloading ASVspoof2021 ${TRACK} keys"
[ -f "$DEST/keys.tar.gz" ] || fetch "$KEYS_URL" "$DEST/keys.tar.gz"
tar -xzf "$DEST/keys.tar.gz" -C "$DEST"

echo "done. expected layout under $DEST:"
echo "  ASVspoof2021_${TRACK}_eval/flac/*.flac"
echo "  keys/${TRACK}/CM/trial_metadata.txt"
echo "build manifest: adfd manifest --dataset asvspoof2021_${TRACK,,} --root $DEST --out data/asvspoof2021_${TRACK,,}/eval.csv"
