#!/usr/bin/env bash
set -euo pipefail

HOST="${CHARISMA_HOST:-charisma}"
REMOTE="${CHARISMA_PATH:-adfd-robustness}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    COMMIT="${COMMIT}-dirty"
fi
printf '%s\n' "$COMMIT" > REVISION
echo "REVISION = $COMMIT"

rsync -az --info=stats1 \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude 'data' \
    --exclude 'runs' \
    --exclude 'third_party' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    -e ssh \
    ./ "$HOST:$REMOTE/"

ssh "$HOST" "cd $REMOTE && chmod +x scripts/*.sh"
echo "синхронизировано в $HOST:$REMOTE"
