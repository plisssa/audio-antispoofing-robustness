#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/adfd-robustness"
REPORT="runs/WATCH_REPORT.txt"
SEEN="runs/.watch_seen"
touch "$REPORT" "$SEEN"

emit() {
    printf '\n########## %s  (%s) ##########\n' "$1" "$(date '+%d.%m %H:%M')" >> "$REPORT"
    cat "$2" >> "$REPORT"
}

for _ in $(seq 1 4032); do
    for f in runs_defence_noise/pipeline/*/defence_aggregate.csv \
             runs_eot/pipeline/*/defence_aggregate.csv \
             runs_eot30_*/pipeline/*/defence_aggregate.csv \
             runs/pipeline/*_attack_matrix/transfer_aggregate.csv \
             runs_fine/pipeline/*_attack/transfer_aggregate.csv; do
        [ -f "$f" ] || continue
        grep -qxF "$f" "$SEEN" && continue
        emit "$f" "$f"
        printf '%s\n' "$f" >> "$SEEN"
    done
    sleep 600
done
