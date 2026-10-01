#!/usr/bin/env bash
JOBID="$(squeue -u "$USER" -h -o '%i' -n adfd-pipeline 2>/dev/null | head -1)"
STATE="$(squeue -u "$USER" -h -o '%T' -n adfd-pipeline 2>/dev/null | head -1)"

echo "=== job ==="
squeue -u "$USER" 2>/dev/null

if [ -n "$JOBID" ] && [ -f "runs/slurm-adfd-pipeline-$JOBID.log" ]; then
    LOG="runs/slurm-adfd-pipeline-$JOBID.log"
else
    LOG="$(ls -t runs/slurm-adfd-pipeline-*.log 2>/dev/null | head -1)"
fi

if [ "$STATE" = "PENDING" ]; then
    echo "  job $JOBID PENDING — в очереди, ещё не стартовал (лог ниже от прошлого прогона)"
fi

echo "=== stages (${LOG:-no log yet}) ==="
if [ -n "$LOG" ]; then
    grep -E '\[pipeline\]' "$LOG" | tail -30
else
    echo "  no pipeline log yet"
fi

if [ -f runs/pipeline/status.json ]; then
    echo "=== counts (runs/pipeline/status.json) ==="
    grep -oE '"state": *"[a-z]+"' runs/pipeline/status.json | sort | uniq -c
fi
