#!/bin/bash
# Watchdog: auto-restart experiment on crash, with exponential backoff
# Usage: bash watchdog.sh <experiment_name> <command...>

EXPERIMENT_NAME="${1:?Usage: watchdog.sh <name> <command...>}"
shift
CMD="$@"
LOG="/root/new_paper/results/watchdog_${EXPERIMENT_NAME}.log"
MAX_RESTARTS=50
RESTART_COUNT=0
BACKOFF=10

echo "[watchdog] Starting experiment: $EXPERIMENT_NAME" | tee -a "$LOG"
echo "[watchdog] Command: $CMD" | tee -a "$LOG"
echo "[watchdog] Max restarts: $MAX_RESTARTS" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') Run #$((RESTART_COUNT+1))" | tee -a "$LOG"
    
    eval "$CMD" 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    
    echo "[watchdog] $(date '+%Y-%m-%d %H:%M:%S') Exit code: $EXIT_CODE" | tee -a "$LOG"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[watchdog] Experiment completed successfully!" | tee -a "$LOG"
        exit 0
    fi
    
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    # Check if result files are still being produced (progress)
    RESULT_DIR="/root/new_paper/results/llm_eval/crossllm_minimax"
    CURRENT_COUNT=$(ls "$RESULT_DIR"/*.json 2>/dev/null | grep -v all_results | grep -v summary | wc -l)
    
    if [ "$CURRENT_COUNT" -ge 20 ]; then
        echo "[watchdog] All 20 files complete! Exiting." | tee -a "$LOG"
        exit 0
    fi
    
    echo "[watchdog] Progress: $CURRENT_COUNT/20 files. Restarting in ${BACKOFF}s..." | tee -a "$LOG"
    sleep $BACKOFF
    
    # Increase backoff but cap at 60s
    BACKOFF=$((BACKOFF < 60 ? BACKOFF + 10 : 60))
done

echo "[watchdog] Exceeded max restarts ($MAX_RESTARTS). Giving up." | tee -a "$LOG"
exit 1
