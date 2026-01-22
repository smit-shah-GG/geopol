#!/bin/bash
#
# TKG Retraining Shell Wrapper
#
# Designed to be called by cron or systemd timer for periodic retraining.
# Activates the Python environment and runs the retraining script.
#
# Usage:
#   ./scripts/schedule_retraining.sh              # Normal execution
#   ./scripts/schedule_retraining.sh --dry-run    # Test without training
#   ./scripts/schedule_retraining.sh --force      # Force immediate retraining
#
# Cron example (weekly Sunday 2 AM):
#   0 2 * * 0 /path/to/geopol/scripts/schedule_retraining.sh >> /path/to/geopol/logs/retraining/cron.log 2>&1
#
# Systemd timer example (create /etc/systemd/system/tkg-retrain.timer):
#   [Unit]
#   Description=TKG Weekly Retraining Timer
#
#   [Timer]
#   OnCalendar=Sun *-*-* 02:00:00
#   Persistent=true
#
#   [Install]
#   WantedBy=timers.target

set -euo pipefail

# Get script directory (handles symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
LOG_DIR="${PROJECT_ROOT}/logs/retraining"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/schedule_${TIMESTAMP}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== TKG Retraining Wrapper ==="
log "Project root: $PROJECT_ROOT"
log "Log file: $LOG_FILE"

# Change to project directory
cd "$PROJECT_ROOT"

# Check for uv (preferred) or fallback to python
if command -v uv &> /dev/null; then
    log "Using uv for Python environment"
    PYTHON_CMD="uv run python"
else
    log "uv not found, using system python"
    if [ -f ".venv/bin/python" ]; then
        PYTHON_CMD=".venv/bin/python"
        log "Using local venv: $PYTHON_CMD"
    else
        PYTHON_CMD="python3"
        log "Using system python3"
    fi
fi

# Run retraining script with passed arguments
log "Starting retraining script..."
log "Command: $PYTHON_CMD scripts/retrain_tkg.py $*"

$PYTHON_CMD scripts/retrain_tkg.py "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log "Retraining completed successfully"
else
    log "Retraining failed with exit code: $EXIT_CODE"
fi

log "=== Wrapper Complete ==="

exit $EXIT_CODE
