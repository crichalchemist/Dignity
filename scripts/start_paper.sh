#!/usr/bin/env bash
# Start the paper trading soak loop.
# Writes PID to reports/paper.pid for stop_paper.sh.
# Requires: METAAPI_TOKEN, METAAPI_ACCOUNT_ID env vars and a checkpoint path.
#
# Usage: ./scripts/start_paper.sh <checkpoint_path>
#   e.g. ./scripts/start_paper.sh checkpoints/quant/epoch_100.pt

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path>}"
REPORTS_DIR="reports"
PID_FILE="${REPORTS_DIR}/paper.pid"
LOG_FILE="${REPORTS_DIR}/paper_soak_stdout.log"

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Paper loop already running (PID ${PID}). Stop it first with stop_paper.sh." >&2
        exit 1
    else
        echo "Stale PID file found — cleaning up."
        rm -f "${PID_FILE}"
    fi
fi

if [ -f "${REPORTS_DIR}/soak_breaker.lock" ]; then
    echo "Soak breaker lock exists at ${REPORTS_DIR}/soak_breaker.lock." >&2
    echo "Delete it manually before resuming." >&2
    exit 1
fi

mkdir -p "${REPORTS_DIR}"

echo "Starting paper trading soak loop..."
echo "  Checkpoint : ${CHECKPOINT}"
echo "  PID file   : ${PID_FILE}"
echo "  Log file   : ${LOG_FILE}"

conda run -n dignity-model python -c "
import asyncio
from backtest.paper_runner import SoakConfig, run_paper_loop
import os

config = SoakConfig(
    model_path='${CHECKPOINT}',
    metaapi_token=os.environ['METAAPI_TOKEN'],
    account_id=os.environ['METAAPI_ACCOUNT_ID'],
)
asyncio.run(run_paper_loop(config))
" >> "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Started. PID=$(cat "${PID_FILE}")"
