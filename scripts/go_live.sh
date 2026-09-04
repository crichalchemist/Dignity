#!/usr/bin/env bash
# Live activation checklist — enforces all gates before starting live execution.
# No manual override path. Every check must pass.
#
# Usage: ./scripts/go_live.sh <checkpoint_path>
#   e.g. ./scripts/go_live.sh checkpoints/quant/epoch_100.pt

set -euo pipefail

CHECKPOINT="${1:?Usage: $0 <checkpoint_path>}"
REPORTS_DIR="reports"
PID_FILE="${REPORTS_DIR}/live.pid"
LOG_FILE="${REPORTS_DIR}/live_stdout.log"

echo "============================================================"
echo "  Dignity Core — Live Activation Checklist"
echo "============================================================"

# --- Gate: credentials ---
if [ -z "${METAAPI_TOKEN:-}" ]; then
    echo "FAIL  METAAPI_TOKEN env var is not set"
    exit 1
fi
echo "PASS  METAAPI_TOKEN is set"

if [ -z "${METAAPI_ACCOUNT_ID:-}" ]; then
    echo "FAIL  METAAPI_ACCOUNT_ID env var is not set"
    exit 1
fi
echo "PASS  METAAPI_ACCOUNT_ID is set"

# --- Gate: checkpoint exists ---
if [ ! -f "${CHECKPOINT}" ]; then
    echo "FAIL  checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
echo "PASS  checkpoint exists: ${CHECKPOINT}"

# --- Gate: no existing live process ---
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "FAIL  live loop already running (PID ${PID})"
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

# --- Gate: no circuit breaker lock ---
if [ -f "${REPORTS_DIR}/circuit_breaker.lock" ]; then
    echo "FAIL  circuit_breaker.lock exists — delete manually to re-authorize"
    exit 1
fi
echo "PASS  no circuit breaker lock"

# --- Gates: backtest report + paper soak (delegated to Python) ---
echo ""
echo "Running backtest and soak gate checks..."
echo ""
conda run -n dignity-model python -m backtest.go_live_check
GATE_EXIT=$?

if [ "${GATE_EXIT}" -ne 0 ]; then
    echo ""
    echo "Live activation BLOCKED. Resolve gate failures above."
    exit 1
fi

# --- All gates passed — start live loop ---
echo ""
echo "============================================================"
echo "  All gates passed. Starting live execution."
echo "  Checkpoint : ${CHECKPOINT}"
echo "  PID file   : ${PID_FILE}"
echo "  Log file   : ${LOG_FILE}"
echo "============================================================"

mkdir -p "${REPORTS_DIR}"

conda run -n dignity-model python -c "
import asyncio
from backtest.live_runner import LiveConfig, run_live_loop
import os

config = LiveConfig(
    model_path='${CHECKPOINT}',
    metaapi_token=os.environ['METAAPI_TOKEN'],
    account_id=os.environ['METAAPI_ACCOUNT_ID'],
)
asyncio.run(run_live_loop(config))
" >> "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "Live loop started. PID=$(cat "${PID_FILE}")"
