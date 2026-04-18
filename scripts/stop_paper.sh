#!/usr/bin/env bash
# Stop the paper trading soak loop cleanly via SIGTERM.
# The run loop catches SIGTERM and shuts down after the current bar completes.
#
# Usage: ./scripts/stop_paper.sh

set -euo pipefail

PID_FILE="reports/paper.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "No PID file at ${PID_FILE}. Is the paper loop running?" >&2
    exit 1
fi

PID=$(cat "${PID_FILE}")

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "Process ${PID} is not running. Cleaning up stale PID file."
    rm -f "${PID_FILE}"
    exit 0
fi

echo "Sending SIGTERM to paper loop (PID ${PID})..."
kill -TERM "${PID}"

# Wait up to 30s for clean exit
for i in $(seq 1 30); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "Paper loop stopped cleanly."
        rm -f "${PID_FILE}"
        exit 0
    fi
    sleep 1
done

echo "Process did not stop within 30s. Sending SIGKILL." >&2
kill -KILL "${PID}"
rm -f "${PID_FILE}"
