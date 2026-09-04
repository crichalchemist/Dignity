#!/usr/bin/env bash
# Live trading monitor — prints rolling 7-day Sharpe and drawdown.
# Tails live_trading_log.jsonl and recomputes stats on each new line.
#
# Usage: ./scripts/monitor.sh

set -euo pipefail

LOG_FILE="reports/live_trading_log.jsonl"

if [ ! -f "${LOG_FILE}" ]; then
    echo "No live trading log found at ${LOG_FILE}." >&2
    echo "Has live trading started? Run go_live.sh first." >&2
    exit 1
fi

echo "Dignity Core — Live Monitor (ctrl-c to stop)"
echo "Watching: ${LOG_FILE}"
echo "---------------------------------------------------"

tail -f "${LOG_FILE}" | conda run -n dignity-model python -c "
import sys
import json
import math
from collections import deque

WINDOW_DAYS = 7
entries = deque()

def compute_stats(entries):
    if not entries:
        return 0.0, 0.0
    pnls = [e.get('realized_pnl', 0.0) for e in entries]
    n = len(pnls)
    mean = sum(pnls) / n
    if n < 2:
        sharpe = 0.0
    else:
        variance = sum((p - mean) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-9
        sharpe = (mean / std) * math.sqrt(252 * 24)  # annualized, hourly bars
    drawdown = sum(pnls)
    return round(sharpe, 3), round(drawdown * 100, 4)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        entry = json.loads(line)
    except json.JSONDecodeError:
        continue
    entries.append(entry)
    # Keep rolling 7-day window (approx 7*24 hourly bars)
    while len(entries) > WINDOW_DAYS * 24:
        entries.popleft()
    sharpe, dd_pct = compute_stats(entries)
    action = entry.get('action', '?')
    ts = entry.get('timestamp', '')[:19]
    print(f'{ts}  action={action:<8}  7d-sharpe={sharpe:>7.3f}  7d-pnl={dd_pct:>+8.4f}%', flush=True)
"
