"""One-off calibration: fetch public hourly bars and write the reference table.

Run from the repo root: ``python -m scripts.calibrate``. Needs ccxt, which is
not a project dependency, so it is imported inside ``main``.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data.stats import CALIBRATION_CHUNK_LEN, STAT_NAMES, calibration_stats, widen_band

PAIRS = ("BTC/USD", "ETH/USD")
WINDOWS = (("2023-11-01", "2024-11-01"), ("2024-11-01", "2025-11-01"))
HOUR_MS = 3_600_000
MIN_BARS = int(0.95 * 8760)
DEFAULT_OUT = "data/calibration/reference_stats.json"


def _ms(date: str) -> int:
    return int(
        datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp() * 1000
    )


def fetch_window(exchange, pair: str, start: str, end: str):
    """Return (close, volume, n_dropped) for hourly bars in [start, end)."""
    end_ms = _ms(end)
    cursor = _ms(start)
    bars: dict[int, tuple[float, float]] = {}
    while cursor < end_ms:
        page = exchange.fetch_ohlcv(pair, "1h", since=cursor, limit=1000)
        if not page:
            break
        for ts, _o, _h, _l, close, volume in page:
            if ts < end_ms:
                bars[ts] = (close, volume)
        last = page[-1][0]
        if last < cursor:
            break
        cursor = last + HOUR_MS
    kept = sorted((ts, c, v) for ts, (c, v) in bars.items() if v > 0)
    dropped = len(bars) - len(kept)
    if len(kept) < MIN_BARS:
        raise RuntimeError(
            f"{pair} {start}..{end}: only {len(kept)} usable bars, need {MIN_BARS}"
        )
    close = np.array([c for _, c, _ in kept], dtype=float)
    volume = np.array([v for _, _, v in kept], dtype=float)
    return close, volume, dropped


def main(argv: list[str] | None = None) -> int:
    import ccxt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", default="bitstamp")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    exchange = getattr(ccxt, args.exchange)()
    fetched = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = []  # (pair, window, stats)
    for pair in PAIRS:
        for start, end in WINDOWS:
            close, volume, dropped = fetch_window(exchange, pair, start, end)
            print(f"{pair} {start}..{end}: kept={len(close)} dropped={dropped}")
            s = calibration_stats(close, volume)
            for name in STAT_NAMES:
                if not math.isfinite(s[name]):
                    raise ValueError(
                        f"{pair} {start}..{end}: {name} is not finite ({s[name]})"
                    )
            rows.append((pair, f"{start}..{end}", s))

    table = {}
    for name in STAT_NAMES:
        lo, hi = widen_band([s[name] for _, _, s in rows])
        table[name] = {
            "min": lo,
            "max": hi,
            "sources": [
                f"{args.exchange} {pair} 1h {window} "
                f"chunk_len={CALIBRATION_CHUNK_LEN} value={s[name]:.6g} "
                f"fetched={fetched}"
                for pair, window, s in rows
            ],
        }

    print("pair window " + " ".join(STAT_NAMES))
    for pair, window, s in rows:
        print(pair, window, *(f"{s[n]:.6g}" for n in STAT_NAMES))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
