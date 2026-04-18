"""MetaApi data source and trade executor.

Two responsibilities, one module:
  MetaApiSource  — pulls historical OHLCV and streams live bars via WebSocket
  MetaApiExecutor — translates PolicyHead action index → MetaApi market order

MetaApi SDK (metaapi-cloud-sdk) is imported lazily inside connect() so the
rest of the codebase can import this module without credentials or network
access. Tests and paper-trading mode never trigger the real SDK import path.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd


def _filter_date_range(
    df: pd.DataFrame,
    date_range: tuple[str, str] | None,
) -> pd.DataFrame:
    """Slice a timestamp-indexed DataFrame to [start, end] inclusive.

    Pure function — no side effects, easily unit-testable without a live
    MetaApi connection. Called by get_history() to enforce temporal splits.
    """
    if date_range is None:
        return df
    return df.loc[date_range[0] : date_range[1]]


# ---------------------------------------------------------------------------
# MetaApiSource
# ---------------------------------------------------------------------------

class MetaApiSource:
    """Stream and fetch OHLCV data from a MetaApi-connected MT4/MT5 account.

    Usage:
        src = MetaApiSource(token=os.environ["METAAPI_TOKEN"],
                            account_id=os.environ["METAAPI_ACCOUNT"],
                            symbol="EURUSD")
        await src.connect()
        df = await src.get_history(bars=5000)  # DataFrame: open/high/low/close/volume
        async for bar in src.stream():          # live 1-bar updates
            process(bar)
        await src.disconnect()
    """

    def __init__(
        self,
        token: str,
        account_id: str,
        symbol: str,
        timeframe: str = "1m",
        date_range: tuple[str, str] | None = None,
    ) -> None:
        self._token = token
        self._account_id = account_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.date_range = date_range  # (start_iso, end_iso) for temporal split enforcement
        self._api = None
        self._account = None
        self._connection = None

    async def connect(self) -> None:
        """Establish MetaApi WebSocket connection. Lazily imports SDK."""
        from metaapi_cloud_sdk import MetaApi  # type: ignore[import]

        self._api = MetaApi(self._token)
        self._account = await self._api.metatrader_account_api.get_account(
            self._account_id
        )
        await self._account.deploy()
        await self._account.wait_connected()
        self._connection = self._account.get_rpc_connection()
        await self._connection.connect()
        await self._connection.wait_synchronized()

    async def disconnect(self) -> None:
        """Close MetaApi connection gracefully."""
        if self._connection is not None:
            await self._connection.close()
        if self._api is not None:
            self._api.close()

    async def get_history(
        self,
        bars: int = 10_000,
        start_time: str | None = "2016-01-01T00:00:00Z",
    ) -> pd.DataFrame:
        """Pull historical OHLCV bars.

        Args:
            bars: Maximum number of bars to fetch.
            start_time: ISO 8601 UTC timestamp for the earliest bar.
                        Defaults to "2016-01-01T00:00:00Z" (project-wide
                        historical data horizon). Pass None to let MetaApi
                        use its own default lookback.

        Returns DataFrame with lowercase columns:
            open, high, low, close, volume
        Indexed by UTC timestamp.
        """
        if self._connection is None:
            raise RuntimeError("Call connect() before get_history().")

        kwargs: dict = {"bars_count": bars}
        if start_time is not None:
            kwargs["start_time"] = start_time

        candles = await self._connection.get_historical_candles(
            self.symbol, self.timeframe, **kwargs
        )
        records = [
            {
                "timestamp": c["time"],
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c.get("tickVolume", 0),
            }
            for c in candles
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        return _filter_date_range(df, self.date_range)

    async def stream(self) -> AsyncIterator[pd.Series]:
        """Yield one pd.Series per closed bar as it arrives via WebSocket."""
        if self._connection is None:
            raise RuntimeError("Call connect() before stream().")

        queue: asyncio.Queue[pd.Series] = asyncio.Queue()

        class _BarListener:
            async def on_candle(self, account_id: str, candle: dict) -> None:
                row = pd.Series(
                    {
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle.get("tickVolume", 0),
                    },
                    name=pd.Timestamp(candle["time"]),
                )
                await queue.put(row)

        self._connection.add_synchronization_listener(_BarListener())

        while True:
            yield await queue.get()


# ---------------------------------------------------------------------------
# MetaApiExecutor
# ---------------------------------------------------------------------------

class MetaApiExecutor:
    """Translate PolicyHead action index → MetaApi market order.

    In paper mode (default) all operations are simulated locally — no SDK
    calls are made. Set paper=False and supply real credentials for live
    execution.
    """

    ACTION_MAP: ClassVar[dict[int, str]] = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        token: str,
        account_id: str,
        symbol: str,
        max_position_size: float = 1.0,
        max_drawdown: float = 0.05,
        paper: bool = True,
    ) -> None:
        self._token = token
        self._account_id = account_id
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.paper = paper
        self._connection = None

    async def connect(self) -> None:
        """Connect for live execution. No-op in paper mode."""
        if self.paper:
            return
        from metaapi_cloud_sdk import MetaApi  # type: ignore[import]

        api = MetaApi(self._token)
        account = await api.metatrader_account_api.get_account(self._account_id)
        await account.deploy()
        await account.wait_connected()
        self._connection = account.get_rpc_connection()
        await self._connection.connect()
        await self._connection.wait_synchronized()

    async def execute(
        self, action_idx: int, position_size: float, var_estimate: float = 0.0
    ) -> dict | None:
        """Place a market order for the given action.

        Runs check_risk_gate() before any order. A gate block returns None
        even in live mode — paper mode reflects real gate behavior too.

        Args:
            action_idx: 0=HOLD, 1=BUY, 2=SELL
            position_size: Desired lot size.
            var_estimate: RiskHead output (fraction of portfolio at risk).

        Returns:
            Order result dict on BUY/SELL when gate allows, None otherwise.
        """
        from core.execution import check_risk_gate

        action = self.ACTION_MAP.get(action_idx, "HOLD")

        if action == "HOLD":
            return None

        gate = check_risk_gate(
            var_estimate=var_estimate,
            position_size=position_size,
            max_drawdown=self.max_drawdown,
            max_position_size=self.max_position_size,
        )
        if not gate.allowed:
            return None

        if self.paper:
            return {"action": action, "symbol": self.symbol, "size": gate.adjusted_size, "paper": True}

        if self._connection is None:
            raise RuntimeError("Call connect() before execute() in live mode.")

        if action == "BUY":
            result = await self._connection.create_market_buy_order(
                self.symbol, gate.adjusted_size
            )
        else:
            result = await self._connection.create_market_sell_order(
                self.symbol, gate.adjusted_size
            )

        return result
