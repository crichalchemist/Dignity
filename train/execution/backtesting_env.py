"""Backtesting-based retail execution environment.

This environment mirrors the lightweight ``SimulatedRetailExecutionEnv`` API
(`reset`/`step`) but delegates order execution and portfolio accounting to
`backtesting.py`. It lets RL agents interact with historical OHLC data while
reusing the battle‑tested fills, equity curve, and PnL accounting provided by
the library.

Design notes
------------
* Each step appends the chosen action to an action schedule and replays the
  episode to date through backtesting.py to obtain the latest equity curve.
  This keeps stepwise rewards consistent with the library's fills and PnL
  math, at the cost of some extra computation.
* Actions follow the same dataclasses used by ``SimulatedRetailExecutionEnv``:
  market/limit/hold with side (buy/sell) and size respecting ``lot_size``.
* Observations report the current bar's close price as ``mid_price`` and the
  latest inventory inferred from backtesting's equity curve ``Position``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

try:
    from backtesting import Backtest, Strategy
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "backtesting.py is required for BacktestingRetailExecutionEnv. "
        "Install with `pip install backtesting>=0.3.2`."
    ) from exc

from train.execution.simulated_retail_env import ExecutionConfig, OrderAction


@dataclass
class BacktestingObservation:
    """Observation returned by the backtesting execution environment."""

    mid_price: float
    inventory: float
    cash: float
    realized_pnl: float
    portfolio_value: float
    step: int

    def as_dict(self) -> dict[str, float]:
        return {
            "mid_price": self.mid_price,
            "inventory": self.inventory,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "portfolio_value": self.portfolio_value,
            "step": float(self.step),
        }


class BacktestingRetailExecutionEnv:
    """Gym-like retail execution environment powered by backtesting.py."""

    def __init__(
            self,
            price_df: pd.DataFrame,
            config: ExecutionConfig | None = None,
            logger: logging.Logger | None = None,
    ) -> None:
        """
        Parameters
        ----------
        price_df:
            Historical OHLC data. Columns can be lower or upper case; they will
            be normalised to the format expected by backtesting.py.
        config:
            Execution configuration (cash, lot size, etc.).
        logger:
            Optional logger for debug/info output.
        """
        self.config = config or ExecutionConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.price_df = self._normalise_price_df(price_df)
        if len(self.price_df) < 2:
            raise ValueError("price_df must contain at least two rows.")

        self._actions: list[OrderAction] = []
        self._step_index = 0
        self._prev_equity = self.config.initial_cash
        self._latest_inventory = 0.0
        self._latest_cash = self.config.initial_cash
        self._latest_realized = 0.0

    # Public API -----------------------------------------------------
    def reset(self) -> dict[str, float]:
        self._actions.clear()
        self._step_index = 0
        self._prev_equity = self.config.initial_cash
        self._latest_inventory = 0.0
        self._latest_cash = self.config.initial_cash
        self._latest_realized = 0.0
        return self._current_observation().as_dict()

    def step(self, action: OrderAction | None) -> tuple[dict[str, float], float, bool, dict]:
        """Advance one bar using the provided order action."""
        normalized = action.normalized(self.config) if action is not None else OrderAction("hold", "buy", 0.0)
        self._actions.append(normalized)
        obs, equity = self._run_prefix()
        reward = equity - self._prev_equity
        self._prev_equity = equity

        self._step_index += 1
        done = self._step_index >= len(self.price_df) - 1
        return obs.as_dict(), reward, done, {}

    # Internals ------------------------------------------------------
    def _normalise_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure price columns match backtesting's expected casing."""
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        cols = {c: rename_map.get(c.lower(), c) for c in df.columns}
        price_df = df.rename(columns=cols)
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(price_df.columns)
        if missing:
            raise ValueError(f"price_df missing columns: {sorted(missing)}")
        return price_df.reset_index(drop=True)

    def _make_strategy(self) -> type:
        """Create a Strategy subclass bound to the current action schedule."""

        actions = list(self._actions)
        lot_size = self.config.lot_size

        class ActionStrategy(Strategy):  # type: ignore[misc]
            def init(self):  # noqa: D401
                """No indicator initialisation needed."""
                return None

            def next(self):  # noqa: D401
                """Place orders scheduled for this bar."""
                idx = int(self.i)
                if idx >= len(actions):
                    return
                act = actions[idx]
                if act.action_type == "hold" or act.size <= 0:
                    return
                size_units = max(act.size / lot_size, 0.0)
                if act.action_type == "market":
                    if act.side == "buy":
                        self.buy(size=size_units)
                    else:
                        self.sell(size=size_units)
                elif act.action_type == "limit":
                    limit_price = act.limit_price or float(self.data.Close[-1])
                    if act.side == "buy":
                        self.buy(size=size_units, limit=limit_price)
                    else:
                        self.sell(size=size_units, limit=limit_price)

        return ActionStrategy

    def _run_prefix(self) -> tuple[BacktestingObservation, float]:
        """Replay the episode so far and return latest observation and equity."""
        end_idx = min(self._step_index + 1, len(self.price_df))
        window = self.price_df.iloc[: end_idx + 1]
        strategy = self._make_strategy()

        bt = Backtest(
            window,
            strategy,
            cash=self.config.initial_cash,
            commission=0.0,
            exclusive_orders=False,
            trade_on_close=True,
        )
        stats = bt.run()
        equity_curve = stats["_equity_curve"]
        equity = float(equity_curve["Equity"].iloc[-1])
        position = float(equity_curve["Position"].iloc[-1])

        # backtesting.py expresses position in units; convert to notional quantity.
        self._latest_inventory = position * self.config.lot_size
        self._latest_cash = equity - self._latest_inventory * float(window["Close"].iloc[-1])
        self._latest_realized = float(stats.get("Equity Final", equity) - stats.get("Open Trades PnL", 0.0))
        obs = self._current_observation(current_price=float(window["Close"].iloc[-1]))
        return obs, equity

    def _current_observation(self, current_price: float | None = None) -> BacktestingObservation:
        price = current_price if current_price is not None else float(self.price_df["Close"].iloc[self._step_index])
        portfolio_value = self._latest_cash + self._latest_inventory * price
        return BacktestingObservation(
            mid_price=price,
            inventory=self._latest_inventory,
            cash=self._latest_cash,
            realized_pnl=self._latest_realized,
            portfolio_value=portfolio_value,
            step=self._step_index,
        )


__all__ = ["BacktestingRetailExecutionEnv", "BacktestingObservation"]
