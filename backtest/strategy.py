"""DignityStrategy — backtesting.py Strategy driven by cascade model signals.

Pre-computed signals are injected via class-level _signals dict before
Backtest.run() is called. This matches backtesting.py's design: indicator
arrays are attached once in init(), values read bar-by-bar in next().
"""

import numpy as np
from backtesting import Strategy

# Action index constants — must match MetaApiExecutor.ACTION_MAP
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


class DignityStrategy(Strategy):
    """Trade execution strategy driven by Dignity cascade model outputs.

    Signals flow: RegimeHead → RiskHead → AlphaHead → [RiskGate] → PolicyHead.
    The risk gate is replicated here: if var_estimate > max_drawdown, skip
    the bar regardless of the action signal. This mirrors core/execution.py.

    Class attributes set by BacktestRunner before Backtest.run():
        _signals: dict with keys "action", "var", "alpha", "regime"

    Optimizable parameters (usable with bt.optimize()):
        max_drawdown: float — blocks trades when var_estimate exceeds this
        position_size: float — fraction of equity per trade (0 < x <= 1)
    """

    # Injected by runner — full-length pre-computed signal arrays
    _signals: dict[str, np.ndarray] = {}

    # Strategy parameters — optimizable via bt.optimize()
    max_drawdown: float = 0.05
    position_size: float = 0.95

    def init(self) -> None:
        signals = type(self)._signals  # class-level, shared across all instances

        self._action_ind = self.I(
            lambda: signals["action"].astype(np.float64),
            name="Action",
            plot=False,
        )
        self._var_ind = self.I(
            lambda: signals["var"].astype(np.float64),
            name="VaR",
            overlay=False,
            color="red",
        )
        self._alpha_ind = self.I(
            lambda: signals["alpha"].astype(np.float64),
            name="Alpha",
            overlay=False,
            color="green",
        )
        self._regime_ind = self.I(
            lambda: signals["regime"].astype(np.float64),
            name="Regime",
            overlay=False,
            color="purple",
        )

    def next(self) -> None:
        action = int(self._action_ind[-1])
        var = float(self._var_ind[-1])

        # Replicate core/execution.py RiskGate — deterministic hard stop
        if var > self.max_drawdown:
            return

        if action == ACTION_BUY:
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                self.buy(size=self.position_size)

        elif action == ACTION_SELL:
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                self.sell(size=self.position_size)

        # ACTION_HOLD → do nothing
