"""Signal processing utilities for transaction and market data sequences."""

from dataclasses import dataclass
from functools import reduce

import numpy as np
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------------
# Asset configuration — calibrates signal parameters per instrument class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetConfig:
    """Per-instrument-class signal parameter set.

    Prevents a single parameter configuration from producing meaningless
    features across instruments with 10x+ volatility differences (e.g.,
    EUR/USD vs BTC/USD). DC threshold is the primary differentiator.
    """

    asset_class: str
    rsi_period: int
    bollinger_window: int
    dc_threshold: float
    sma_windows: tuple[int, ...]


ASSET_CONFIGS: dict[str, AssetConfig] = {
    "forex": AssetConfig("forex", 14, 20, 0.0005, (10, 20, 50)),
    "crypto": AssetConfig("crypto", 7, 10, 0.005, (5, 10, 20)),
    "equity": AssetConfig("equity", 14, 20, 0.001, (10, 20, 50)),
    "commodity": AssetConfig("commodity", 14, 20, 0.002, (10, 20, 50)),
}


# ---------------------------------------------------------------------------
# Helper — exponential moving average (used by MACD, RSI, ATR, ADX)
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via recursive formula. No NaN in output."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; early values filled with expanding mean."""
    result = np.empty(len(values), dtype=np.float64)
    cumsum = np.cumsum(values)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start : i + 1])
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling std; early values filled with expanding std (0 for first elem)."""
    result = np.empty(len(values), dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.std(values[start : i + 1], ddof=0)
    return result


class SignalProcessor:
    """Compute market signals from OHLCV data.

    All methods are pure static functions returning float64 arrays of the same
    length as the input. Early-bar warmup values are filled with the first
    valid computed value rather than NaN, so downstream pipeline scaling never
    encounters NaN.
    """

    # ------------------------------------------------------------------
    # Original signals (preserved)
    # ------------------------------------------------------------------

    @staticmethod
    def volatility(values: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling standard deviation."""
        if len(values) < window:
            return np.zeros_like(values)
        result = np.zeros_like(values)
        for i in range(window, len(values)):
            result[i] = np.std(values[i - window : i])
        if window < len(values):
            result[:window] = result[window]
        return result

    @staticmethod
    def entropy(values: np.ndarray, bins: int = 10) -> float:
        """Shannon entropy of value distribution."""
        if len(values) == 0:
            return 0.0
        hist, _ = np.histogram(values, bins=bins)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return scipy_entropy(hist, base=2)

    @staticmethod
    def price_momentum(prices: np.ndarray, window: int = 10) -> np.ndarray:
        """Rate of change over window bars."""
        if len(prices) < window:
            return np.zeros_like(prices)
        result = np.zeros_like(prices)
        for i in range(window, len(prices)):
            result[i] = (prices[i] - prices[i - window]) / prices[i - window]
        return result

    @staticmethod
    def directional_change(prices: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Detect ±threshold price changes: 1 up, -1 down, 0 none."""
        if len(prices) < 2:
            return np.zeros_like(prices)
        changes = np.diff(prices) / prices[:-1]
        directions = np.zeros(len(prices))
        directions[1:][changes > threshold] = 1
        directions[1:][changes < -threshold] = -1
        return directions

    @staticmethod
    def bars_since_significant_move(
        prices: np.ndarray, vol_window: int = 20, vol_multiplier: float = 1.0
    ) -> np.ndarray:
        """
        Bars since last significant price movement.
        
        A significant move is when absolute price change exceeds vol_multiplier * 
        standard deviation of recent volatility. Tracks bars since last such event.
        """
        n = len(prices)
        bars_since = np.zeros(n, dtype=np.float64)
        
        if n < vol_window + 1:
            return bars_since
        
        # Compute rolling volatility
        price_changes = np.abs(np.diff(prices) / prices[:-1])
        
        for i in range(vol_window, n):
            # Rolling volatility window
            recent_changes = price_changes[max(0, i - vol_window):i]
            vol_std = np.std(recent_changes) if len(recent_changes) > 0 else 0.001
            threshold = vol_multiplier * vol_std

            # Check if current move is significant
            current_change = price_changes[i - 1] if i > 0 else 0

            if current_change > threshold:
                bars_since[i] = 0  # Reset counter
            elif i > 0:
                bars_since[i] = bars_since[i - 1] + 1
            else:
                bars_since[i] = 1

        # Warmup bars have no prior volatility context; fill with sequential
        # count so the model sees "many bars since last event" rather than 0
        # (0 would look like a constant event at every warmup bar).
        for j in range(min(vol_window, n)):
            bars_since[j] = float(j + 1)

        return bars_since

    @staticmethod
    def regime_detection(volatility: np.ndarray, vol_threshold: float = 0.5) -> np.ndarray:
        """Classify regime: 0=calm, 1=normal, 2=turbulent."""
        if len(volatility) == 0:
            return np.zeros(0, dtype=int)
        vol_mean = np.mean(volatility)
        vol_std = np.std(volatility)
        if vol_std == 0:
            return np.ones(len(volatility), dtype=int)
        normalized = (volatility - vol_mean) / vol_std
        regimes = np.ones(len(volatility), dtype=int)
        regimes[normalized < -vol_threshold] = 0
        regimes[normalized > vol_threshold] = 2
        return regimes

    # ------------------------------------------------------------------
    # New quant finance signals
    # ------------------------------------------------------------------

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index in [0, 100]."""
        prices = prices.astype(np.float64)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = _ema(np.concatenate([[gains[0]], gains]), period)
        avg_loss = _ema(np.concatenate([[losses[0]], losses]), period)

        # Avoid division by zero — errstate suppresses warning from the
        # unevaluated branch that numpy computes before np.where selects.
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        result = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))
        return result.astype(np.float64)

    @staticmethod
    def macd(
        prices: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD line, signal line, histogram. All same shape as prices."""
        prices = prices.astype(np.float64)
        ema_fast = _ema(prices, fast)
        ema_slow = _ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        prices: np.ndarray, window: int = 20, n_std: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bollinger %B and bandwidth. Both same shape as prices."""
        prices = prices.astype(np.float64)
        mid = _rolling_mean(prices, window)
        std = _rolling_std(prices, window)
        upper = mid + n_std * std
        lower = mid - n_std * std
        band_width = upper - lower
        # %B: where price sits within the band; safe divide when band=0
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_pct_b = (prices - lower) / band_width
        pct_b = np.where(band_width == 0, 0.5, raw_pct_b)
        return pct_b.astype(np.float64), band_width.astype(np.float64)

    @staticmethod
    def atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average True Range — smoothed via EMA."""
        high = high.astype(np.float64)
        low = low.astype(np.float64)
        close = close.astype(np.float64)
        prev_close = np.concatenate([[close[0]], close[:-1]])
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        return _ema(tr, period)

    @staticmethod
    def stochastic(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic oscillator %K and %D, both in [0, 100]."""
        high = high.astype(np.float64)
        low = low.astype(np.float64)
        close = close.astype(np.float64)
        n = len(close)
        k = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - k_period + 1)
            h_max = np.max(high[start : i + 1])
            l_min = np.min(low[start : i + 1])
            denom = h_max - l_min
            k[i] = 50.0 if denom == 0 else 100.0 * (close[i] - l_min) / denom
        d = _rolling_mean(k, d_period)
        return k, d

    @staticmethod
    def adx(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average Directional Index in [0, 100]. Measures trend strength."""
        high = high.astype(np.float64)
        low = low.astype(np.float64)
        close = close.astype(np.float64)
        n = len(close)
        prev_high = np.concatenate([[high[0]], high[:-1]])
        prev_low = np.concatenate([[low[0]], low[:-1]])

        dm_plus = np.where(
            (high - prev_high) > (prev_low - low),
            np.maximum(high - prev_high, 0.0),
            0.0,
        )
        dm_minus = np.where(
            (prev_low - low) > (high - prev_high),
            np.maximum(prev_low - low, 0.0),
            0.0,
        )
        atr_vals = SignalProcessor.atr(high, low, close, period)

        smooth_dm_plus = _ema(dm_plus, period)
        smooth_dm_minus = _ema(dm_minus, period)
        safe_atr = np.where(atr_vals == 0, 1e-10, atr_vals)
        di_plus = 100.0 * smooth_dm_plus / safe_atr
        di_minus = 100.0 * smooth_dm_minus / safe_atr
        dx_denom = di_plus + di_minus
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_dx = 100.0 * np.abs(di_plus - di_minus) / dx_denom
        dx = np.where(dx_denom == 0, 0.0, raw_dx)
        return _ema(dx, period)

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On-Balance Volume — cumulative volume signed by price direction."""
        close = close.astype(np.float64)
        volume = volume.astype(np.float64)
        direction = np.sign(np.diff(close))
        signed_vol = np.concatenate([[0.0], direction * volume[1:]])
        return np.cumsum(signed_vol)

    @staticmethod
    def vwap(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        """Volume-Weighted Average Price (cumulative session VWAP)."""
        typical = (high.astype(np.float64) + low.astype(np.float64) + close.astype(np.float64)) / 3.0
        volume = volume.astype(np.float64)
        cum_vol = np.cumsum(volume)
        cum_tp_vol = np.cumsum(typical * volume)
        safe_vol = np.where(cum_vol == 0, 1e-10, cum_vol)
        return cum_tp_vol / safe_vol

    @staticmethod
    def roc(prices: np.ndarray, period: int = 5) -> np.ndarray:
        """Rate of Change: (price[t] - price[t-period]) / price[t-period]."""
        prices = prices.astype(np.float64)
        result = np.zeros(len(prices), dtype=np.float64)
        for i in range(period, len(prices)):
            base = prices[i - period]
            result[i] = 0.0 if base == 0 else (prices[i] - base) / base
        # Fill warmup with first valid value
        if period < len(prices):
            result[:period] = result[period]
        return result

    @staticmethod
    def realized_volatility(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling realized volatility of log returns."""
        prices = prices.astype(np.float64)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        # Prepend 0 so output length matches input
        log_returns = np.concatenate([[0.0], log_returns])
        return SignalProcessor.volatility(log_returns, window)

    @staticmethod
    def vol_ratio(
        prices: np.ndarray, short_window: int = 5, long_window: int = 20
    ) -> np.ndarray:
        """Ratio of short-term to long-term realized volatility."""
        short_vol = SignalProcessor.realized_volatility(prices, short_window)
        long_vol = SignalProcessor.realized_volatility(prices, long_window)
        safe_long = np.where(long_vol == 0, 1e-10, long_vol)
        return (short_vol / safe_long).astype(np.float64)

    @staticmethod
    def order_flow_imbalance(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume signed by price direction: volume × sign(Δprice)."""
        close = close.astype(np.float64)
        volume = volume.astype(np.float64)
        direction = np.sign(np.diff(close))
        signed = np.concatenate([[0.0], direction * volume[1:]])
        return signed.astype(np.float64)

    @staticmethod
    def dc_state_machine(
        prices: np.ndarray, threshold: float = 0.0005
    ) -> dict[str, np.ndarray]:
        """Directional-change intrinsic time (dense enrichment mode).

        Tracks a DC state machine: emits an event when price moves by
        `threshold` from its last extreme. Returns three arrays aligned
        to the input:
          - dc_direction: 1 (upward DC), -1 (downward DC), 0 (none)
          - overshoot: price move beyond the DC event level
          - bars_since_event: integer count of bars since last event

        Dense mode: all rows annotated, inter-event information preserved.
        """
        prices = prices.astype(np.float64)
        n = len(prices)
        dc_direction = np.zeros(n, dtype=np.float64)
        overshoot = np.zeros(n, dtype=np.float64)
        bars_since_event = np.zeros(n, dtype=np.float64)

        extreme = prices[0]
        # Start looking for an upward DC; first threshold move sets direction.
        current_direction = -1
        bars_since = 0

        for i in range(1, n):
            p = prices[i]
            bars_since += 1

            if current_direction == 1:
                # In uptrend — looking for downward DC
                if p <= extreme * (1.0 - threshold):
                    dc_direction[i] = -1.0
                    overshoot[i] = max(0.0, (extreme - p) / extreme - threshold)
                    extreme = p
                    current_direction = -1
                    bars_since = 0
                elif p > extreme:
                    extreme = p
            else:
                # In downtrend or uninitialized — looking for upward DC
                if p >= extreme * (1.0 + threshold):
                    dc_direction[i] = 1.0
                    overshoot[i] = max(0.0, (p - extreme) / extreme - threshold)
                    extreme = p
                    current_direction = 1
                    bars_since = 0
                elif p < extreme:
                    extreme = p

            bars_since_event[i] = float(bars_since)

        return {
            "dc_direction": dc_direction,
            "overshoot": overshoot,
            "bars_since_event": bars_since_event,
        }

    # ------------------------------------------------------------------
    # process_sequence — 32-feature output
    # ------------------------------------------------------------------

    @classmethod
    def process_sequence(
        cls,
        volumes: np.ndarray,
        prices: np.ndarray,
        fees: np.ndarray | None = None,
        high: np.ndarray | None = None,
        low: np.ndarray | None = None,
        tx_count: np.ndarray | None = None,
        asset_config: AssetConfig | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute all 32 quant features from OHLCV and transaction inputs.

        This method assumes all inputs are provided and computes the full
        32-feature set required for the quant finance cascade model.
        """
        signals: dict[str, np.ndarray] = {}
        cfg = asset_config or ASSET_CONFIGS["equity"]

        # --- Ensure correct types ---
        volumes = volumes.astype(np.float64)
        prices = prices.astype(np.float64)
        fees = fees.astype(np.float64) if fees is not None else np.zeros_like(volumes)
        _high = high.astype(np.float64) if high is not None else prices
        _low = low.astype(np.float64) if low is not None else prices
        tx_count = tx_count.astype(np.float64) if tx_count is not None else np.zeros_like(volumes)

        # --- Feature Computation ---
        vol_raw = cls.volatility(prices)
        dc = cls.dc_state_machine(prices, threshold=cfg.dc_threshold)
        macd_line, macd_sig, macd_hist = cls.macd(prices, 12, 26, 9)
        pct_b, bw = cls.bollinger_bands(prices, window=cfg.bollinger_window, n_std=2.0)
        stoch_k, stoch_d = cls.stochastic(_high, _low, prices, 14, 3)
        bars_since = cls.bars_since_significant_move(prices, vol_window=20, vol_multiplier=1.0)

        # --- Assemble final 32-feature dictionary ---
        signals = {
            "volume": volumes,
            "price": prices,
            "fee_rate": fees,
            "tx_count": tx_count,
            "rsi": cls.rsi(prices, period=cfg.rsi_period),
            "macd_line": macd_line,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
            "bollinger_pct_b": pct_b,
            "bollinger_width": bw,
            "atr": cls.atr(_high, _low, prices, period=14),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "adx": cls.adx(_high, _low, prices, period=14),
            "obv": cls.obv(prices, volumes),
            "vwap": cls.vwap(_high, _low, prices, volumes),
            "roc_5": cls.roc(prices, period=5),
            "roc_20": cls.roc(prices, period=20),
            "momentum_10": cls.price_momentum(prices, window=10),
            "momentum_20": cls.price_momentum(prices, window=20),
            "volatility_5": cls.realized_volatility(prices, window=5),
            "volatility_20": cls.realized_volatility(prices, window=20),
            "vol_ratio": cls.vol_ratio(prices, short_window=5, long_window=20),
            "order_flow_imbalance": cls.order_flow_imbalance(prices, volumes),
            "dc_direction": dc["dc_direction"],
            "dc_overshoot": dc["overshoot"],
            "bars_since_event": bars_since,
            "dc_bars_since_event": dc["bars_since_event"],
            "volume_volatility": cls.volatility(volumes),
            "volume_entropy": np.array(
                [cls.entropy(volumes[max(0, i - 50):i + 1]) for i in range(len(volumes))]
            ),
            "price_change": np.gradient(prices),
            "directional_change": cls.directional_change(prices, threshold=cfg.dc_threshold),
            "regime": cls.regime_detection(vol_raw).astype(np.float64),
        }

        feature_order = [
            'volume', 'price', 'fee_rate', 'tx_count', 'rsi', 'macd_line',
            'macd_signal', 'macd_hist', 'bollinger_pct_b', 'bollinger_width',
            'atr', 'stoch_k', 'stoch_d', 'adx', 'obv', 'vwap', 'roc_5', 'roc_20',
            'momentum_10', 'momentum_20', 'volatility_5', 'volatility_20',
            'vol_ratio', 'order_flow_imbalance', 'dc_direction', 'dc_overshoot',
            'bars_since_event', 'dc_bars_since_event', 'volume_volatility',
            'volume_entropy', 'price_change', 'directional_change',
        ]

        # This will raise a KeyError if a feature is missing and also ensures order
        ordered_signals = {key: signals[key] for key in feature_order}

        return ordered_signals

