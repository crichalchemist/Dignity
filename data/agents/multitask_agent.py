"""Multitask data agent – extends :class:`BaseDataAgent`.

The multitask version computes additional targets beyond the base agent:
- **direction_class**: Up/down/flat classification of future return
- **return_reg**: Raw future log return (regression)
- **next_close_reg**: Absolute future close price
- **vol_class**: Binary volatility increase/decrease
- **trend_class**: Trend direction over future window
- **vol_regime_class**: Volatility regime (decreasing/stable/increasing)
- **candle_class**: Candle pattern classification (doji/bullish/bearish/hammer)

Plus the standard auxiliary targets from BaseDataAgent:
- max_return, topk_returns, topk_prices, sell_now (optional)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import MultiTaskDataConfig

from .base_agent import BaseDataAgent, _label_from_return


class MultiTaskDataAgent(BaseDataAgent):
    """Multitask data agent with extended target computation.

    Parameters
    ----------
    cfg : MultiTaskDataConfig
        Configuration including vol_min_change threshold for volatility targets.
    """

    def __init__(self, cfg: MultiTaskDataConfig):
        # Re‑use the same underlying DataConfig fields that BaseDataAgent expects.
        super().__init__(cfg)

    def _create_primary_target(self, log_ret: float) -> int:
        """Multitask models always use direction classification as primary target.

        Parameters
        ----------
        log_ret : float
            Future log return.

        Returns
        -------
        int
            0 (down), 1 (flat), or 2 (up)
        """
        flat = self.cfg.flat_threshold
        if log_ret > flat:
            return 2
        if log_ret < -flat:
            return 0
        return 1

    def _build_windows(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            norm_stats,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build sliding windows with multitask targets.

        Extends BaseDataAgent._build_windows to compute additional multitask targets:
        - return_reg: Raw log return
        - next_close_reg: Future close price
        - vol_class: Volatility increase (binary)
        - trend_class: Trend direction over future window
        - vol_regime_class: Volatility regime (3-class)
        - candle_class: Candle pattern (4-class)

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe with OHLCV columns.
        feature_cols : List[str]
            List of feature column names to normalize.
        norm_stats : NormalizationStats
            Pre-computed normalization statistics.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            (sequences, targets) where targets dict includes all multitask labels.
        """
        t_in, t_out = self.cfg.t_in, self.cfg.t_out
        lookahead = self.cfg.lookahead_window or t_out
        top_k = max(1, self.cfg.top_k_predictions)

        # Normalize features
        features = norm_stats.apply(df[feature_cols].to_numpy(dtype=np.float32))

        # Compute log prices and returns
        log_close = np.log(df["close"].to_numpy())
        log_returns = np.diff(log_close, prepend=np.nan)

        # Storage for sequences and all targets
        sequences: list[np.ndarray] = []
        targets_dir_cls: list[int] = []
        targets_ret_reg: list[float] = []
        targets_next_close: list[float] = []
        targets_vol_cls: list[int] = []
        targets_max_return: list[float] = []
        targets_topk_returns: list[np.ndarray] = []
        targets_topk_prices: list[np.ndarray] = []
        targets_sell_now: list[int] = []
        targets_trend_cls: list[int] = []
        targets_vol_regime_cls: list[int] = []
        targets_candle_cls: list[int] = []

        last_idx = len(df) - max(t_out, lookahead)

        for idx in range(t_in - 1, last_idx):
            # Extract sequence window
            seq = features[idx - t_in + 1: idx + 1]

            # Compute primary future return target
            future_log_ret = log_close[idx + t_out] - log_close[idx]
            if not np.isfinite(future_log_ret):
                continue

            # Primary targets: direction, return, next_close
            direction_label = self._create_primary_target(future_log_ret)
            return_target = float(future_log_ret)
            next_close_target = float(df["close"].iloc[idx + t_out])

            # Auxiliary targets: max_return, topk
            future_returns = log_close[idx + 1: idx + lookahead + 1] - log_close[idx]
            if len(future_returns) < lookahead or np.isnan(future_returns).any():
                continue

            max_future_return = float(np.max(future_returns))
            sorted_returns = np.sort(future_returns)[::-1]
            topk_returns = sorted_returns[:top_k]
            if len(topk_returns) < top_k:
                topk_returns = np.pad(
                    topk_returns,
                    (0, top_k - len(topk_returns)),
                    constant_values=sorted_returns[-1]
                )
            topk_prices = np.exp(topk_returns) * df["close"].iloc[idx]

            # Volatility targets: compare past vs future volatility
            past_ret_window = log_returns[idx - t_out + 1: idx + 1]
            future_ret_window = log_returns[idx + 1: idx + t_out + 1]
            if len(past_ret_window) < t_out or len(future_ret_window) < t_out:
                continue
            if np.isnan(past_ret_window).any() or np.isnan(future_ret_window).any():
                continue

            past_vol = float(np.std(past_ret_window))
            future_vol = float(np.std(future_ret_window))
            vol_delta = future_vol - past_vol

            # Binary volatility change (increase/decrease)
            vol_label = 1 if vol_delta > self.cfg.vol_min_change else 0

            # 3-class volatility regime (decreasing/stable/increasing)
            if vol_delta > self.cfg.vol_min_change:
                vol_regime_label = 2  # increasing
            elif vol_delta < -self.cfg.vol_min_change:
                vol_regime_label = 0  # decreasing
            else:
                vol_regime_label = 1  # stable

            # Trend classification: average return over future window
            trend_avg_return = float(np.mean(future_ret_window))
            trend_label = int(_label_from_return(trend_avg_return, self.cfg.flat_threshold))

            # Candle pattern classification
            candle_row = df.iloc[idx]
            open_price = float(candle_row["open"])
            high_price = float(candle_row["high"])
            low_price = float(candle_row["low"])
            close_price = float(candle_row["close"])

            range_size = max(high_price - low_price, 1e-8)
            body = close_price - open_price
            body_ratio = abs(body) / range_size
            high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price

            if body_ratio < 0.1:
                candle_label = 0  # doji/indecision
            elif body > 0 and body_ratio > 0.55:
                candle_label = 1  # strong bullish body
            elif body < 0 and body_ratio > 0.55:
                candle_label = 2  # strong bearish body
            elif lower_wick / range_size > 0.35:
                candle_label = 3  # hammer/long lower wick
            else:
                candle_label = 3  # default

            # Append all targets
            sequences.append(seq)
            targets_dir_cls.append(direction_label)
            targets_ret_reg.append(return_target)
            targets_next_close.append(next_close_target)
            targets_vol_cls.append(vol_label)
            targets_max_return.append(max_future_return)
            targets_topk_returns.append(topk_returns)
            targets_topk_prices.append(topk_prices)
            if self.cfg.predict_sell_now:
                targets_sell_now.append(int(max_future_return <= 0.0))
            targets_trend_cls.append(trend_label)
            targets_vol_regime_cls.append(vol_regime_label)
            targets_candle_cls.append(candle_label)

        if not sequences:
            raise ValueError("No sequences created; check t_in/t_out and data length.")

        # Build targets dictionary
        targets: dict[str, np.ndarray] = {
            "primary": np.array(targets_dir_cls),  # Alias for direction_class
            "direction_class": np.array(targets_dir_cls),
            "return_reg": np.array(targets_ret_reg),
            "next_close_reg": np.array(targets_next_close),
            "vol_class": np.array(targets_vol_cls),
            "max_return": np.array(targets_max_return),
            "topk_returns": np.stack(targets_topk_returns),
            "topk_prices": np.stack(targets_topk_prices),
            "trend_class": np.array(targets_trend_cls),
            "vol_regime_class": np.array(targets_vol_regime_cls),
            "candle_class": np.array(targets_candle_cls),
        }

        if self.cfg.predict_sell_now:
            targets["sell_now"] = np.array(targets_sell_now)

        return np.stack(sequences), targets
