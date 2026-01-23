"""
Cross-market correlation analysis for FX, crypto, and commodities.

This module provides tools to analyze correlations between different asset classes,
which can be used as features for multi-market trading strategies.

Usage:
    python -m features.correlation_analysis \
        --data-dir yfinance_output \
        --output-dir correlation_output \
        --interval 1h
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Asset class categorization
ASSET_CLASSES = {
    "major_fx": ["eurusd", "gbpusd", "usdjpy", "usdchf", "audusd", "usdcad", "nzdusd"],
    "cross_fx": ["eurgbp", "eurjpy", "eurchf", "gbpjpy", "gbpchf", "euraud", "eurcad",
                 "gbpcad", "gbpaud", "gbpnzd", "eurnzd", "audjpy", "audcad", "audchf",
                 "audnzd", "nzdjpy", "nzdcad", "nzdchf", "cadchf", "cadjpy", "chfjpy"],
    "emerging_fx": ["usdbrl", "usdrub", "usdinr", "usdcny", "usdzar", "usdtry"],
    "crypto": ["btcusd", "ethusd", "solusd", "bnbusd", "adausd", "xrpusd",
               "dogeusd", "avaxusd", "maticusd", "linkusd", "dotusd"],
    "commodities": ["xauusd", "xagusd"],
}

# Flatten for quick lookup
ALL_PAIRS = {pair: cls for cls, pairs in ASSET_CLASSES.items() for pair in pairs}


def load_pair_data(data_dir: Path, pair: str, interval: str) -> pd.DataFrame | None:
    """Load OHLCV data for a single pair."""
    pair_dir = data_dir / pair
    csv_path = pair_dir / f"{pair}_{interval}.csv"

    if not csv_path.exists():
        # Try daily fallback
        csv_path = pair_dir / f"{pair}_1d.csv"
        if not csv_path.exists():
            return None

    df = pd.read_csv(csv_path, parse_dates=["Date"] if "Date" in pd.read_csv(csv_path, nrows=1).columns else [0])

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})

    # Ensure datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

    return df


def compute_returns(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """Compute log returns over multiple periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' column and datetime index.
    periods : List[int]
        Return horizons (e.g., [1, 5, 20] for 1-bar, 5-bar, 20-bar returns).

    Returns
    -------
    pd.DataFrame
        Returns for each period as columns: ret_1, ret_5, ret_20, etc.
    """
    if periods is None:
        periods = [1, 5, 20]
    results = {}
    close = df["close"] if "close" in df.columns else df.iloc[:, 3]  # Assume 4th col is close

    for p in periods:
        results[f"ret_{p}"] = np.log(close / close.shift(p))

    return pd.DataFrame(results, index=df.index)


def build_returns_matrix(
        data_dir: Path,
        pairs: list[str],
        interval: str,
        return_period: int = 1
) -> pd.DataFrame:
    """Build a matrix of returns across all pairs, aligned by datetime.

    Parameters
    ----------
    data_dir : Path
        Directory containing pair subdirectories.
    pairs : List[str]
        List of pair names to include.
    interval : str
        Data interval (e.g., '1h', '1d').
    return_period : int
        Return horizon in bars.

    Returns
    -------
    pd.DataFrame
        Matrix with datetime index and pair returns as columns.
    """
    returns_dict = {}

    for pair in pairs:
        df = load_pair_data(data_dir, pair, interval)
        if df is None:
            print(f"[warn] No data for {pair}")
            continue

        ret = compute_returns(df, periods=[return_period])
        returns_dict[pair] = ret[f"ret_{return_period}"]

    if not returns_dict:
        return pd.DataFrame()

    # Align all series by datetime
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how="all")

    return returns_df


def compute_rolling_correlation(
        returns_df: pd.DataFrame,
        window: int = 20,
        min_periods: int = 10
) -> dict[tuple[str, str], pd.Series]:
    """Compute rolling pairwise correlations.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns matrix from build_returns_matrix.
    window : int
        Rolling window size.
    min_periods : int
        Minimum observations required.

    Returns
    -------
    Dict[Tuple[str, str], pd.Series]
        Dictionary mapping (pair1, pair2) to rolling correlation series.
    """
    pairs = returns_df.columns.tolist()
    correlations = {}

    for i, p1 in enumerate(pairs):
        for p2 in pairs[i + 1:]:
            corr = returns_df[p1].rolling(window=window, min_periods=min_periods).corr(returns_df[p2])
            correlations[(p1, p2)] = corr

    return correlations


def compute_correlation_matrix(
        returns_df: pd.DataFrame,
        method: str = "pearson"
) -> pd.DataFrame:
    """Compute static correlation matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns matrix from build_returns_matrix.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
        Full correlation matrix.
    """
    return returns_df.corr(method=method)


def compute_cross_asset_correlations(
        returns_df: pd.DataFrame,
        reference_pairs: list[str] | None = None
) -> pd.DataFrame:
    """Compute correlations of each pair against reference pairs (e.g., BTC, Gold).

    This is useful for understanding how FX pairs move relative to crypto/commodities.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns matrix.
    reference_pairs : List[str], optional
        Reference pairs to correlate against. Default: ['btcusd', 'ethusd', 'xauusd'].

    Returns
    -------
    pd.DataFrame
        DataFrame with pairs as index, reference correlations as columns.
    """
    if reference_pairs is None:
        reference_pairs = ["btcusd", "ethusd", "xauusd"]

    # Filter to available reference pairs
    reference_pairs = [p for p in reference_pairs if p in returns_df.columns]

    if not reference_pairs:
        return pd.DataFrame()

    correlations = {}
    for ref in reference_pairs:
        ref_returns = returns_df[ref]
        corr_with_ref = returns_df.corrwith(ref_returns)
        correlations[f"corr_{ref}"] = corr_with_ref

    result = pd.DataFrame(correlations)
    result["asset_class"] = result.index.map(lambda x: ALL_PAIRS.get(x, "unknown"))

    return result


def find_highly_correlated_pairs(
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7
) -> list[tuple[str, str, float]]:
    """Find pairs with absolute correlation above threshold.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix.
    threshold : float
        Minimum absolute correlation to report.

    Returns
    -------
    List[Tuple[str, str, float]]
        List of (pair1, pair2, correlation) tuples.
    """
    pairs = []
    cols = corr_matrix.columns.tolist()

    for i, p1 in enumerate(cols):
        for p2 in cols[i + 1:]:
            corr = corr_matrix.loc[p1, p2]
            if abs(corr) >= threshold:
                pairs.append((p1, p2, corr))

    # Sort by absolute correlation descending
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def compute_regime_correlations(
        returns_df: pd.DataFrame,
        regime_series: pd.Series,
) -> dict[int, pd.DataFrame]:
    """Compute correlation matrices for each market regime.

    Useful for understanding how correlations change in risk-on vs risk-off environments.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns matrix.
    regime_series : pd.Series
        Series of regime labels (e.g., 0=low_vol, 1=high_vol, 2=trending).

    Returns
    -------
    Dict[int, pd.DataFrame]
        Correlation matrix for each regime.
    """
    # Align indices
    common_idx = returns_df.index.intersection(regime_series.index)
    returns_aligned = returns_df.loc[common_idx]
    regimes_aligned = regime_series.loc[common_idx]

    regime_corrs = {}
    for regime in regimes_aligned.unique():
        mask = regimes_aligned == regime
        regime_returns = returns_aligned.loc[mask]
        if len(regime_returns) > 10:
            regime_corrs[regime] = regime_returns.corr()

    return regime_corrs


def generate_correlation_features(
        returns_df: pd.DataFrame,
        target_pair: str,
        reference_pairs: list[str] | None = None,
        window: int = 20
) -> pd.DataFrame:
    """Generate rolling correlation features for a target pair.

    These can be used as input features for the trading model.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Returns matrix.
    target_pair : str
        The pair to generate features for.
    reference_pairs : List[str], optional
        Pairs to compute correlations against.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling correlation features.
    """
    if reference_pairs is None:
        # Default: major crypto, gold, major FX
        reference_pairs = ["btcusd", "ethusd", "xauusd", "eurusd", "usdjpy"]

    reference_pairs = [p for p in reference_pairs if p in returns_df.columns and p != target_pair]

    if target_pair not in returns_df.columns:
        return pd.DataFrame()

    target_returns = returns_df[target_pair]
    features = {}

    for ref in reference_pairs:
        ref_returns = returns_df[ref]
        rolling_corr = target_returns.rolling(window=window, min_periods=window // 2).corr(ref_returns)
        features[f"corr_{ref}_{window}"] = rolling_corr

    return pd.DataFrame(features, index=returns_df.index)


def main():
    parser = argparse.ArgumentParser(description="Cross-market correlation analysis")
    parser.add_argument("--data-dir", default="yfinance_output", help="Directory with downloaded data")
    parser.add_argument("--output-dir", default="correlation_output", help="Output directory for results")
    parser.add_argument("--interval", default="1h", help="Data interval")
    parser.add_argument("--return-period", type=int, default=1, help="Return horizon in bars")
    parser.add_argument("--corr-window", type=int, default=20, help="Rolling correlation window")
    parser.add_argument("--threshold", type=float, default=0.5, help="Correlation threshold for reporting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all available pairs
    all_pairs = list(ALL_PAIRS.keys())

    print(f"Loading returns for {len(all_pairs)} pairs...")
    returns_df = build_returns_matrix(data_dir, all_pairs, args.interval, args.return_period)

    if returns_df.empty:
        print("No data loaded. Run data/downloaders/yfinance.py first.")
        return

    print(f"Loaded {len(returns_df.columns)} pairs with {len(returns_df)} observations")

    # Compute static correlation matrix
    print("\nComputing correlation matrix...")
    corr_matrix = compute_correlation_matrix(returns_df)
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv")

    # Find highly correlated pairs
    print(f"\nPairs with |correlation| >= {args.threshold}:")
    high_corr_pairs = find_highly_correlated_pairs(corr_matrix, args.threshold)
    for p1, p2, corr in high_corr_pairs[:20]:
        class1, class2 = ALL_PAIRS.get(p1, "?"), ALL_PAIRS.get(p2, "?")
        print(f"  {p1} ({class1}) <-> {p2} ({class2}): {corr:.3f}")

    # Cross-asset correlations
    print("\nCross-asset correlations (vs BTC, ETH, Gold):")
    cross_corr = compute_cross_asset_correlations(returns_df)
    if not cross_corr.empty:
        cross_corr.to_csv(output_dir / "cross_asset_correlations.csv")

        # Show by asset class
        for asset_class in ASSET_CLASSES:
            class_pairs = cross_corr[cross_corr["asset_class"] == asset_class]
            if not class_pairs.empty:
                print(f"\n  {asset_class}:")
                for col in cross_corr.columns:
                    if col.startswith("corr_"):
                        mean_corr = class_pairs[col].mean()
                        print(f"    avg {col}: {mean_corr:.3f}")

    # Generate example correlation features for EURUSD
    print("\nGenerating correlation features for eurusd...")
    if "eurusd" in returns_df.columns:
        corr_features = generate_correlation_features(returns_df, "eurusd", window=args.corr_window)
        if not corr_features.empty:
            corr_features.to_csv(output_dir / "eurusd_correlation_features.csv")
            print(f"  Saved {len(corr_features.columns)} features to eurusd_correlation_features.csv")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
