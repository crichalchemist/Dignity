"""Synthetic transaction data generator."""

import numpy as np
import pandas as pd


class SyntheticGenerator:
    """
    Generate synthetic transaction sequences for training/testing.

    Simulates realistic transaction patterns with:
    - Volume variations
    - Price movements
    - Fee dynamics
    - Anomalous patterns (for risk detection)
    """

    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_normal_sequence(
        self, length: int = 1000, base_volume: float = 100.0, volatility: float = 0.2
    ) -> dict[str, np.ndarray]:
        """
        Generate a normal transaction sequence.

        Args:
            length: Sequence length
            base_volume: Base transaction volume
            volatility: Volume volatility factor

        Returns:
            Dictionary with transaction features
        """
        # Volume with random walk
        volumes = np.zeros(length)
        volumes[0] = base_volume

        for i in range(1, length):
            change = np.random.randn() * volatility * base_volume
            volumes[i] = max(0.1, volumes[i - 1] + change)

        # Price (simulated market price)
        price = np.zeros(length)
        price[0] = 100.0

        for i in range(1, length):
            price_change = np.random.randn() * 0.01 * price[i - 1]
            price[i] = max(1.0, price[i - 1] + price_change)

        # Fee rate (mostly stable with occasional spikes)
        base_fee = 0.001
        fee_rate = np.full(length, base_fee)
        spike_indices = np.random.choice(length, size=int(length * 0.05), replace=False)
        fee_rate[spike_indices] *= np.random.uniform(2, 5, size=len(spike_indices))

        # Transaction count per block
        tx_count = np.random.poisson(lam=50, size=length)

        return {
            "volume": volumes,
            "price": price,
            "fee_rate": fee_rate,
            "tx_count": tx_count,
        }

    def generate_anomalous_sequence(
        self, length: int = 100, anomaly_type: str = "volume_spike"
    ) -> dict[str, np.ndarray]:
        """
        Generate anomalous transaction patterns.

        Args:
            length: Sequence length
            anomaly_type: Type of anomaly to inject
                - 'volume_spike': Sudden volume increase
                - 'price_manipulation': Abnormal price movement
                - 'fee_evasion': Unusually low fees

        Returns:
            Dictionary with anomalous transaction features
        """
        # Start with normal sequence
        data = self.generate_normal_sequence(length)

        # Inject anomaly in middle portion
        anomaly_start = length // 3
        anomaly_end = 2 * length // 3

        if anomaly_type == "volume_spike":
            # Sudden 10x volume increase
            data["volume"][anomaly_start:anomaly_end] *= 10

        elif anomaly_type == "price_manipulation":
            # Abnormal price pump and dump
            peak = anomaly_start + (anomaly_end - anomaly_start) // 2
            for i in range(anomaly_start, peak):
                data["price"][i] *= 1.05  # Pump
            for i in range(peak, anomaly_end):
                data["price"][i] *= 0.95  # Dump

        elif anomaly_type == "fee_evasion":
            # Suspiciously low fees despite high volume
            data["fee_rate"][anomaly_start:anomaly_end] *= 0.1
            data["volume"][anomaly_start:anomaly_end] *= 5

        return data

    def generate_dataset(
        self, num_normal: int = 800, num_anomalous: int = 200, seq_len: int = 100
    ) -> pd.DataFrame:
        """
        Generate a balanced dataset of normal and anomalous sequences.

        Args:
            num_normal: Number of normal sequences
            num_anomalous: Number of anomalous sequences
            seq_len: Length of each sequence

        Returns:
            DataFrame with sequences and labels
        """
        sequences = []
        labels = []

        # Generate normal sequences
        for _ in range(num_normal):
            data = self.generate_normal_sequence(length=seq_len)
            sequences.append(data)
            labels.append(0)  # Normal

        # Generate anomalous sequences
        anomaly_types = ["volume_spike", "price_manipulation", "fee_evasion"]
        for _ in range(num_anomalous):
            anomaly_type = np.random.choice(anomaly_types)
            data = self.generate_anomalous_sequence(
                length=seq_len, anomaly_type=anomaly_type
            )
            sequences.append(data)
            labels.append(1)  # Anomalous

        # Convert to DataFrame
        all_data = []
        for seq, label in zip(sequences, labels, strict=True):
            for t in range(seq_len):
                all_data.append(
                    {
                        "volume": seq["volume"][t],
                        "price": seq["price"][t],
                        "fee_rate": seq["fee_rate"][t],
                        "tx_count": seq["tx_count"][t],
                        "label": label,
                    }
                )

        return pd.DataFrame(all_data)

    def generate_ohlcv(
        self,
        n_bars: int = 2000,
        start_date: str = "2016-01-01",
        freq: str = "1min",
        dc_event_frequency: float = 0.0,
    ) -> pd.DataFrame:
        """Generate a single continuous OHLCV series for cascade training.

        Produces realistic-looking price action: geometric random walk with
        clustered volatility, synthetic spread for high/low, and volume that
        correlates with price movement magnitude.

        Args:
            n_bars: Number of bars to generate.
            start_date: ISO date string for the first bar's timestamp.
                        Defaults to "2016-01-01" to match the project-wide
                        historical data horizon.
            freq: Pandas frequency string matching the MetaApi timeframe
                  (default "1min" = 1-minute bars).
            dc_event_frequency: Probability per bar of injecting a move large
                enough to trigger a directional-change event. 0.0 means no
                injection (pure GARCH-lite). Use 0.3 in tests to stress-test
                DC signal paths.

        Returns DataFrame indexed by UTC timestamps with columns:
            open, high, low, close, price, volume, fee_rate
        """
        rng = np.random.default_rng(self.seed)

        # Geometric random walk with volatility clustering (GARCH-lite)
        vol = 0.005
        returns = np.zeros(n_bars)
        for i in range(1, n_bars):
            vol = 0.94 * vol + 0.06 * abs(returns[i - 1]) + 0.001
            returns[i] = rng.normal(0, vol)

        # Inject DC-triggering moves at the requested frequency.
        # 0.005 is well above all AssetConfig DC thresholds, so the DC state
        # machine will fire regardless of asset class.
        if dc_event_frequency > 0.0:
            _DC_INJECT_MAGNITUDE = 0.005
            mask = rng.random(n_bars) < dc_event_frequency
            directions = rng.choice(np.array([-1.0, 1.0]), n_bars)
            amplitudes = rng.uniform(1.5, 3.0, n_bars)
            returns = np.where(
                mask,
                directions * _DC_INJECT_MAGNITUDE * amplitudes,
                returns,
            )

        close = np.cumprod(1 + returns) * 100.0

        # High/low from spread proportional to realized move
        spread_frac = np.abs(returns) + rng.uniform(0.001, 0.003, n_bars)
        high = close * (1 + spread_frac)
        low = close * (1 - spread_frac)
        open_ = np.roll(close, 1)
        open_[0] = close[0]

        # Volume correlates with price movement magnitude
        volume = rng.uniform(1000, 5000, n_bars) * (1 + 2 * np.abs(returns) / max(vol, 1e-9))

        idx = pd.date_range(start=start_date, periods=n_bars, freq=freq, tz="UTC")
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "price": close,
                "volume": volume,
                "fee_rate": np.full(n_bars, 0.0002),
            },
            index=idx,
        )
