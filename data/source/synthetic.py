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
        for seq, label in zip(sequences, labels):
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
