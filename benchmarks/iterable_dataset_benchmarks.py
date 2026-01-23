"""Performance benchmarks comparing IterableFXDataset vs standard Dataset.

Benchmarks memory usage, processing speed, and scalability for different dataset sizes.
"""

import gc
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from torch.utils.data import DataLoader

from config.config import DataConfig, FeatureConfig
from data.agents.single_task_agent import SingleTaskDataAgent
from data.iterable_dataset import IterableFXDataset


class MemoryProfiler:
    """Utility class for memory profiling."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def profile_function(self, func, *args, **kwargs) -> tuple[any, dict[str, float]]:
        """Profile memory and time usage of a function."""
        gc.collect()  # Clean up before measurement

        memory_before = self.get_memory_usage()
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        memory_after = self.get_memory_usage()

        stats = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_after - memory_before,
            'execution_time_s': end_time - start_time,
            'peak_memory_mb': memory_after  # Simplified peak detection
        }

        return result, stats


class DatasetBenchmark:
    """Benchmark comparing IterableFXDataset vs standard Dataset."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.profiler = MemoryProfiler()

    def create_test_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create test dataset of specified size."""
        np.random.seed(42)

        # Create realistic OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')

        # Generate random walk prices
        base_price = 1.1000
        returns = np.random.normal(0, 0.0001, n_samples)
        log_prices = np.log(base_price) + np.cumsum(returns)
        prices = np.exp(log_prices)

        data = {
            'datetime': dates,
            'open': prices * (1 + np.random.normal(0, 0.0005, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, n_samples))),
            'close': prices,
            'volume': np.random.randint(100, 1000, n_samples),
        }

        df = pd.DataFrame(data)
        df = df.round(5)

        # Add technical indicators
        df['rsi'] = np.random.uniform(20, 80, n_samples)
        df['macd'] = np.random.normal(0, 0.001, n_samples)
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()

        return df

    def create_configs(self) -> tuple[DataConfig, FeatureConfig]:
        """Create test configurations."""
        data_cfg = DataConfig(
            datetime_column='datetime',
            t_in=60,
            t_out=10,
            lookahead_window=20,
            train_range=('2024-01-01', '2024-01-02'),
            val_range=('2024-01-02', '2024-01-03'),
            test_range=('2024-01-03', '2024-01-04'),
            feature_columns=None,
            target_type='classification',
            flat_threshold=0.001,
            top_k_predictions=3,
            predict_sell_now=True,
        )

        feature_cfg = FeatureConfig(
            sma_windows=[10, 20, 50],
            ema_windows=[10, 20, 50],
            rsi_window=14,
            bollinger_window=20,
            bollinger_num_std=2.0,
            atr_window=14,
            short_vol_window=10,
            long_vol_window=50,
            spread_windows=[20],
            imbalance_smoothing=5,
        )

        return data_cfg, feature_cfg

    def benchmark_standard_dataset(self, data_cfg: DataConfig, feature_df: pd.DataFrame) -> tuple[
        DataLoader, dict[str, float]]:
        """Benchmark standard Dataset approach."""

        agent = SingleTaskDataAgent(data_cfg)
        datasets = agent.build_datasets(feature_df)

        def create_dataloader():
            return agent.build_dataloaders(datasets, batch_size=32, num_workers=0)

        dataloader, stats = self.profiler.profile_function(create_dataloader)

        # Measure data loading time
        def load_batch():
            for i, _batch in enumerate(dataloader['train']):
                if i >= 10:  # Load first 10 batches
                    break
            return True

        _, loading_stats = self.profiler.profile_function(load_batch)
        stats.update({'data_loading_time_s': loading_stats['execution_time_s']})

        return dataloader, stats

    def benchmark_iterable_dataset(self, data_cfg: DataConfig, feature_cfg: FeatureConfig,
                                   cache_path: Path) -> tuple[IterableFXDataset, dict[str, float]]:
        """Benchmark IterableFXDataset approach."""
        from unittest.mock import patch

        # Mock cache file finding
        def mock_find_cached_features():
            return cache_path

        with patch.object(IterableFXDataset, '_find_cached_features', mock_find_cached_features):
            def create_iterable_dataset():
                return IterableFXDataset(
                    pair="test",
                    data_cfg=data_cfg,
                    feature_cfg=feature_cfg,
                    input_root=self.temp_dir,
                    cache_dir=self.temp_dir,
                    chunksize=1000,
                )

            dataset, stats = self.profiler.profile_function(create_iterable_dataset)

            # Measure iteration time
            def iterate_dataset():
                count = 0
                for _sample in dataset:
                    count += 1
                    if count >= 100:  # Iterate first 100 samples
                        break
                return count

            _, iteration_stats = self.profiler.profile_function(iterate_dataset)
            stats.update({'iteration_time_s': iteration_stats['execution_time_s']})
            stats.update({'samples_iterated': iteration_stats.get('samples_iterated', 100)})

            return dataset, stats

    def benchmark_dataset_sizes(self, sizes: list[int]) -> dict[int, dict[str, dict[str, float]]]:
        """Benchmark both approaches with different dataset sizes."""
        data_cfg, feature_cfg = self.create_configs()
        results = {}

        for size in sizes:
            print(f"\nBenchmarking dataset size: {size} samples")

            # Create test data
            feature_df = self.create_test_data(size)
            cache_path = self.temp_dir / f"test_features_{size}.feather"
            feature_df.to_feather(cache_path)

            size_results = {}

            # Benchmark standard dataset
            try:
                dataloader, std_stats = self.benchmark_standard_dataset(data_cfg, feature_df)
                size_results['standard_dataset'] = std_stats
                print(f"  Standard Dataset: {std_stats['memory_used_mb']:.1f} MB, {std_stats['execution_time_s']:.2f}s")
            except Exception as e:
                print(f"  Standard Dataset: Failed - {e}")
                size_results['standard_dataset'] = {'error': str(e)}

            # Benchmark iterable dataset
            try:
                iterable_dataset, iter_stats = self.benchmark_iterable_dataset(
                    data_cfg, feature_cfg, cache_path
                )
                size_results['iterable_dataset'] = iter_stats
                print(
                    f"  Iterable Dataset: {iter_stats['memory_used_mb']:.1f} MB, {iter_stats['execution_time_s']:.2f}s")
            except Exception as e:
                print(f"  Iterable Dataset: Failed - {e}")
                size_results['iterable_dataset'] = {'error': str(e)}

            results[size] = size_results

            # Clean up
            if cache_path.exists():
                cache_path.unlink()

            # Force garbage collection between tests
            gc.collect()

        return results

    def benchmark_chunksize_impact(self) -> dict[str, dict[str, float]]:
        """Benchmark impact of different chunk sizes."""
        data_cfg, feature_cfg = self.create_configs()

        # Create medium-sized test data
        feature_df = self.create_test_data(2000)
        cache_path = self.temp_dir / "chunksize_test.feather"
        feature_df.to_feather(cache_path)

        chunk_sizes = [100, 500, 1000, 2000, 5000]
        results = {}

        for chunksize in chunk_sizes:
            print(f"\nBenchmarking chunksize: {chunksize}")

            from unittest.mock import patch

            def mock_find_cached_features():
                return cache_path

            with patch.object(IterableFXDataset, '_find_cached_features', mock_find_cached_features):
                def create_and_iterate():
                    dataset = IterableFXDataset(
                        pair="test",
                        data_cfg=data_cfg,
                        feature_cfg=feature_cfg,
                        input_root=self.temp_dir,
                        cache_dir=self.temp_dir,
                        chunksize=chunksize,
                    )
                    # Iterate through all samples
                    count = 0
                    for _sample in dataset:
                        count += 1
                    return count

                count, stats = self.profiler.profile_function(create_and_iterate)
                stats['samples_processed'] = count
                results[f'chunksize_{chunksize}'] = stats

                print(
                    f"  Processed {count} samples in {stats['execution_time_s']:.2f}s using {stats['memory_used_mb']:.1f} MB")

        # Clean up
        if cache_path.exists():
            cache_path.unlink()

        return results

    def generate_report(self, size_results: dict, chunksize_results: dict) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# IterableFXDataset Performance Benchmark Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Dataset size comparison
        report.append("## Dataset Size Comparison")
        report.append("| Size | Standard Dataset (MB, s) | Iterable Dataset (MB, s) | Memory Savings | Speed Ratio |")
        report.append("|------|---------------------------|---------------------------|----------------|-------------|")

        for size in sorted(size_results.keys()):
            std_result = size_results[size].get('standard_dataset', {})
            iter_result = size_results[size].get('iterable_dataset', {})

            if 'error' not in std_result and 'error' not in iter_result:
                std_mem = std_result.get('memory_used_mb', 0)
                std_time = std_result.get('execution_time_s', 0)
                iter_mem = iter_result.get('memory_used_mb', 0)
                iter_time = iter_result.get('execution_time_s', 0)

                memory_savings = ((std_mem - iter_mem) / std_mem * 100) if std_mem > 0 else 0
                speed_ratio = (std_time / iter_time) if iter_time > 0 else 0

                report.append(
                    f"| {size:,} | {std_mem:.1f}, {std_time:.2f} | {iter_mem:.1f}, {iter_time:.2f} | {memory_savings:.1f}% | {speed_ratio:.2f}x |")
            else:
                report.append(f"| {size:,} | ERROR | ERROR | - | - |")

        report.append("")

        # Chunk size impact
        report.append("## Chunk Size Impact")
        report.append("| Chunksize | Memory (MB) | Time (s) | Samples/s |")
        report.append("|-----------|-------------|----------|-----------|")

        for chunksize_name, result in chunksize_results.items():
            chunksize = chunksize_name.split('_')[1]
            memory = result.get('memory_used_mb', 0)
            time_s = result.get('execution_time_s', 0)
            samples = result.get('samples_processed', 0)
            samples_per_sec = samples / time_s if time_s > 0 else 0

            report.append(f"| {chunksize} | {memory:.1f} | {time_s:.2f} | {samples_per_sec:.1f} |")

        report.append("")

        # Summary
        report.append("## Summary")

        # Calculate averages
        memory_savings_list = []
        speed_ratios_list = []

        for size in size_results:
            std_result = size_results[size].get('standard_dataset', {})
            iter_result = size_results[size].get('iterable_dataset', {})

            if 'error' not in std_result and 'error' not in iter_result:
                std_mem = std_result.get('memory_used_mb', 0)
                iter_mem = iter_result.get('memory_used_mb', 0)
                std_time = std_result.get('execution_time_s', 0)
                iter_time = iter_result.get('execution_time_s', 0)

                if std_mem > 0:
                    memory_savings = ((std_mem - iter_mem) / std_mem * 100)
                    memory_savings_list.append(memory_savings)

                if iter_time > 0:
                    speed_ratio = std_time / iter_time
                    speed_ratios_list.append(speed_ratio)

        if memory_savings_list:
            avg_memory_savings = np.mean(memory_savings_list)
            report.append(f"- Average memory savings: {avg_memory_savings:.1f}%")

        if speed_ratios_list:
            avg_speed_ratio = np.mean(speed_ratios_list)
            report.append(f"- Average speed ratio: {avg_speed_ratio:.2f}x")

        report.append("- IterableFXDataset provides memory-efficient streaming for large datasets")
        report.append("- Chunk size should be tuned based on available memory and dataset characteristics")

        return "\n".join(report)


def main():
    """Run all benchmarks and generate report."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        benchmark = DatasetBenchmark(temp_path)

        print("Starting IterableFXDataset performance benchmarks...")

        # Benchmark different dataset sizes
        sizes = [500, 1000, 2000, 5000]
        print("\n=== Dataset Size Benchmarks ===")
        size_results = benchmark.benchmark_dataset_sizes(sizes)

        # Benchmark chunk size impact
        print("\n=== Chunk Size Impact Benchmarks ===")
        chunksize_results = benchmark.benchmark_chunksize_impact()

        # Generate report
        print("\n=== Generating Report ===")
        report = benchmark.generate_report(size_results, chunksize_results)

        # Save report
        report_path = Path("docs/phase3_5_memory_analysis.md")
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nBenchmark report saved to: {report_path}")
        print("\nReport contents:")
        print(report)


if __name__ == "__main__":
    main()
