"""Memory profiling utilities for dataset performance analysis.

Provides tools to monitor memory usage, compare Dataset implementations,
and identify memory bottlenecks in data loading pipelines.
"""

import gc
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import psutil
import torch


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: float
    rss_mb: float  # Resident Set Size (physical memory)
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    used_mb: float  # Used system memory


@dataclass
class ProfileResult:
    """Result of a memory profiling session."""
    function_name: str
    start_stats: MemoryStats
    end_stats: MemoryStats
    peak_stats: MemoryStats
    duration_s: float
    memory_delta_mb: float
    peak_delta_mb: float
    samples_processed: int = 0


class MemoryProfiler:
    """Advanced memory profiler for data loading operations."""

    def __init__(self, process: psutil.Process | None = None):
        self.process = process or psutil.Process(os.getpid())
        self.sampling_interval = 0.1  # seconds
        self.monitoring = False
        self.samples: list[MemoryStats] = []

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return MemoryStats(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=self.process.memory_percent(),
            available_mb=virtual_memory.available / 1024 / 1024,
            used_mb=virtual_memory.used / 1024 / 1024,
        )

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        self.monitoring = True
        self.samples = []

        def monitor():
            while self.monitoring:
                stats = self.get_memory_stats()
                self.samples.append(stats)
                time.sleep(self.sampling_interval)

        import threading
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> list[MemoryStats]:
        """Stop monitoring and return collected samples."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        return self.samples.copy()

    def profile_function(self, func: Callable, *args, **kwargs) -> ProfileResult:
        """Profile memory usage of a function."""
        # Force garbage collection before profiling
        gc.collect()

        start_stats = self.get_memory_stats()
        start_time = time.time()

        # Start monitoring
        self.start_monitoring()

        try:
            func(*args, **kwargs)
        finally:
            # Stop monitoring
            samples = self.stop_monitoring()

        end_time = time.time()
        end_stats = self.get_memory_stats()

        # Find peak memory usage
        peak_stats = start_stats
        if samples:
            max(s.rss_mb for s in samples)
            peak_stats = max(samples, key=lambda s: s.rss_mb)

        return ProfileResult(
            function_name=func.__name__ if hasattr(func, '__name__') else str(func),
            start_stats=start_stats,
            end_stats=end_stats,
            peak_stats=peak_stats,
            duration_s=end_time - start_time,
            memory_delta_mb=end_stats.rss_mb - start_stats.rss_mb,
            peak_delta_mb=peak_stats.rss_mb - start_stats.rss_mb,
        )

    def profile_dataset_iteration(self, dataset, max_samples: int = 1000) -> ProfileResult:
        """Profile iteration through a dataset."""

        def iterate():
            count = 0
            for _sample in dataset:
                count += 1
                if count >= max_samples:
                    break
            return count

        result = self.profile_function(iterate)
        result.samples_processed = min(max_samples, len(list(dataset)))
        return result

    def compare_datasets(self, datasets: dict[str, Any], max_samples: int = 1000) -> dict[str, ProfileResult]:
        """Compare memory usage of multiple datasets."""
        results = {}

        for name, dataset in datasets.items():
            print(f"Profiling {name}...")
            try:
                result = self.profile_dataset_iteration(dataset, max_samples)
                results[name] = result
            except Exception as e:
                print(f"Error profiling {name}: {e}")
                results[name] = None

        return results

    def generate_memory_report(self, results: dict[str, ProfileResult]) -> str:
        """Generate a comprehensive memory report."""
        report = []
        report.append("# Memory Profiling Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary table
        report.append("## Memory Usage Summary")
        report.append("| Dataset | Memory Delta (MB) | Peak Delta (MB) | Duration (s) | Samples/s |")
        report.append("|---------|-------------------|------------------|--------------|-----------|")

        for name, result in results.items():
            if result is not None:
                samples_per_sec = result.samples_processed / result.duration_s if result.duration_s > 0 else 0
                report.append(
                    f"| {name} | {result.memory_delta_mb:.1f} | {result.peak_delta_mb:.1f} | {result.duration_s:.2f} | {samples_per_sec:.1f} |")
            else:
                report.append(f"| {name} | ERROR | ERROR | ERROR | ERROR |")

        report.append("")

        # Detailed analysis
        report.append("## Detailed Analysis")

        valid_results = {k: v for k, v in results.items() if v is not None}

        if valid_results:
            # Find most and least efficient
            by_memory = sorted(valid_results.items(), key=lambda x: x[1].memory_delta_mb)
            by_speed = sorted(valid_results.items(), key=lambda x: x[1].samples_processed / x[1].duration_s,
                              reverse=True)

            report.append("### Most Memory Efficient")
            for name, result in by_memory[:2]:
                report.append(f"- **{name}**: {result.memory_delta_mb:.1f} MB delta")

            report.append("### Fastest Processing")
            for name, result in by_speed[:2]:
                speed = result.samples_processed / result.duration_s if result.duration_s > 0 else 0
                report.append(f"- **{name}**: {speed:.1f} samples/second")

            # Memory efficiency comparison
            if len(valid_results) >= 2:
                baseline = list(valid_results.values())[0]
                report.append("### Memory Efficiency vs Baseline")
                for name, result in valid_results.items():
                    if name != list(valid_results.keys())[0]:
                        efficiency = (
                                    baseline.memory_delta_mb / result.memory_delta_mb) if result.memory_delta_mb > 0 else 0
                        report.append(f"- **{name}**: {efficiency:.2f}x more efficient")

        return "\n".join(report)

    def plot_memory_usage(self, samples: list[MemoryStats], title: str = "Memory Usage Over Time"):
        """Plot memory usage over time."""
        if not samples:
            print("No samples to plot")
            return

        timestamps = [(s.timestamp - samples[0].timestamp) for s in samples]
        rss_values = [s.rss_mb for s in samples]

        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, rss_values, linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_path = Path("memory_usage_plot.png")
        plt.savefig(output_path)
        plt.close()

        print(f"Memory usage plot saved to: {output_path}")
        return output_path


class DataLoaderProfiler:
    """Specialized profiler for PyTorch DataLoader operations."""

    def __init__(self):
        self.profiler = MemoryProfiler()

    def profile_dataloader_batches(self, dataloader, num_batches: int = 10) -> dict[str, Any]:
        """Profile DataLoader batch processing."""

        def process_batches():
            batch_times = []
            memory_samples = []

            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                batch_start = time.time()

                # Simulate some processing
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    sequences, targets = batch[0], batch[1]
                    # Perform some tensor operations to simulate model processing
                    _ = sequences.sum()
                    for _key, target in targets.items():
                        if hasattr(target, 'sum'):
                            _ = target.sum()

                batch_end = time.time()
                batch_times.append(batch_end - batch_start)

                # Sample memory after each batch
                memory_samples.append(self.profiler.get_memory_stats())

            return batch_times, memory_samples

        # Profile the batch processing
        batch_times, memory_samples = process_batches()

        return {
            'batch_times': batch_times,
            'memory_samples': memory_samples,
            'avg_batch_time': sum(batch_times) / len(batch_times),
            'total_time': sum(batch_times),
            'batches_processed': len(batch_times),
        }

    def compare_dataloaders(self, dataloaders: dict[str, torch.utils.data.DataLoader]) -> dict[str, dict[str, Any]]:
        """Compare multiple DataLoaders."""
        results = {}

        for name, dataloader in dataloaders.items():
            print(f"Profiling DataLoader: {name}")
            try:
                stats = self.profile_dataloader_batches(dataloader)
                results[name] = stats
            except Exception as e:
                print(f"Error profiling {name}: {e}")
                results[name] = None

        return results


@contextmanager
def memory_monitor(interval: float = 0.1):
    """Context manager for temporary memory monitoring."""
    profiler = MemoryProfiler()
    profiler.sampling_interval = interval
    profiler.start_monitoring()

    try:
        yield profiler
    finally:
        samples = profiler.stop_monitoring()

        if samples:
            # Calculate summary statistics
            rss_values = [s.rss_mb for s in samples]
            print("\nMemory monitoring summary:")
            print(f"  Samples: {len(samples)}")
            print(f"  Duration: {samples[-1].timestamp - samples[0].timestamp:.2f}s")
            print(f"  Min RSS: {min(rss_values):.1f} MB")
            print(f"  Max RSS: {max(rss_values):.1f} MB")
            print(f"  Delta: {rss_values[-1] - rss_values[0]:.1f} MB")


def profile_data_pipeline(data_func: Callable, *args, **kwargs) -> ProfileResult:
    """Convenience function to profile a data pipeline function."""
    profiler = MemoryProfiler()
    return profiler.profile_function(data_func, *args, **kwargs)


def create_memory_report(results: dict[str, ProfileResult], output_path: str = "memory_report.md"):
    """Create a memory profiling report file."""
    profiler = MemoryProfiler()
    report = profiler.generate_memory_report(results)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Memory report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example usage
    print("Memory Profiling Utilities")
    print("==========================")


    # Test with sample data
    def sample_data_function():
        """Sample function to profile."""
        import numpy as np
        data = np.random.randn(1000, 100)
        result = data.sum(axis=1)
        return result


    # Profile the function
    result = profile_data_pipeline(sample_data_function)
    print(f"Function: {result.function_name}")
    print(f"Memory delta: {result.memory_delta_mb:.2f} MB")
    print(f"Duration: {result.duration_s:.2f} seconds")

    # Create a simple report
    results = {"sample_function": result}
    create_memory_report(results)
