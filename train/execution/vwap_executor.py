"""
Volume-Weighted Average Price (VWAP) execution algorithm.

Splits large orders into smaller slices to reduce market impact:
- Spreads execution across time (e.g., 30 minutes)
- Weights slices by volume profile (trade more when market is liquid)
- Tracks VWAP vs arrival price for execution quality

Research basis: liquidatingforex.pdf
- VWAP reduces market impact for orders > 10 lots
- Participation rate: 5-15% of market volume
- Time horizon: 15-60 minutes typical for FX

Key concepts:
- Market impact: price moves against you when trading large size
- VWAP benchmark: industry standard for execution quality
- Volume profile: intraday volume patterns (higher at open/close)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class VWAPConfig:
    """Configuration for VWAP execution algorithm.

    Attributes:
        time_horizon: Number of time steps to spread order across (e.g., 30 minutes)
        participation_rate: Target % of market volume to trade (5-15% typical)
        min_slice_size: Minimum slice size (avoid tiny orders)
        max_slice_size: Maximum slice size (avoid market impact)
        volume_profile: Intraday volume pattern (if None, uses uniform distribution)
    """

    time_horizon: int = 30  # minutes
    participation_rate: float = 0.10  # 10% of market volume
    min_slice_size: float = 0.1  # 0.1 lots
    max_slice_size: float = 5.0  # 5 lots
    volume_profile: list[float] | None = None  # Custom volume curve


@dataclass
class VWAPSlice:
    """A single slice of a VWAP order.

    Attributes:
        time_step: When to execute this slice (relative to order start)
        size: Order size for this slice
        executed: Whether this slice has been filled
        fill_price: Actual fill price (None until executed)
    """

    time_step: int
    size: float
    executed: bool = False
    fill_price: float | None = None


class VWAPExecutor:
    """VWAP order execution algorithm.

    Slices large orders across time based on volume profile to minimize
    market impact while achieving fair execution prices.

    Example:
        config = VWAPConfig(time_horizon=30, participation_rate=0.10)
        executor = VWAPExecutor(config)

        # Start VWAP order for 20 lots
        executor.start_order(side='buy', total_size=20.0)

        # Each time step, get next slice
        for step in range(30):
            slice_info = executor.get_next_slice(current_volume=100.0)
            if slice_info:
                # Execute slice_info.size at market
                executor.record_execution(slice_info.time_step, fill_price=1.10050)

        # Check execution quality
        vwap = executor.get_vwap()
        arrival_price = 1.10000
        slippage = vwap - arrival_price  # Positive = overpaid
    """

    def __init__(self, config: VWAPConfig | None = None):
        """Initialize VWAP executor.

        Args:
            config: VWAP configuration (uses defaults if None)
        """
        self.config = config or VWAPConfig()

        # Active order state
        self.active = False
        self.side: str | None = None
        self.total_size = 0.0
        self.remaining_size = 0.0
        self.slices: list[VWAPSlice] = []
        self.current_step = 0
        self.arrival_price: float | None = None

        # Volume profile (weights for each time step)
        if self.config.volume_profile is not None:
            self.volume_profile = np.array(self.config.volume_profile)
            # Normalize to sum to 1
            self.volume_profile = self.volume_profile / self.volume_profile.sum()
        else:
            # Default: Uniform distribution
            self.volume_profile = np.ones(self.config.time_horizon) / self.config.time_horizon

    def start_order(
            self, side: str, total_size: float, arrival_price: float | None = None
    ) -> None:
        """Start a new VWAP order.

        Args:
            side: 'buy' or 'sell'
            total_size: Total order size to execute
            arrival_price: Price when order was submitted (for benchmarking)
        """
        if self.active:
            raise RuntimeError("VWAP order already active. Call reset() first.")

        self.active = True
        self.side = side
        self.total_size = total_size
        self.remaining_size = total_size
        self.arrival_price = arrival_price
        self.current_step = 0
        self.slices = []

    def get_next_slice(
            self, current_volume: float, current_step: int | None = None
    ) -> VWAPSlice | None:
        """Get next order slice based on volume and time.

        Args:
            current_volume: Current market volume (for participation rate)
            current_step: Current time step (if None, uses internal counter)

        Returns:
            VWAPSlice if order should be placed, None otherwise
        """
        if not self.active or self.remaining_size <= 0:
            return None

        if current_step is not None:
            self.current_step = current_step

        # Check if we're on the last step or beyond time horizon
        if self.current_step >= self.config.time_horizon - 1:
            # Final step: execute all remaining size to ensure completion
            slice_size = self.remaining_size
        else:
            # Calculate slice size based on:
            # 1. Volume profile weight for this time step
            # 2. Participation rate of market volume
            # 3. Remaining order size

            # Volume-weighted target
            profile_weight = self.volume_profile[min(self.current_step, len(self.volume_profile) - 1)]
            volume_based_size = current_volume * self.config.participation_rate * profile_weight

            # Proportional to remaining size
            proportion_remaining = self.remaining_size / self.total_size
            target_slice = volume_based_size * (1 + proportion_remaining)  # Catch up if behind

            # Clamp to min/max and remaining
            slice_size = np.clip(
                target_slice,
                self.config.min_slice_size,
                min(self.config.max_slice_size, self.remaining_size),
            )

        # Create slice
        slice_obj = VWAPSlice(time_step=self.current_step, size=slice_size)
        self.slices.append(slice_obj)
        self.remaining_size -= slice_size
        self.current_step += 1

        return slice_obj

    def record_execution(self, time_step: int, fill_price: float) -> None:
        """Record execution of a slice.

        Args:
            time_step: Time step that was executed
            fill_price: Actual fill price
        """
        for slice_obj in self.slices:
            if slice_obj.time_step == time_step and not slice_obj.executed:
                slice_obj.executed = True
                slice_obj.fill_price = fill_price
                return

    def get_vwap(self) -> float | None:
        """Calculate volume-weighted average price of executed slices.

        Returns:
            VWAP if slices have been executed, None otherwise
        """
        executed_slices = [s for s in self.slices if s.executed and s.fill_price is not None]

        if not executed_slices:
            return None

        total_quantity = sum(s.size for s in executed_slices)
        total_value = sum(s.size * s.fill_price for s in executed_slices)

        return total_value / total_quantity if total_quantity > 0 else None

    def get_completion_ratio(self) -> float:
        """Get order completion ratio (0.0 = not started, 1.0 = complete).

        Returns:
            Fraction of order executed
        """
        if self.total_size == 0:
            return 0.0
        return 1.0 - (self.remaining_size / self.total_size)

    def get_execution_quality(self, benchmark_price: float | None = None) -> dict:
        """Calculate execution quality metrics.

        Args:
            benchmark_price: Price to compare against (uses arrival_price if None)

        Returns:
            Dict with metrics:
            - vwap: Volume-weighted average price
            - slippage: VWAP - benchmark (positive = overpaid)
            - completion: Fraction of order completed
            - num_slices: Number of slices executed
        """
        vwap = self.get_vwap()
        completion = self.get_completion_ratio()
        num_slices = sum(1 for s in self.slices if s.executed)

        benchmark = benchmark_price or self.arrival_price
        slippage = (vwap - benchmark) if (vwap is not None and benchmark is not None) else None

        return {
            "vwap": vwap,
            "slippage": slippage,
            "completion": completion,
            "num_slices": num_slices,
        }

    def reset(self) -> None:
        """Reset executor state for new order."""
        self.active = False
        self.side = None
        self.total_size = 0.0
        self.remaining_size = 0.0
        self.slices = []
        self.current_step = 0
        self.arrival_price = None


# Example usage and testing
if __name__ == "__main__":
    print("Testing VWAP Executor...\n")

    # Create VWAP executor with 30-minute horizon
    config = VWAPConfig(
        time_horizon=30,
        participation_rate=0.10,  # 10% of market volume
        min_slice_size=0.5,
        max_slice_size=3.0,
    )
    executor = VWAPExecutor(config)

    # Simulate executing 20 lots
    print("=" * 60)
    print("Test Case: Execute 20 lots with VWAP")
    print("=" * 60)

    arrival_price = 1.10000
    executor.start_order(side='buy', total_size=20.0, arrival_price=arrival_price)

    print("Order: Buy 20 lots")
    print(f"Arrival price: {arrival_price:.5f}")
    print("Time horizon: 30 minutes")
    print()

    # Simulate 30 time steps
    np.random.seed(42)
    for step in range(30):
        # Simulate varying market volume (100 +/- 20 lots per minute)
        market_volume = 100.0 + np.random.randn() * 20.0

        # Get next slice
        slice_obj = executor.get_next_slice(current_volume=market_volume)

        if slice_obj:
            # Simulate execution with some price drift
            drift = np.random.randn() * 0.00002  # 0.2 pip std dev
            fill_price = arrival_price + drift + (step * 0.00001)  # Slight upward drift

            executor.record_execution(slice_obj.time_step, fill_price)

            if step < 5 or step % 10 == 0 or step >= 28:  # Print first 5, every 10th, and last 2
                print(f"Step {step:2d}: Execute {slice_obj.size:.2f} lots at {fill_price:.5f}")

    print()

    # Calculate execution quality
    quality = executor.get_execution_quality()
    print("=" * 60)
    print("Execution Quality:")
    print("=" * 60)
    print(f"VWAP: {quality['vwap']:.5f}")
    print(f"Arrival price: {arrival_price:.5f}")
    print(f"Slippage: {quality['slippage'] * 10000:+.2f} pips")
    print(f"Completion: {quality['completion']:.1%}")
    print(f"Number of slices: {quality['num_slices']}")
    print()

    # Slippage should be small (VWAP close to arrival)
    assert quality['completion'] == 1.0, "Order not fully completed"
    assert abs(quality['slippage']) < 0.0005, "Slippage too high"

    print("✓ VWAP executor tests complete!")
