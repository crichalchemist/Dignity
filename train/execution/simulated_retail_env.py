"""Simulated retail execution environment.

This module provides a gym-like interface for reinforcement-learning agents
that want to simulate order execution under typical retail conditions.
The environment is intentionally lightweight (no gym dependency) while
exposing familiar ``reset``/``step`` semantics.

Key features
------------
* Spread-inclusive execution with configurable slippage distribution.
* Limit-order fill probabilities with optional delays and FIFO position
  management for realized PnL.
* Logging hooks for execution metrics that are important for safety
  (fills, slippage, and inventory drift).

Security considerations
-----------------------
The environment avoids any external I/O and keeps stochastic elements seeded
through ``numpy.random.Generator``. Users should avoid passing untrusted
objects for callbacks or loggers; the module only expects standard logging
interfaces.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from train.execution.limit_order_engine import LimitOrderConfig, LimitOrderEngine


@dataclass
class SlippageModel:
    """Simple slippage model returning price-impact proportions.

    The sampled slippage is expressed as a proportion of the current mid price.
    A value of ``0.001`` means a 0.1% move adverse to the trade direction.
    """

    mean: float = 0.0005
    std: float = 0.0007
    max_slippage: float = 0.003

    def sample(self, rng: np.random.Generator) -> float:
        """Return a clipped slippage sample."""

        raw = rng.normal(self.mean, self.std)
        return float(np.clip(raw, -self.max_slippage, self.max_slippage))


@dataclass
class ExecutionConfig:
    """Configuration for the simulated retail execution environment."""

    initial_mid_price: float = 100.0
    initial_cash: float = 50_000.0
    spread: float = 0.02  # dollars (base spread)
    price_drift: float = 0.0
    price_volatility: float = 0.05
    time_horizon: int = 390  # minutes in a trading day
    decision_lag: int = 1  # steps to wait before acting
    lot_size: float = 1.0
    limit_fill_probability: float = 0.35
    limit_price_improvement: float = 0.01
    slippage_model: SlippageModel = field(default_factory=SlippageModel)

    # Phase 3: Transaction costs
    commission_per_lot: float = 0.0  # Commission per lot (e.g., $7 per 100k lot in FX)
    commission_pct: float = 0.0  # Commission as % of notional (alternative to per-lot)
    variable_spread: bool = False  # Enable volatility-dependent spread widening
    spread_volatility_multiplier: float = 2.0  # Spread multiplier during high volatility

    # Phase 3.3: Risk management
    enable_stop_loss: bool = False  # Enable automatic stop-loss exits
    stop_loss_pct: float = 0.02  # Stop loss as % of entry price (2% default)
    enable_take_profit: bool = False  # Enable automatic take-profit exits
    take_profit_pct: float = 0.04  # Take profit as % of entry price (4% default)
    max_drawdown_pct: float = 0.20  # Maximum portfolio drawdown before episode termination (20%)

    # Week 2: Intelligent limit order execution
    use_limit_order_engine: bool = False  # Enable intelligent limit order pricing
    limit_order_config: LimitOrderConfig | None = None  # Limit order engine configuration
    enable_drawdown_limit: bool = False  # Enable drawdown-based episode termination

    # Phase 4: Reward engineering
    reward_type: str = "incremental_pnl"  # "portfolio_value", "incremental_pnl", "sharpe", "cost_aware"
    cost_penalty_weight: float = 0.0  # Weight for transaction cost penalty (0.0 = no penalty)
    drawdown_penalty_weight: float = 0.0  # Weight for drawdown penalty (0.0 = no penalty)
    sharpe_window: int = 50  # Window size for Sharpe ratio calculation
    reward_scaling: float = 1e-4  # Scale reward to improve training stability

    def validate(self) -> None:
        if self.spread < 0:
            raise ValueError("Spread must be non-negative")
        if not 0.0 <= self.limit_fill_probability <= 1.0:
            raise ValueError("Limit fill probability must be within [0, 1]")
        if self.time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
        if self.decision_lag < 0:
            raise ValueError("Decision lag cannot be negative")
        if self.lot_size <= 0:
            raise ValueError("Lot size must be positive")
        if self.price_volatility < 0:
            raise ValueError("Price volatility must be non-negative")
        if self.initial_cash < 0:
            raise ValueError("Initial cash must be non-negative")
        if self.reward_type not in ["portfolio_value", "incremental_pnl", "sharpe", "cost_aware"]:
            raise ValueError(f"Invalid reward_type: {self.reward_type}")
        if self.sharpe_window <= 0:
            raise ValueError("sharpe_window must be positive")


@dataclass
class OrderAction:
    """Order action supplied to :meth:`SimulatedRetailExecutionEnv.step`."""

    action_type: str  # "market", "limit", or "hold"
    side: str  # "buy" or "sell"
    size: float
    limit_price: float | None = None
    aggressiveness: float | None = None  # For limit order engine (0=passive, 1=aggressive)

    def normalized(self, config: ExecutionConfig) -> OrderAction:
        size = max(self.size, 0.0)
        size = round(size / config.lot_size) * config.lot_size
        limit_price = self.limit_price
        action_type = self.action_type.lower()
        side = self.side.lower()
        if action_type not in {"market", "limit", "hold"}:
            raise ValueError(f"Unsupported action_type '{self.action_type}'")
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side '{self.side}'")
        if limit_price is not None and limit_price <= 0:
            raise ValueError("Limit price must be positive when provided")
        return OrderAction(action_type, side, size, limit_price)


@dataclass
class Position:
    quantity: float
    entry_price: float


class ActionSpace:
    """Minimal gym-like action space for sampling."""

    def __init__(self, lot_size: float, rng: np.random.Generator):
        self.lot_size = lot_size
        self._rng = rng

    def sample(self) -> OrderAction:
        side = "buy" if self._rng.random() < 0.5 else "sell"
        action_type = "market" if self._rng.random() < 0.6 else "limit"
        size = (self._rng.integers(1, 4) * self.lot_size)
        return OrderAction(action_type, side, size)


class ObservationSpace:
    """Minimal observation space stub for compatibility."""

    keys = (
        "mid_price",
        "inventory",
        "pending_orders",
        "cash",
        "realized_pnl",
        "unrealized_pnl",
        "portfolio_value",
    )

    def sample(self) -> dict[str, float]:
        return dict.fromkeys(self.keys, 0.0)


class SimulatedRetailExecutionEnv:
    """A lightweight execution environment for RL agents.

    The environment does not depend on ``gym`` but mirrors its API surface with
    ``reset`` and ``step`` methods. It simulates mid-price evolution, spread
    costs, slippage, limit-order fills, action delays, and FIFO position
    management. Metrics are logged via the standard ``logging`` module.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            config: ExecutionConfig | None = None,
            seed: int | None = None,
            logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or ExecutionConfig()
        self.config.validate()
        self.rng = np.random.default_rng(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.action_space = ActionSpace(self.config.lot_size, self.rng)
        self.observation_space = ObservationSpace()

        self._action_buffer: deque[OrderAction] = deque()
        self._pending_limits: list[OrderAction] = []
        self.positions: deque[Position] = deque()

        self.step_count = 0
        self.mid_price = self.config.initial_mid_price
        self.cash = self.config.initial_cash
        self.realized_pnl = 0.0
        self.inventory = 0.0
        self._slippage_paid = 0.0
        self._spread_paid = 0.0
        self._commission_paid = 0.0  # Phase 3: Track commission costs
        self._fill_events: list[dict[str, float]] = []
        self._recent_volatility = self.config.price_volatility  # Track for variable spreads

        # Phase 3.3: Risk management tracking
        self._peak_portfolio_value = self.config.initial_cash  # Track peak for drawdown
        self._stop_loss_triggered = 0  # Count stop-loss exits
        self._take_profit_triggered = 0  # Count take-profit exits

        # Phase 4: Reward engineering tracking
        self._previous_portfolio_value = self.config.initial_cash  # For incremental rewards
        self._portfolio_value_history: list[float] = []  # For Sharpe ratio calculation
        self._costs_at_last_step = 0.0  # Track total costs for cost-aware rewards

        # Week 2: Intelligent limit order engine
        if self.config.use_limit_order_engine:
            limit_config = self.config.limit_order_config or LimitOrderConfig()
            self.limit_order_engine = LimitOrderEngine(limit_config)
            self._limit_order_savings = 0.0  # Track cost savings from limit orders
        else:
            self.limit_order_engine = None
            self._limit_order_savings = 0.0

    def reset(self) -> dict[str, float]:
        """Reset the environment and return the initial observation."""

        self._action_buffer.clear()
        self._pending_limits.clear()
        self.positions.clear()
        self.step_count = 0
        self.mid_price = self.config.initial_mid_price
        self.cash = self.config.initial_cash
        self.realized_pnl = 0.0
        self.inventory = 0.0
        self._slippage_paid = 0.0
        self._spread_paid = 0.0
        self._commission_paid = 0.0
        self._recent_volatility = self.config.price_volatility
        self._peak_portfolio_value = self.config.initial_cash
        self._stop_loss_triggered = 0
        self._take_profit_triggered = 0
        self._fill_events.clear()
        self._previous_portfolio_value = self.config.initial_cash
        self._portfolio_value_history.clear()
        self._costs_at_last_step = 0.0
        self._limit_order_savings = 0.0
        return self._observation()

    def step(self, action: OrderAction | None):
        """Advance the simulation.

        Parameters
        ----------
        action:
            The order instruction to enqueue. A decision lag may delay when the
            action is actually applied.
        """

        if self.step_count >= self.config.time_horizon:
            raise RuntimeError("Environment already terminated; call reset().")

        normalized_action = (
            action.normalized(self.config) if action is not None else None
        )
        self._enqueue_action(normalized_action)
        executable = self._dequeue_action()
        if executable is not None:
            self._apply_action(executable)

        self._advance_mid_price()
        self._try_fill_limits()

        # Phase 3.3: Check for stop-loss and take-profit triggers
        self._check_stop_loss_take_profit()

        self.step_count += 1
        obs = self._observation()

        # Phase 3.3: Check for drawdown-based termination
        drawdown_exceeded = self._check_drawdown(obs["portfolio_value"])
        done = self.step_count >= self.config.time_horizon or drawdown_exceeded

        # Phase 4: Calculate reward using configured reward function
        reward = self._calculate_reward(obs["portfolio_value"])

        # Update tracking variables for next step
        self._previous_portfolio_value = obs["portfolio_value"]
        self._costs_at_last_step = self._commission_paid + self._spread_paid + self._slippage_paid

        if done:
            self._log_metrics()

        return obs, reward, done, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.logger.info("Step %s | Mid %.4f | Inv %.2f | PnL %.2f", self.step_count, self.mid_price,
                             self.inventory, self.realized_pnl)

    # Internal mechanics

    def _enqueue_action(self, action: OrderAction | None) -> None:
        if action is not None:
            self._action_buffer.append(action)

    def _dequeue_action(self) -> OrderAction | None:
        if self.config.decision_lag == 0:
            return self._action_buffer.pop() if self._action_buffer else None
        if len(self._action_buffer) > self.config.decision_lag - 1:
            return self._action_buffer.popleft()
        return None

    def _advance_mid_price(self) -> None:
        drift = self.config.price_drift
        shock = self.rng.normal(0.0, self.config.price_volatility)
        self.mid_price = max(0.01, self.mid_price * (1.0 + drift + shock))

        # Phase 3: Track recent volatility with exponential moving average
        # Use the absolute shock as a proxy for instantaneous volatility
        alpha = 0.1  # Smoothing factor for EMA
        self._recent_volatility = (
                alpha * abs(shock) + (1 - alpha) * self._recent_volatility
        )

    def _apply_action(self, action: OrderAction) -> None:
        if action.action_type == "hold":
            return

        # Week 2.3: Use limit order engine with learned aggressiveness
        if (
                action.action_type == "market"
                and self.limit_order_engine is not None
                and action.aggressiveness is not None
        ):
            # Use limit order engine to compute optimal limit price
            self._get_current_spread() / 2.0
            limit_price, fill_prob, expected_savings = self.limit_order_engine.place_limit_order(
                side=action.side,
                mid_price=self.mid_price,
                spread=self._get_current_spread(),
                volatility=self._recent_volatility,
                inventory=self.inventory,
                order_size=action.size,
                cash=self.cash,
                aggressiveness_override=action.aggressiveness,
            )

            # Submit as limit order with computed price
            limit_action = OrderAction("limit", action.side, action.size, limit_price)
            self._submit_limit(limit_action)
            self._limit_order_savings += expected_savings * action.size
        elif action.action_type == "market":
            self._execute_market(action)
        elif action.action_type == "limit":
            self._submit_limit(action)
        else:
            self.logger.warning("Unsupported action_type '%s'", action.action_type)

    def _get_current_spread(self) -> float:
        """Get current bid-ask spread (may be wider during high volatility)."""
        base_spread = self.config.spread
        if self.config.variable_spread:
            # Widen spread during high volatility
            volatility_ratio = self._recent_volatility / max(self.config.price_volatility, 1e-9)
            if volatility_ratio > 1.5:  # High volatility threshold
                base_spread *= self.config.spread_volatility_multiplier
        return base_spread

    def _execution_price(self, side: str, size: float) -> float:
        direction = 1.0 if side == "buy" else -1.0

        # Phase 3: Variable spread based on volatility
        base_spread = self._get_current_spread()
        half_spread = base_spread / 2.0
        slippage = self.config.slippage_model.sample(self.rng) * self.mid_price
        exec_price = self.mid_price + direction * (half_spread + slippage)
        spread_cost = half_spread * 2 * size  # distance between bid/ask around mid scaled by size
        self._spread_paid += spread_cost
        self._slippage_paid += abs(slippage) * size
        return exec_price

    def _execute_market(self, action: OrderAction) -> None:
        price = self._execution_price(action.side, action.size)
        self._fill(order_type="market", side=action.side, size=action.size, price=price)

    def _submit_limit(self, action: OrderAction) -> None:
        limit_price = action.limit_price
        if limit_price is None:
            direction = 1.0 if action.side == "buy" else -1.0
            limit_price = self.mid_price - direction * self.config.limit_price_improvement
        limit_action = OrderAction("limit", action.side, action.size, limit_price)
        self._pending_limits.append(limit_action)

    def _try_fill_limits(self) -> None:
        if not self._pending_limits:
            return

        remaining_limits: list[OrderAction] = []
        for limit_order in self._pending_limits:
            should_fill = self._should_fill_limit(limit_order)
            if should_fill:
                self._fill(
                    order_type="limit",
                    side=limit_order.side,
                    size=limit_order.size,
                    price=limit_order.limit_price,
                )
            else:
                remaining_limits.append(limit_order)
        self._pending_limits = remaining_limits

    def _should_fill_limit(self, limit_order: OrderAction) -> bool:
        crossed = False
        if limit_order.side == "buy" and self.mid_price <= limit_order.limit_price:
            crossed = True
        if limit_order.side == "sell" and self.mid_price >= limit_order.limit_price:
            crossed = True
        if not crossed:
            return False

        # Week 2: Use intelligent fill probability if engine is enabled
        if self.limit_order_engine is not None:
            # Compute market price (what we'd pay with market order)
            half_spread = self._get_current_spread() / 2.0
            if limit_order.side == "buy":
                market_price = self.mid_price + half_spread
            else:
                market_price = self.mid_price - half_spread

            # Use limit order engine to estimate fill probability
            fill_prob = self.limit_order_engine._estimate_fill_probability(
                limit_price=limit_order.limit_price,
                market_price=market_price,
                mid_price=self.mid_price,
                volatility=self._recent_volatility,
                side=limit_order.side,
            )
        else:
            # Fall back to simple fixed probability
            fill_prob = self.config.limit_fill_probability

        return bool(self.rng.random() <= fill_prob)

    def _check_stop_loss_take_profit(self) -> None:
        """
        Phase 3.3: Check if any positions should be closed due to stop-loss or take-profit.

        For each open position, calculate unrealized PnL as a percentage of entry price.
        If stop-loss or take-profit thresholds are breached, close the position at market.
        """
        if not self.positions:
            return

        if not (self.config.enable_stop_loss or self.config.enable_take_profit):
            return

        # Check each position for stop/take-profit triggers
        positions_to_close = []
        for pos in self.positions:
            # Calculate unrealized PnL percentage
            if pos.quantity > 0:  # Long position
                pnl_pct = (self.mid_price - pos.entry_price) / pos.entry_price
            else:  # Short position
                pnl_pct = (pos.entry_price - self.mid_price) / pos.entry_price

            # Check stop-loss (negative PnL exceeds threshold)
            if self.config.enable_stop_loss and pnl_pct <= -self.config.stop_loss_pct:
                positions_to_close.append((pos, "stop_loss"))

            # Check take-profit (positive PnL exceeds threshold)
            elif self.config.enable_take_profit and pnl_pct >= self.config.take_profit_pct:
                positions_to_close.append((pos, "take_profit"))

        # Close triggered positions
        for pos, trigger_type in positions_to_close:
            side = "sell" if pos.quantity > 0 else "buy"
            size = abs(pos.quantity)

            # Execute market order to close position
            price = self._execution_price(side, size)
            self._fill(order_type=trigger_type, side=side, size=size, price=price)

            if trigger_type == "stop_loss":
                self._stop_loss_triggered += 1
                self.logger.info(
                    "Stop-loss triggered at step %d | closed %.2f at %.4f",
                    self.step_count, size, price
                )
            else:
                self._take_profit_triggered += 1
                self.logger.info(
                    "Take-profit triggered at step %d | closed %.2f at %.4f",
                    self.step_count, size, price
                )

    def _check_drawdown(self, portfolio_value: float) -> bool:
        """
        Phase 3.3: Check if portfolio drawdown exceeds maximum threshold.

        Args:
            portfolio_value: Current total portfolio value

        Returns:
            True if drawdown limit exceeded (episode should terminate)
        """
        if not self.config.enable_drawdown_limit:
            return False

        # Update peak portfolio value
        if portfolio_value > self._peak_portfolio_value:
            self._peak_portfolio_value = portfolio_value

        # Calculate current drawdown
        if self._peak_portfolio_value > 0:
            drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value
        else:
            drawdown = 0.0

        # Check if drawdown exceeds limit
        if drawdown >= self.config.max_drawdown_pct:
            self.logger.warning(
                "Drawdown limit exceeded at step %d | drawdown=%.2f%% | peak=%.2f | current=%.2f",
                self.step_count, drawdown * 100, self._peak_portfolio_value, portfolio_value
            )
            return True

        return False

    def _fill(self, order_type: str, side: str, size: float, price: float) -> None:
        if size <= 0:
            return
        signed_size = size if side == "buy" else -size
        self._apply_fifo_fill(signed_size, price)
        self.cash -= signed_size * price

        # Phase 3: Apply commission costs
        notional = abs(signed_size * price)
        commission = 0.0
        if self.config.commission_per_lot > 0:
            num_lots = abs(signed_size) / self.config.lot_size
            commission += num_lots * self.config.commission_per_lot
        if self.config.commission_pct > 0:
            commission += notional * self.config.commission_pct

        self.cash -= commission
        self._commission_paid += commission

        self.inventory += signed_size
        self._fill_events.append(
            {
                "type": order_type,
                "side": side,
                "size": size,
                "price": price,
                "commission": commission,
                "step": self.step_count,
            }
        )
        self.logger.debug(
            "Filled %s %s of size %.2f at %.4f | commission %.4f | inventory %.2f",
            order_type,
            side,
            size,
            price,
            commission,
            self.inventory,
        )

    def _apply_fifo_fill(self, signed_size: float, price: float) -> None:
        remaining = signed_size
        while remaining != 0 and self.positions and np.sign(self.positions[0].quantity) != np.sign(remaining):
            open_pos = self.positions[0]
            nettable = min(abs(open_pos.quantity), abs(remaining))
            if open_pos.quantity > 0:
                pnl = (price - open_pos.entry_price) * nettable
                open_pos.quantity -= nettable
                remaining += nettable if remaining < 0 else -nettable
            else:
                pnl = (open_pos.entry_price - price) * nettable
                open_pos.quantity += nettable
                remaining += nettable if remaining < 0 else -nettable
            self.realized_pnl += pnl
            if abs(open_pos.quantity) < 1e-9:
                self.positions.popleft()

        if remaining != 0:
            self.positions.append(Position(quantity=remaining, entry_price=price))

    def _calculate_reward(self, portfolio_value: float) -> float:
        """Calculate reward based on configured reward_type.

        Args:
            portfolio_value: Current total portfolio value

        Returns:
            Calculated reward value
        """
        if self.config.reward_type == "portfolio_value":
            return self._calculate_reward_portfolio_value(portfolio_value)
        elif self.config.reward_type == "incremental_pnl":
            return self._calculate_reward_incremental_pnl(portfolio_value)
        elif self.config.reward_type == "sharpe":
            return self._calculate_reward_sharpe(portfolio_value)
        elif self.config.reward_type == "cost_aware":
            return self._calculate_reward_cost_aware(portfolio_value)
        else:
            raise ValueError(f"Unknown reward_type: {self.config.reward_type}")

    def _calculate_reward_portfolio_value(self, portfolio_value: float) -> float:
        """Simple reward: raw portfolio value (original implementation)."""
        return portfolio_value * self.config.reward_scaling

    def _calculate_reward_incremental_pnl(self, portfolio_value: float) -> float:
        """Reward based on change in portfolio value since last step."""
        pnl_change = portfolio_value - self._previous_portfolio_value

        # Apply penalties
        penalty = 0.0
        if self.config.drawdown_penalty_weight > 0:
            drawdown = max(0.0, (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value)
            penalty += self.config.drawdown_penalty_weight * drawdown * self.config.initial_cash

        reward = pnl_change - penalty
        return reward * self.config.reward_scaling

    def _calculate_reward_sharpe(self, portfolio_value: float) -> float:
        """Risk-adjusted reward using rolling Sharpe-like metric."""
        # Track portfolio value history
        self._portfolio_value_history.append(portfolio_value)

        # Keep only recent history for Sharpe calculation
        if len(self._portfolio_value_history) > self.config.sharpe_window:
            self._portfolio_value_history.pop(0)

        # Compute returns
        if len(self._portfolio_value_history) < 2:
            return 0.0  # Not enough data yet

        returns = [
            (self._portfolio_value_history[i] - self._portfolio_value_history[i - 1]) / self._portfolio_value_history[
                i - 1]
            for i in range(1, len(self._portfolio_value_history))
        ]

        if not returns:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe-like ratio (annualization factor omitted for simplicity)
        if std_return > 1e-9:
            sharpe = mean_return / std_return
        else:
            sharpe = 0.0

        return sharpe * self.config.reward_scaling

    def _calculate_reward_cost_aware(self, portfolio_value: float) -> float:
        """Reward that explicitly penalizes transaction costs."""
        pnl_change = portfolio_value - self._previous_portfolio_value

        # Calculate cost incurred since last step
        total_costs = self._commission_paid + self._spread_paid + self._slippage_paid
        costs_this_step = total_costs - self._costs_at_last_step

        # Apply cost penalty
        cost_penalty = self.config.cost_penalty_weight * costs_this_step

        # Apply drawdown penalty
        drawdown_penalty = 0.0
        if self.config.drawdown_penalty_weight > 0:
            drawdown = max(0.0, (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value)
            drawdown_penalty = self.config.drawdown_penalty_weight * drawdown * self.config.initial_cash

        reward = pnl_change - cost_penalty - drawdown_penalty
        return reward * self.config.reward_scaling

    def _observation(self) -> dict[str, float]:
        unrealized = sum(
            (
                (self.mid_price - pos.entry_price) * pos.quantity
                if pos.quantity > 0
                else (pos.entry_price - self.mid_price) * (-pos.quantity)
            )
            for pos in self.positions
        )
        portfolio_value = self.cash + self.inventory * self.mid_price + unrealized
        return {
            "mid_price": self.mid_price,
            "inventory": self.inventory,
            "pending_orders": float(len(self._pending_limits)),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "portfolio_value": portfolio_value,
            "step": float(self.step_count),
        }

    def _log_metrics(self) -> None:
        total_fills = len(self._fill_events)
        avg_slippage = self._slippage_paid / max(total_fills, 1)
        total_costs = self._spread_paid + self._slippage_paid + self._commission_paid

        # Phase 3.3: Calculate final drawdown
        final_portfolio = self.cash + self.inventory * self.mid_price
        final_drawdown = 0.0
        if self._peak_portfolio_value > 0:
            final_drawdown = (self._peak_portfolio_value - final_portfolio) / self._peak_portfolio_value

        self.logger.info(
            "Execution summary | fills=%d | realized_pnl=%.4f | inventory=%.2f | "
            "avg_slippage=%.4f | spread_paid=%.4f | commission_paid=%.4f | total_costs=%.4f",
            total_fills,
            self.realized_pnl,
            self.inventory,
            avg_slippage,
            self._spread_paid,
            self._commission_paid,
            total_costs,
        )

        # Phase 3.3: Log risk management statistics
        if self.config.enable_stop_loss or self.config.enable_take_profit or self.config.enable_drawdown_limit:
            self.logger.info(
                "Risk management | stop_loss_triggers=%d | take_profit_triggers=%d | "
                "max_drawdown=%.2f%% | final_portfolio=%.2f",
                self._stop_loss_triggered,
                self._take_profit_triggered,
                final_drawdown * 100,
                final_portfolio,
            )

        if total_fills:
            last_fill = self._fill_events[-1]
            self.logger.debug("Last fill: %s", last_fill)


__all__ = [
    "ExecutionConfig",
    "OrderAction",
    "SimulatedRetailExecutionEnv",
    "SlippageModel",
]
