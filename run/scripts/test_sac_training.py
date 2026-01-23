"""Quick test of SAC training loop."""

import logging
import sys

import numpy as np

sys.path.insert(0, "/")

from config.config import SignalModelConfig
from execution.limit_order_engine import LimitOrderConfig
from execution.simulated_retail_env import ExecutionConfig
from models.signal_policy import SignalModel
from train.core.sac_training import SACConfig, train_with_sac

logging.basicConfig(level=logging.INFO, format="%(message)s")

print("Testing SAC Training Loop (Quick Test)...\n")

# Create configurations
signal_cfg = SignalModelConfig(num_features=76)
signal_model = SignalModel(signal_cfg)

sac_config = SACConfig(
    buffer_size=1_000,  # Small buffer
    batch_size=32,  # Small batch
    updates_per_step=1,
    warmup_steps=50,  # Quick warmup
    auto_entropy_tuning=True,
)

env_config = ExecutionConfig(
    use_limit_order_engine=True,
    limit_order_config=LimitOrderConfig(),
    initial_cash=50_000.0,
)

# Dummy training data
train_data = np.random.randn(1000, 76)

# Run very short training
results = train_with_sac(
    signal_model=signal_model,
    sac_config=sac_config,
    signal_cfg=signal_cfg,
    env_config=env_config,
    train_data=train_data,
    num_episodes=3,  # Only 3 episodes
    max_steps_per_episode=20,  # Only 20 steps per episode
    log_frequency=2,
    checkpoint_dir=None,
    device="cpu",
)

print("\n" + "=" * 60)
print("✅ SAC Training Test Complete!")
print("=" * 60)
print("Results:")
print(f"  - Episodes: {len(results['episode_rewards'])}")
print(f"  - Total steps: {results['total_steps']}")
print(f"  - Final buffer size: {results['final_buffer_size']}")
print(f"  - Avg reward: {np.mean(results['episode_rewards']):.2f}")
print(f"  - Avg length: {np.mean(results['episode_lengths']):.0f}")
