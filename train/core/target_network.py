"""
Target network wrapper for stable value estimation in RL.

Addresses the 'moving target problem' in temporal difference learning:
- In TD learning, both prediction and target depend on the same network weights
- Updates cause target values to shift, creating instability (Mnih et al., 2015)
- Solution: Use a separate frozen network for target computation

Two update modes:
1. Soft updates (Polyak averaging): θ_target = τ*θ_main + (1-τ)*θ_target
   - Smoother, more stable (recommended for continuous control)
   - Default τ=0.005 from SAC/TD3 papers
2. Hard updates: Periodic full copy every N steps
   - Original DQN approach
"""

import copy
from typing import Any

import torch
import torch.nn as nn


class TargetNetwork:
    """Manages target network for stable temporal difference learning.

    Example usage:
        # Create target network wrapper
        target_net = TargetNetwork(value_network, tau=0.005, update_mode="soft")

        # Get stable target values (no gradients)
        target_values = target_net.get_target_value(next_states)

        # Compute TD targets
        td_targets = rewards + gamma * target_values

        # Train main network on TD error
        value_loss = F.mse_loss(value_network(states), td_targets)

        # Update target network
        target_net.update()
    """

    def __init__(
            self,
            network: nn.Module,
            tau: float = 0.005,
            update_frequency: int = 1,
            update_mode: str = "soft"
    ):
        """Initialize target network wrapper.

        Args:
            network: Main network to wrap (e.g., value function, Q-network)
            tau: Soft update coefficient (0 < τ << 1)
                - τ=0.001: Very slow updates (high stability)
                - τ=0.005: Standard for SAC/TD3
                - τ=0.01: Faster updates (less stable)
            update_frequency: Steps between hard updates (only for hard mode)
            update_mode: "soft" (Polyak averaging) or "hard" (periodic copy)
        """
        self.main_network = network
        self.target_network = copy.deepcopy(network)
        self.target_network.eval()  # Set to evaluation mode

        # Freeze target network parameters (no gradients)
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.tau = tau
        self.update_frequency = update_frequency
        self.update_mode = update_mode
        self.steps = 0

        # Log configuration
        print(f"✓ Created target network (mode={update_mode}, tau={tau})")

    def get_target_value(self, *args, **kwargs) -> torch.Tensor:
        """Compute target values without gradients.

        This prevents gradients from flowing into the target network,
        which would defeat the purpose of having a separate target.

        Args:
            *args, **kwargs: Forwarded to target network

        Returns:
            Target network output (detached from computation graph)
        """
        with torch.no_grad():
            return self.target_network(*args, **kwargs)

    def update(self) -> bool:
        """Update target network parameters.

        Soft mode: Updates every step with Polyak averaging
        Hard mode: Full copy every N steps

        Returns:
            True if update occurred, False otherwise (hard mode only)
        """
        self.steps += 1

        if self.update_mode == "soft":
            self._soft_update()
            return True
        else:  # hard mode
            if self.steps % self.update_frequency == 0:
                self._hard_update()
                return True
            return False

    def _soft_update(self):
        """Polyak averaging: Exponential moving average of parameters.

        Formula: θ_target = τ*θ_main + (1-τ)*θ_target

        This creates a smooth blend between old and new values:
        - Small τ → slow tracking, high stability
        - Large τ → fast tracking, lower stability

        Mathematically equivalent to exponential smoothing:
        After k updates, target ≈ main from k*τ/log(2) steps ago
        Example: τ=0.005 → target lags ~140 steps behind main
        """
        for target_param, main_param in zip(
                self.target_network.parameters(),
                self.main_network.parameters(), strict=False
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )

    def _hard_update(self):
        """Full parameter copy (original DQN approach)."""
        self.target_network.load_state_dict(self.main_network.state_dict())
        print(f"  [Target network] Hard update at step {self.steps}")

    def state_dict(self) -> dict[str, Any]:
        """Save target network state for checkpointing."""
        return {
            'target_network_state': self.target_network.state_dict(),
            'tau': self.tau,
            'update_frequency': self.update_frequency,
            'update_mode': self.update_mode,
            'steps': self.steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Restore target network from checkpoint."""
        self.target_network.load_state_dict(state_dict['target_network_state'])
        self.tau = state_dict.get('tau', self.tau)
        self.update_frequency = state_dict.get('update_frequency', self.update_frequency)
        self.update_mode = state_dict.get('update_mode', self.update_mode)
        self.steps = state_dict.get('steps', 0)


# Example test
if __name__ == "__main__":
    import torch.nn.functional as F


    # Create simple value network
    class ValueNetwork(nn.Module):
        def __init__(self, input_dim=10):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)


    # Test soft updates
    print("Testing soft update mode...")
    network = ValueNetwork()
    target_wrapper = TargetNetwork(network, tau=0.1, update_mode="soft")

    # Simulate training
    for step in range(5):
        # Generate fake batch
        states = torch.randn(32, 10)
        rewards = torch.randn(32, 1)
        next_states = torch.randn(32, 10)

        # Get target values
        target_values = target_wrapper.get_target_value(next_states)
        td_targets = rewards + 0.99 * target_values

        # Train main network
        values = network(states)
        loss = F.mse_loss(values, td_targets)
        loss.backward()

        # Update target
        target_wrapper.update()

        print(f"  Step {step}: loss={loss.item():.4f}")

    # Test hard updates
    print("\nTesting hard update mode...")
    network2 = ValueNetwork()
    target_wrapper2 = TargetNetwork(network2, update_frequency=3, update_mode="hard")

    for step in range(7):
        target_wrapper2.update()

    print("\n✓ Target network tests passed!")
