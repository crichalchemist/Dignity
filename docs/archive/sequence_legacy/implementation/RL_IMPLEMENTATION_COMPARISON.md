# RL Implementation Comparison: Fake vs Real

## Summary

This document compares the **fake RL** implementation (supervised learning disguised as RL) with the **real RL**
implementation that uses actual trading environment and PnL rewards.

---

## Fake RL (Old Approach)

**Files:** `train/core/agent_train.py`, `train/core/agent_train_rl.py`

### How It Works

1. **Data Source:** Pre-labeled dataset with classification labels (0=SELL, 1=HOLD, 2=BUY)
2. **Reward Generation:** Lookup table converts labels to rewards
   ```python
   reward_lookup = torch.tensor([-1.0, 0.0, 1.0])
   rewards = reward_lookup[actions]  # Fake rewards from labels!
   ```
3. **Loss Function:** Cross-entropy loss on labels
   ```python
   policy_loss = nn.functional.cross_entropy(policy_logits, y)
   ```
4. **Training:** Standard supervised learning with batches

### Problems

❌ **Not Actually RL:** Uses classification labels, not environment interaction  
❌ **No Trading Simulation:** Never executes trades or measures PnL  
❌ **Fake Rewards:** Rewards come from labels, not actual trading outcomes  
❌ **Unrealistic:** Doesn't account for spread, slippage, position management  
❌ **Misleading:** Claims to be RL but is just supervised learning

---

## Real RL (New Approach)

**File:** `train/core/env_based_rl_training.py`

### How It Works

1. **Environment:** Uses `SimulatedRetailExecutionEnv` for realistic trading
2. **Episode Generation:** Agent interacts with environment to collect trajectories
   ```python
   obs, reward, done, info = env.step(order_action)
   # reward = actual portfolio value from trading!
   ```
3. **Real Rewards:** PnL from executed trades with spread and slippage
4. **Policy Gradient:** A2C-style updates using actual trading outcomes
   ```python
   policy_loss = -(log_probs * advantages).mean()  # advantages from real returns
   value_loss = F.mse_loss(values, returns)        # returns from trading PnL
   ```
5. **Training:** Episode collection → trajectory rollout → policy update

### Advantages

✅ **True RL:** Agent learns from environment interaction  
✅ **Realistic Trading:** Simulates spread, slippage, position management  
✅ **Real Rewards:** PnL comes from actual executed trades  
✅ **FIFO Position Tracking:** Proper realized/unrealized PnL calculation  
✅ **Risk Aware:** Environment includes inventory, cash, portfolio value

---

## Key Differences

| Aspect              | Fake RL                  | Real RL                        |
|---------------------|--------------------------|--------------------------------|
| **Data Source**     | Pre-labeled dataset      | Trading environment            |
| **Rewards**         | Label lookup [-1, 0, +1] | Actual PnL from trades         |
| **Training**        | Supervised batches       | Episode trajectories           |
| **Execution**       | None                     | Simulated with spread/slippage |
| **Position Mgmt**   | None                     | FIFO with realized PnL         |
| **Realism**         | Low                      | High                           |
| **Learning Signal** | Classification accuracy  | Trading profitability          |

---

## Migration Path

### Option 1: Replace Fake RL (Recommended)

1. Deprecate `agent_train_rl.py` and `agent_train.py:train_execution_policy`
2. Update training pipelines to use `env_based_rl_training.py`
3. Add warnings to old scripts pointing to new implementation

### Option 2: Keep Both (Transitional)

1. Rename old scripts:
    - `agent_train_rl.py` → `supervised_policy_training.py`
    - Mark as "supervised learning" not "RL"
2. Use `env_based_rl_training.py` for true RL experiments
3. Compare performance and decide later

---

## Example Usage

### Old Fake RL

```bash
python train/core/agent_train_rl.py \
  --pairs gbpusd \
  --epochs 10 \
  --learning-rate 1e-4
  # Uses labels as rewards - NOT TRUE RL!
```

### New Real RL

```bash
python train/core/env_based_rl_training.py \
  --signal-model-path models/gbpusd_signal.pt \
  --train-data-path data/gbpusd_train.npy \
  --epochs 10 \
  --learning-rate 1e-4 \
  --gamma 0.99
  # Uses trading environment - TRUE RL!
```

---

## Technical Details

### Environment Interaction (Real RL)

```python
# Episode loop
for episode_idx in range(num_episodes):
    env = SimulatedRetailExecutionEnv(config)
    obs = env.reset()

    for t in range(episode_length):
        # Model predicts action
        action = policy.get_action(obs)

        # Execute in environment
        order = convert_to_order_action(action)
        next_obs, reward, done, info = env.step(order)

        # Reward is actual PnL!
        # portfolio_value = cash + inventory * mid_price + unrealized_pnl

        episode.store(obs, action, reward)
        obs = next_obs

    # Update policy using real trading rewards
    update_policy(episode)
```

### Reward Calculation (Real RL)

The environment computes rewards from actual trading outcomes:

```python
# From simulated_retail_env.py:227
reward = obs["portfolio_value"]

# Where portfolio_value includes:
# - Cash from executed trades
# - Inventory valued at current mid price
# - Realized PnL from closed positions
# - Unrealized PnL from open positions
# - Spread costs (bid-ask)
# - Slippage costs (price impact)
```

This is **fundamentally different** from the fake RL approach which just maps labels to fixed rewards.

---

## Conclusion

The old "RL" implementation was **supervised learning** with extra steps. The new implementation uses a proper trading
environment with realistic execution mechanics and learns from actual PnL, making it **true reinforcement learning**.

**Recommendation:** Migrate to `env_based_rl_training.py` for all future RL experiments.
