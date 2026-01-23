# Dignity Core Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Transform Sequence research prototype into Dignity Core - a modular, production-ready framework for transaction sequence modeling.

**Architecture:** Extract monolithic components into composable modules (core, data, models, train, export), implement config-driven training, enable ONNX deployment.

**Tech Stack:** PyTorch, YAML configs, ONNX Runtime, pytest

---

## Task 1: Scaffold Directory Structure

**Files:**
- Create: `dignity/__init__.py`
- Create: `dignity/core/__init__.py`
- Create: `dignity/data/__init__.py`
- Create: `dignity/models/__init__.py`
- Create: `dignity/models/backbone/__init__.py`
- Create: `dignity/models/head/__init__.py`
- Create: `dignity/train/__init__.py`
- Create: `dignity/export/__init__.py`
- Create: `dignity/config/__init__.py`
- Create: `.gitignore`

**Step 1: Create core directory structure**

```bash
mkdir -p dignity/{core,data,models/{backbone,head},train,export,config}
touch dignity/__init__.py
touch dignity/core/__init__.py
touch dignity/data/__init__.py
touch dignity/models/__init__.py
touch dignity/models/backbone/__init__.py
touch dignity/models/head/__init__.py
touch dignity/train/__init__.py
touch dignity/export/__init__.py
touch dignity/config/__init__.py
```

**Step 2: Create comprehensive .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# PyCharm
.idea/

# VSCode
.vscode/

# Jupyter
.ipynb_checkpoints
*.ipynb

# Pytest
.pytest_cache/
.cache/

# Data & Models
*.csv
*.pkl
*.pt
*.pth
*.onnx
checkpoints/
output_central/
data/data/
*.h5

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Secrets
.env
*.key
*.secret

# Sequence source (cloned separately)
Sequence/
```

**Step 3: Commit scaffold**

```bash
git add dignity/ .gitignore
git commit -m "feat: scaffold Dignity core directory structure"
```

---

## Task 2: Core Utilities Module

**Files:**
- Create: `dignity/core/config.py`
- Create: `dignity/core/signals.py`
- Create: `dignity/core/privacy.py`
- Create: `tests/test_core_signals.py`

**Step 1: Write failing test for signal processing**

```python
# tests/test_core_signals.py
import pytest
import numpy as np
from dignity.core.signals import compute_entropy, compute_volatility

def test_entropy_calculation():
    """Entropy should be 0 for uniform distribution"""
    values = np.ones(100)
    entropy = compute_entropy(values, window=10)
    assert len(entropy) == 100
    assert np.allclose(entropy[-10:], 0.0)

def test_volatility_calculation():
    """Volatility should detect changes"""
    values = np.concatenate([np.ones(50), np.ones(50) * 2])
    volatility = compute_volatility(values, window=10)
    assert len(volatility) == 100
    assert volatility[55] > volatility[5]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core_signals.py -v
```
Expected: `ModuleNotFoundError: No module named 'dignity.core.signals'`

**Step 3: Implement signal processing utilities**

```python
# dignity/core/signals.py
"""Signal processing utilities for transaction sequences."""
import numpy as np
from scipy.stats import entropy as scipy_entropy

def compute_entropy(values: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute rolling Shannon entropy.
    
    Args:
        values: Input signal
        window: Rolling window size
        
    Returns:
        Entropy values (same length as input)
    """
    result = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        
        # Bin data for probability distribution
        hist, _ = np.histogram(window_data, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        result[i] = scipy_entropy(hist)
    
    return result

def compute_volatility(values: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute rolling standard deviation (volatility).
    
    Args:
        values: Input signal
        window: Rolling window size
        
    Returns:
        Volatility values (same length as input)
    """
    result = np.zeros(len(values))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_data = values[start:i+1]
        result[i] = np.std(window_data)
    
    return result

def compute_returns(values: np.ndarray) -> np.ndarray:
    """Compute percentage returns."""
    return np.diff(values) / values[:-1]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_core_signals.py -v
```
Expected: `2 passed`

**Step 5: Implement config loader**

```python
# dignity/core/config.py
"""Configuration management."""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config.
    
    Args:
        base: Base configuration
        override: Override values
        
    Returns:
        Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
```

**Step 6: Implement privacy utilities**

```python
# dignity/core/privacy.py
"""Privacy-preserving utilities for transaction data."""
import hashlib
from typing import Union

def hash_identifier(identifier: str, salt: str = "") -> str:
    """
    Hash sensitive identifiers (merchant IDs, user IDs).
    
    Args:
        identifier: Raw identifier
        salt: Optional salt for hashing
        
    Returns:
        Hex digest of hashed identifier
    """
    data = f"{identifier}{salt}".encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def anonymize_amount(amount: float, bucket_size: float = 10.0) -> float:
    """
    Bucket transaction amounts for privacy.
    
    Args:
        amount: Original amount
        bucket_size: Size of buckets (e.g., $10)
        
    Returns:
        Bucketed amount
    """
    return (amount // bucket_size) * bucket_size
```

**Step 7: Commit core utilities**

```bash
git add dignity/core/ tests/test_core_signals.py
git commit -m "feat: add core signal processing and config utilities"
```

---

## Task 3: Model Backbone Components

**Files:**
- Create: `dignity/models/backbone/cnn1d.py`
- Create: `dignity/models/backbone/lstm.py`
- Create: `dignity/models/backbone/attention.py`
- Create: `dignity/models/backbone/hybrid.py`
- Create: `tests/test_model_backbone.py`

**Step 1: Write failing test for CNN1D**

```python
# tests/test_model_backbone.py
import pytest
import torch
from dignity.models.backbone.cnn1d import CNN1D

def test_cnn1d_forward():
    """CNN1D should process sequential input"""
    batch_size, seq_len, input_size = 8, 100, 9
    hidden_size = 64
    
    model = CNN1D(input_size, hidden_size, kernel_size=3)
    x = torch.randn(batch_size, seq_len, input_size)
    
    out = model(x)
    
    assert out.shape == (batch_size, seq_len, hidden_size)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_model_backbone.py::test_cnn1d_forward -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement CNN1D backbone**

```python
# dignity/models/backbone/cnn1d.py
"""1D Convolutional backbone for sequence processing."""
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    1D CNN for extracting local patterns from sequences.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of output features
        kernel_size: Convolution kernel size
        n_layers: Number of conv layers
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 kernel_size: int = 3, n_layers: int = 2):
        super().__init__()
        
        layers = []
        in_channels = input_size
        
        for i in range(n_layers):
            out_channels = hidden_size if i == n_layers - 1 else hidden_size // 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(0.1)
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, features]
            
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Conv1d expects [batch, features, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_model_backbone.py::test_cnn1d_forward -v
```
Expected: `1 passed`

**Step 5: Write failing test for LSTM**

```python
# Add to tests/test_model_backbone.py
from dignity.models.backbone.lstm import StackedLSTM

def test_stacked_lstm_forward():
    """StackedLSTM should process sequences"""
    batch_size, seq_len, hidden_size = 8, 100, 64
    
    model = StackedLSTM(hidden_size, hidden_size, n_layers=2)
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    out, (h_n, c_n) = model(x)
    
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (2, batch_size, hidden_size)  # n_layers=2
```

**Step 6: Implement StackedLSTM**

```python
# dignity/models/backbone/lstm.py
"""LSTM backbone for temporal modeling."""
import torch
import torch.nn as nn
from typing import Tuple

class StackedLSTM(nn.Module):
    """
    Stacked LSTM for temporal sequence modeling.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        n_layers: Number of LSTM layers
        dropout: Dropout probability (applied between layers)
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor, 
                hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, features]
            hidden: Optional (h_0, c_0) hidden states
            
        Returns:
            output: [batch, seq_len, hidden_size]
            (h_n, c_n): Final hidden states
        """
        out, hidden = self.lstm(x, hidden)
        return out, hidden
```

**Step 7: Run LSTM test**

```bash
pytest tests/test_model_backbone.py::test_stacked_lstm_forward -v
```
Expected: `1 passed`

**Step 8: Write failing test for Attention**

```python
# Add to tests/test_model_backbone.py
from dignity.models.backbone.attention import AdditiveAttention

def test_additive_attention():
    """Attention should compute context vector"""
    batch_size, seq_len, hidden_size = 8, 100, 64
    
    model = AdditiveAttention(hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    context, weights = model(x)
    
    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))
```

**Step 9: Implement AdditiveAttention**

```python
# dignity/models/backbone/attention.py
"""Attention mechanism for sequence aggregation."""
import torch
import torch.nn as nn
from typing import Tuple, Optional

class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    
    Args:
        hidden_size: Dimension of hidden states
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention over sequence.
        
        Args:
            x: [batch, seq_len, hidden_size]
            mask: Optional [batch, seq_len] boolean mask (True = attend)
            
        Returns:
            context: [batch, hidden_size] weighted sum
            weights: [batch, seq_len] attention weights
        """
        # Compute attention scores
        scores = self.v(torch.tanh(self.W(x)))  # [B, T, 1]
        scores = scores.squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)  # [B, T]
        
        # Compute context as weighted sum
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, H]
        
        return context, weights
```

**Step 10: Run attention test**

```bash
pytest tests/test_model_backbone.py::test_additive_attention -v
```
Expected: `1 passed`

**Step 11: Write failing test for Hybrid backbone**

```python
# Add to tests/test_model_backbone.py
from dignity.models.backbone.hybrid import DignityBackbone

def test_dignity_backbone_integration():
    """Full backbone should integrate CNN + LSTM + Attention"""
    batch_size, seq_len, input_size = 8, 100, 9
    hidden_size = 64
    
    model = DignityBackbone(input_size, hidden_size, n_layers=2)
    x = torch.randn(batch_size, seq_len, input_size)
    
    context, weights = model(x)
    
    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len)
```

**Step 12: Implement DignityBackbone**

```python
# dignity/models/backbone/hybrid.py
"""Hybrid CNN-LSTM-Attention backbone."""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .cnn1d import CNN1D
from .lstm import StackedLSTM
from .attention import AdditiveAttention

class DignityBackbone(nn.Module):
    """
    Dignity core backbone: CNN → LSTM → Attention.
    
    Extracts temporal context from transaction sequences.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    
    def __init__(self, input_size: int = 9, hidden_size: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.cnn = CNN1D(input_size, hidden_size, kernel_size=3)
        self.lstm = StackedLSTM(hidden_size, hidden_size, n_layers, dropout)
        self.attn = AdditiveAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, features] input sequences
            mask: Optional [batch, seq_len] attention mask
            
        Returns:
            context: [batch, hidden_size] context vector
            weights: [batch, seq_len] attention weights
        """
        # Extract local patterns
        x = self.cnn(x)  # [B, T, H]
        
        # Model temporal dependencies
        x, _ = self.lstm(x)  # [B, T, H]
        x = self.dropout(x)
        
        # Aggregate with attention
        context, weights = self.attn(x, mask)  # [B, H], [B, T]
        
        return context, weights
```

**Step 13: Run backbone integration test**

```bash
pytest tests/test_model_backbone.py::test_dignity_backbone_integration -v
```
Expected: `1 passed`

**Step 14: Commit backbone components**

```bash
git add dignity/models/backbone/ tests/test_model_backbone.py
git commit -m "feat: implement CNN-LSTM-Attention backbone components"
```

---

## Task 4: Task-Specific Heads

**Files:**
- Create: `dignity/models/head/risk.py`
- Create: `dignity/models/head/forecast.py`
- Create: `tests/test_model_heads.py`

**Step 1: Write failing test for RiskHead**

```python
# tests/test_model_heads.py
import pytest
import torch
from dignity.models.head.risk import RiskHead

def test_risk_head():
    """RiskHead should output risk score [0, 1]"""
    batch_size, hidden_size = 8, 64
    
    model = RiskHead(hidden_size)
    context = torch.randn(batch_size, hidden_size)
    
    risk_score = model(context)
    
    assert risk_score.shape == (batch_size, 1)
    assert (risk_score >= 0).all() and (risk_score <= 1).all()
```

**Step 2: Implement RiskHead**

```python
# dignity/models/head/risk.py
"""Risk scoring head."""
import torch
import torch.nn as nn

class RiskHead(nn.Module):
    """
    Risk score prediction head.
    
    Outputs a single risk score in [0, 1].
    
    Args:
        hidden_size: Input context dimension
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict risk score.
        
        Args:
            context: [batch, hidden_size]
            
        Returns:
            [batch, 1] risk scores
        """
        return self.fc(context)
```

**Step 3: Run risk test**

```bash
pytest tests/test_model_heads.py::test_risk_head -v
```
Expected: `1 passed`

**Step 4: Write failing test for ForecastHead**

```python
# Add to tests/test_model_heads.py
from dignity.models.head.forecast import ForecastHead

def test_forecast_head():
    """ForecastHead should predict future values"""
    batch_size, hidden_size, pred_len = 8, 64, 5
    
    model = ForecastHead(hidden_size, pred_len)
    context = torch.randn(batch_size, hidden_size)
    
    predictions = model(context)
    
    assert predictions.shape == (batch_size, pred_len)
```

**Step 5: Implement ForecastHead**

```python
# dignity/models/head/forecast.py
"""Forecasting head for future value prediction."""
import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    """
    Multi-step forecasting head.
    
    Predicts future values (e.g., volume, transaction count).
    
    Args:
        hidden_size: Input context dimension
        pred_len: Number of future steps to predict
    """
    
    def __init__(self, hidden_size: int, pred_len: int = 5):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, pred_len)
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict future values.
        
        Args:
            context: [batch, hidden_size]
            
        Returns:
            [batch, pred_len] predictions
        """
        return self.fc(context)
```

**Step 6: Run forecast test**

```bash
pytest tests/test_model_heads.py::test_forecast_head -v
```
Expected: `1 passed`

**Step 7: Commit heads**

```bash
git add dignity/models/head/ tests/test_model_heads.py
git commit -m "feat: add task-specific prediction heads"
```

---

## Task 5: Main Dignity Model

**Files:**
- Create: `dignity/models/dignity.py`
- Create: `tests/test_dignity_model.py`

**Step 1: Write failing test**

```python
# tests/test_dignity_model.py
import pytest
import torch
from dignity.models.dignity import Dignity

def test_dignity_risk_model():
    """Dignity risk model should integrate backbone + risk head"""
    batch_size, seq_len, input_size = 8, 100, 9
    
    model = Dignity(task='risk', input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    
    risk_score, attention = model(x)
    
    assert risk_score.shape == (batch_size, 1)
    assert attention.shape == (batch_size, seq_len)

def test_dignity_forecast_model():
    """Dignity forecast model should predict future values"""
    batch_size, seq_len, input_size = 8, 100, 9
    pred_len = 10
    
    model = Dignity(task='forecast', input_size=input_size, pred_len=pred_len)
    x = torch.randn(batch_size, seq_len, input_size)
    
    predictions, attention = model(x)
    
    assert predictions.shape == (batch_size, pred_len)
    assert attention.shape == (batch_size, seq_len)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_dignity_model.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement Dignity model**

```python
# dignity/models/dignity.py
"""Main Dignity model: task-agnostic backbone + task-specific head."""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .backbone.hybrid import DignityBackbone
from .head.risk import RiskHead
from .head.forecast import ForecastHead

class Dignity(nn.Module):
    """
    Dignity: Modular sequence model for transaction behavior.
    
    Combines shared backbone with task-specific heads.
    
    Args:
        task: Task type ('risk', 'forecast')
        input_size: Number of input features
        hidden_size: Backbone hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout probability
        pred_len: Prediction horizon (for forecast task)
    """
    
    VALID_TASKS = {'risk', 'forecast'}
    
    def __init__(self, task: str = 'risk', input_size: int = 9,
                 hidden_size: int = 256, n_layers: int = 2,
                 dropout: float = 0.1, pred_len: int = 5):
        super().__init__()
        
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got {task}")
        
        self.task = task
        self.backbone = DignityBackbone(input_size, hidden_size, n_layers, dropout)
        
        # Task-specific head
        if task == 'risk':
            self.head = RiskHead(hidden_size)
        elif task == 'forecast':
            self.head = ForecastHead(hidden_size, pred_len)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, features] input sequences
            mask: Optional [batch, seq_len] attention mask
            
        Returns:
            output: Task-specific predictions
            attention: [batch, seq_len] attention weights
        """
        context, attention = self.backbone(x, mask)
        output = self.head(context)
        
        return output, attention
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_dignity_model.py -v
```
Expected: `2 passed`

**Step 5: Commit main model**

```bash
git add dignity/models/dignity.py tests/test_dignity_model.py
git commit -m "feat: implement main Dignity model with task routing"
```

---

## Task 6: Base Configuration Files

**Files:**
- Create: `dignity/config/base.yaml`
- Create: `dignity/config/train_risk.yaml`
- Create: `dignity/config/train_forecast.yaml`

**Step 1: Create base configuration**

```yaml
# dignity/config/base.yaml
# Base configuration for Dignity training

model:
  input_size: 9  # volume, fee_rate, entropy, tx_count, etc.
  hidden_size: 256
  n_layers: 2
  dropout: 0.1

data:
  seq_len: 100
  batch_size: 64
  num_workers: 2
  test_size: 0.2
  val_size: 0.1

train:
  epochs: 50
  lr: 3e-4
  weight_decay: 1e-5
  patience: 10  # Early stopping
  use_amp: true  # Automatic Mixed Precision
  grad_clip: 1.0

device: cuda

logging:
  log_interval: 10  # Log every N batches
  checkpoint_dir: checkpoints
  save_best_only: true
```

**Step 2: Create risk-specific config**

```yaml
# dignity/config/train_risk.yaml
# Risk scoring task configuration

# Inherit from base
base: base.yaml

model:
  task: risk

data:
  source: synthetic  # or 'crypto', 'gdelt'
  target_col: risk_label

train:
  loss: bce  # Binary Cross-Entropy
  metric: auc  # Area Under ROC Curve
  
  # Class weights for imbalanced data
  pos_weight: 2.0
```

**Step 3: Create forecast-specific config**

```yaml
# dignity/config/train_forecast.yaml
# Volume forecasting task configuration

base: base.yaml

model:
  task: forecast
  pred_len: 10  # Predict next 10 time steps

data:
  source: crypto
  target_col: volume

train:
  loss: mse  # Mean Squared Error
  metric: rmse  # Root Mean Squared Error
  
  # Normalize targets
  normalize_targets: true
```

**Step 4: Commit configs**

```bash
git add dignity/config/
git commit -m "feat: add base and task-specific config templates"
```

---

## Task 7: Data Pipeline Foundation

**Files:**
- Create: `dignity/data/pipeline.py`
- Create: `dignity/data/loader.py`
- Create: `tests/test_data_pipeline.py`

**Step 1: Write failing test for pipeline**

```python
# tests/test_data_pipeline.py
import pytest
import numpy as np
import pandas as pd
from dignity.data.pipeline import TransactionPipeline

@pytest.fixture
def sample_dataframe():
    """Create sample transaction data"""
    np.random.seed(42)
    n_samples = 1000
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
        'volume': np.random.lognormal(5, 2, n_samples),
        'fee_rate': np.random.uniform(0.001, 0.01, n_samples),
        'tx_count': np.random.poisson(10, n_samples),
        'risk_label': np.random.binomial(1, 0.1, n_samples)
    })

def test_pipeline_fit_transform(sample_dataframe):
    """Pipeline should scale features correctly"""
    pipeline = TransactionPipeline(seq_len=50)
    
    X_scaled = pipeline.fit_transform(sample_dataframe)
    
    assert X_scaled.shape[0] == len(sample_dataframe)
    assert X_scaled.shape[1] == 3  # volume, fee_rate, tx_count
    
    # Scaled data should have reasonable range
    assert X_scaled.mean() < 1.0
    assert X_scaled.std() < 5.0

def test_pipeline_to_sequence(sample_dataframe):
    """Pipeline should create sliding windows"""
    pipeline = TransactionPipeline(seq_len=50)
    X_scaled = pipeline.fit_transform(sample_dataframe)
    y = sample_dataframe['risk_label'].values
    
    X_seq, y_seq = pipeline.to_sequence(X_scaled, y)
    
    # Check shapes
    n_windows = len(sample_dataframe) - 50
    assert X_seq.shape == (n_windows, 50, 3)
    assert y_seq.shape == (n_windows,)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_pipeline.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement TransactionPipeline**

```python
# dignity/data/pipeline.py
"""Data preprocessing pipeline for transaction sequences."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Optional, List

class TransactionPipeline:
    """
    Preprocessing pipeline for transaction data.
    
    Handles:
    - Feature scaling
    - Sequence window generation
    - Train/test splitting
    
    Args:
        seq_len: Length of input sequences
        feature_cols: Column names to use as features
    """
    
    def __init__(self, seq_len: int = 100, 
                 feature_cols: Optional[List[str]] = None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols or ['volume', 'fee_rate', 'tx_count']
        self.scaler = RobustScaler()
        self._fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and transform features.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Scaled feature array [n_samples, n_features]
        """
        X = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self._fitted = True
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Scaled feature array
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        X = df[self.feature_cols].values
        return self.scaler.transform(X)
    
    def to_sequence(self, X: np.ndarray, 
                    y: Optional[np.ndarray] = None,
                    stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate sliding window sequences.
        
        Args:
            X: Feature array [n_samples, n_features]
            y: Optional target array [n_samples]
            stride: Step size for sliding window
            
        Returns:
            X_seq: [n_windows, seq_len, n_features]
            y_seq: [n_windows] (if y provided)
        """
        n_samples = len(X)
        n_windows = (n_samples - self.seq_len) // stride
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(0, n_samples - self.seq_len, stride):
            X_seq.append(X[i:i + self.seq_len])
            
            if y is not None:
                # Use label at end of sequence
                y_seq.append(y[i + self.seq_len - 1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_data_pipeline.py -v
```
Expected: `2 passed`

**Step 5: Write failing test for DataLoader**

```python
# Add to tests/test_data_pipeline.py
from dignity.data.loader import DignityDataset
from torch.utils.data import DataLoader

def test_dignity_dataset(sample_dataframe):
    """DignityDataset should provide PyTorch tensors"""
    pipeline = TransactionPipeline(seq_len=50)
    X_scaled = pipeline.fit_transform(sample_dataframe)
    y = sample_dataframe['risk_label'].values
    X_seq, y_seq = pipeline.to_sequence(X_scaled, y)
    
    dataset = DignityDataset(X_seq, y_seq)
    
    assert len(dataset) == len(X_seq)
    
    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (50, 3)
    assert y_sample.shape == ()

def test_dignity_dataloader(sample_dataframe):
    """DataLoader should batch data correctly"""
    pipeline = TransactionPipeline(seq_len=50)
    X_scaled = pipeline.fit_transform(sample_dataframe)
    y = sample_dataframe['risk_label'].values
    X_seq, y_seq = pipeline.to_sequence(X_scaled, y)
    
    dataset = DignityDataset(X_seq, y_seq)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (16, 50, 3)
    assert y_batch.shape == (16,)
```

**Step 6: Implement DignityDataset**

```python
# dignity/data/loader.py
"""PyTorch Dataset and DataLoader utilities."""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class DignityDataset(Dataset):
    """
    PyTorch Dataset for transaction sequences.
    
    Args:
        X: Sequence array [n_samples, seq_len, n_features]
        y: Target array [n_samples]
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx] if self.y is not None else torch.tensor(0.0)
        return x, y
```

**Step 7: Run dataloader tests**

```bash
pytest tests/test_data_pipeline.py -v
```
Expected: `4 passed`

**Step 8: Commit data pipeline**

```bash
git add dignity/data/ tests/test_data_pipeline.py
git commit -m "feat: implement data preprocessing pipeline and PyTorch Dataset"
```

---

## Task 8: Training Engine

**Files:**
- Create: `dignity/train/engine.py`
- Create: `tests/test_train_engine.py`

**Step 1: Write failing test**

```python
# tests/test_train_engine.py
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dignity.train.engine import train_epoch, validate_epoch

@pytest.fixture
def dummy_model():
    """Simple model for testing"""
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

@pytest.fixture
def dummy_dataloader():
    """Create dummy data"""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,)).float()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16)

def test_train_epoch(dummy_model, dummy_dataloader):
    """train_epoch should update model and return loss"""
    optimizer = torch.optim.Adam(dummy_model.parameters())
    criterion = nn.BCELoss()
    
    loss = train_epoch(
        dummy_model, dummy_dataloader, optimizer, criterion,
        device='cpu', use_amp=False
    )
    
    assert isinstance(loss, float)
    assert loss > 0

def test_validate_epoch(dummy_model, dummy_dataloader):
    """validate_epoch should compute validation loss"""
    criterion = nn.BCELoss()
    
    loss = validate_epoch(
        dummy_model, dummy_dataloader, criterion,
        device='cpu'
    )
    
    assert isinstance(loss, float)
    assert loss > 0
```

**Step 2: Implement training engine**

```python
# dignity/train/engine.py
"""Training and validation loops."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: str = 'cuda', use_amp: bool = True,
                grad_clip: float = 1.0) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping threshold
        
    Returns:
        Average loss for epoch
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Handle models that return (output, attention)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = criterion(out.squeeze(), y)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches

def validate_epoch(model: nn.Module, dataloader: DataLoader,
                   criterion: nn.Module, device: str = 'cuda') -> float:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data
        criterion: Loss function
        device: Device to use
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # Handle models that return (output, attention)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = criterion(out.squeeze(), y)
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches
```

**Step 3: Run test**

```bash
pytest tests/test_train_engine.py -v
```
Expected: `2 passed`

**Step 4: Commit training engine**

```bash
git add dignity/train/engine.py tests/test_train_engine.py
git commit -m "feat: implement training and validation loops with AMP support"
```

---

## Task 9: ONNX Export

**Files:**
- Create: `dignity/export/to_onnx.py`
- Create: `tests/test_onnx_export.py`

**Step 1: Write failing test**

```python
# tests/test_onnx_export.py
import pytest
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from dignity.models.dignity import Dignity
from dignity.export.to_onnx import export_to_onnx

def test_onnx_export(tmp_path):
    """Should export model to ONNX format"""
    model = Dignity(task='risk', input_size=9, hidden_size=64, n_layers=1)
    model.eval()
    
    output_path = tmp_path / "test_model.onnx"
    
    export_to_onnx(
        model, 
        output_path=str(output_path),
        seq_len=100,
        input_size=9
    )
    
    # Verify file exists
    assert output_path.exists()
    
    # Verify ONNX model is valid
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

def test_onnx_inference(tmp_path):
    """ONNX model should produce same output as PyTorch"""
    model = Dignity(task='risk', input_size=9, hidden_size=64, n_layers=1)
    model.eval()
    
    output_path = tmp_path / "test_model.onnx"
    export_to_onnx(model, str(output_path), seq_len=100, input_size=9)
    
    # Create test input
    x = torch.randn(1, 100, 9)
    
    # PyTorch inference
    with torch.no_grad():
        torch_out, _ = model(x)
    
    # ONNX inference
    ort_session = ort.InferenceSession(str(output_path))
    onnx_out = ort_session.run(
        None,
        {'input': x.numpy()}
    )[0]
    
    # Compare outputs (allow small numerical difference)
    assert torch.allclose(torch_out, torch.tensor(onnx_out), atol=1e-5)
```

**Step 2: Implement ONNX export**

```python
# dignity/export/to_onnx.py
"""ONNX export utilities."""
import torch
import torch.nn as nn
from pathlib import Path

def export_to_onnx(model: nn.Module, output_path: str,
                   seq_len: int = 100, input_size: int = 9,
                   opset_version: int = 13) -> None:
    """
    Export Dignity model to ONNX format.
    
    Args:
        model: Trained Dignity model
        output_path: Path to save ONNX model
        seq_len: Input sequence length
        input_size: Number of input features
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output', 'attention'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
            'attention': {0: 'batch'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")
```

**Step 3: Run test**

```bash
pytest tests/test_onnx_export.py -v
```
Expected: `2 passed`

**Step 4: Commit ONNX export**

```bash
git add dignity/export/ tests/test_onnx_export.py
git commit -m "feat: add ONNX export with validation"
```

---

## Task 10: Requirements and Setup

**Files:**
- Create: `requirements.txt`
- Create: `setup.py`
- Modify: `pyproject.toml`

**Step 1: Create requirements.txt**

```text
# requirements.txt
# Core ML
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Data processing
pandas>=2.0.0
scikit-learn>=1.3.0
pyyaml>=6.0

# Training utilities
tqdm>=4.65.0

# Export
onnx>=1.14.0
onnxruntime>=1.15.0

# Development
pytest>=7.4.0
pytest-cov>=4.1.0
```

**Step 2: Create setup.py**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="dignity",
    version="0.1.0",
    description="Modular sequence modeling for transaction behavior",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
)
```

**Step 3: Update pyproject.toml (if exists)**

```toml
# Add to pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests"
]
```

**Step 4: Commit setup files**

```bash
git add requirements.txt setup.py pyproject.toml
git commit -m "feat: add package requirements and setup"
```

---

## Task 11: README and Documentation

**Files:**
- Create: `README.md`
- Create: `docs/QUICKSTART.md`

**Step 1: Create README**

```markdown
# Dignity Core

> Modular sequence modeling for transaction behavior patterns.

Dignity is a **privacy-first**, **low-resource**, **deployable** framework for training and serving neural sequence models on transactional data.

## Key Features

- **Modular Architecture**: Composable backbone (CNN-LSTM-Attention) + task-specific heads
- **Config-Driven**: YAML-based training with sensible defaults
- **Production-Ready**: ONNX export for <50ms inference
- **Resource-Efficient**: Optimized for Colab T4/A100
- **Privacy-Preserving**: Built-in hashing and anonymization

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train risk model
python -m dignity.train.cli --config dignity/config/train_risk.yaml

# Export to ONNX
python -m dignity.export.to_onnx --checkpoint checkpoints/best_model.pt --output model.onnx
```

## Architecture

```
Input Sequence → CNN1D → LSTM → Attention → Task Head → Output
[B, T, F]         [B, T, H]   [B, H]      [B, K]
```

**Tasks:**
- `risk`: Transaction risk scoring [0, 1]
- `forecast`: Multi-step volume prediction

## Project Structure

```
dignity/
├── core/           # Config, signals, privacy utils
├── data/           # Preprocessing pipelines
├── models/
│   ├── backbone/   # CNN, LSTM, Attention
│   ├── head/       # Task-specific heads
│   └── dignity.py  # Main model
├── train/          # Training loops
├── export/         # ONNX export
└── config/         # YAML configs
```

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest tests/test_*.py    # Specific test
```

## License

MIT
```

**Step 2: Create QUICKSTART guide**

```markdown
# Dignity Quickstart

## 1. Installation

```bash
git clone <repo>
cd dignity
pip install -r requirements.txt
```

## 2. Prepare Data

Your data should have these columns:
- `timestamp`: Transaction time
- `volume`: Transaction amount
- `fee_rate`: Fee percentage
- `tx_count`: Number of transactions
- `risk_label`: Binary risk flag (0/1) for risk task

## 3. Train Risk Model

```bash
python -m dignity.train.cli --config dignity/config/train_risk.yaml
```

## 4. Export to ONNX

```bash
python -m dignity.export.to_onnx \
  --checkpoint checkpoints/best_model.pt \
  --output dignity_risk.onnx \
  --task risk
```

## 5. Inference (Python)

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("dignity_risk.onnx")
x = np.random.randn(1, 100, 9).astype(np.float32)

risk_score, attention = session.run(None, {'input': x})
print(f"Risk: {risk_score[0][0]:.3f}")
```

## Next Steps

- Customize configs in `dignity/config/`
- Add custom data sources in `dignity/data/source/`
- Implement new task heads in `dignity/models/head/`
```

**Step 3: Commit documentation**

```bash
git add README.md docs/QUICKSTART.md
git commit -m "docs: add README and quickstart guide"
```

---

## Final Verification

**Step 1: Run full test suite**

```bash
pytest -v
```
Expected: All tests pass

**Step 2: Verify package structure**

```bash
tree dignity -I __pycache__
```

**Step 3: Test imports**

```bash
python -c "from dignity.models import Dignity; print('✓ Import successful')"
```

**Step 4: Create final commit**

```bash
git add .
git commit -m "chore: final verification and cleanup"
git tag v0.1.0
```

---

## Summary

This implementation plan transforms Sequence into Dignity Core through:

1. ✅ Clean directory structure
2. ✅ Modular components (backbone, heads)
3. ✅ Config-driven operation
4. ✅ Comprehensive testing
5. ✅ ONNX deployment
6. ✅ Documentation

**Total commits**: 11 focused commits following TDD

**Next Phase**: 
- Add synthetic data generator
- Implement training CLI
- Colab notebook for end-to-end demo
