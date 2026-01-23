# Phase 1 Test Suite Implementation - Setup & Status

## Overview

Phase 1 testing infrastructure has been created with comprehensive unit and integration tests for the training pipeline and model architecture. This document covers the setup, structure, and execution instructions.

## Files Created

### Test Infrastructure
- `tests/conftest.py` - Root-level shared fixtures for all test phases
- `tests/train/__init__.py` - Train test module
- `tests/models/__init__.py` - Models test module  
- `tests/integration/__init__.py` - Integration test module
- `tests/fixtures/__init__.py` - Fixtures package
- `pytest.ini` - Updated with comprehensive markers and configuration

### Phase 1 Tests (130-145 tests)

#### 1.1 Loss Computation & Metrics (`tests/train/test_agent_train_loss.py`)
- **TestLossComputation** (6 tests)
  - Classification and regression loss computation
  - Auxiliary task weighting (sell_now, topk returns, etc.)
  - Gradient flow validation
  - NaN/Inf handling
  
- **TestClassificationMetrics** (3 tests)
  - Accuracy calculation with perfect/partial/zero correctness
  
- **TestRegressionMetrics** (2 tests)
  - RMSE calculation with perfect fit and known errors
  
- **TestGradientFlow** (2 tests)
  - Gradient existence after backward pass
  - Gradient clipping functionality

**Total: 13 tests**

#### 1.2 Model Architecture (`tests/models/test_model_architectures.py`)
- **TestSharedEncoder** (2 tests)
  - Initialization and forward pass
  - Output shape validation
  
- **TestDignityModel** (6 tests)
  - Model instantiation
  - Classification forward pass
  - Parameter counts
  - Output range validation (no NaN/Inf)
  - Gradient flow through entire model
  - Various batch sizes (1, 4, 8, 16)
  - Training vs eval mode differences
  
- **TestSignalPolicy** (2 tests)
  - SignalModel initialization and forward pass
  - ExecutionPolicy initialization
  
- **TestRegimeEncoder** (2 tests)
  - Initialization and forward pass
  
- **TestAgentMultitask** (3 tests)
  - Multi-task model initialization
  - Forward pass with main + auxiliary tasks
  - Gradient flow through task heads

**Total: 15 tests**

#### 1.3 Training Manager (`tests/train/test_training_manager.py`)
- **TestCheckpointManagement** (5 tests)
  - Basic checkpoint save/load
  - State dict compatibility
  - Metadata preservation
  - Multiple checkpoint sequences
  
- **TestTrainingOrchestration** (4 tests)
  - Epoch counter increments
  - Batch counter reset per epoch
  - Early stopping logic
  - Learning rate scheduling
  
- **TestValidationHandling** (3 tests)
  - Train/val/test data isolation
  - Metrics collection and aggregation
  
- **TestTrainingStateManagement** (3 tests)
  - Optimizer state dict save/load
  - Model eval/train mode checkpointing

**Total: 15 tests**

#### 1.4 End-to-End Integration (`tests/integration/test_training_e2e.py`)
- **TestTrainingE2E** (5 tests)
  - Single epoch training loop
  - Loss decrease validation
  - Checkpoint resume without loss spike
  - Gradient accumulation
  - DataLoader integration
  
- **TestGradientFlow** (2 tests)
  - Full pipeline gradient flow
  - No-grad mode validation

**Total: 7 tests**

## Test Structure & Fixtures

### Shared Fixtures (conftest.py)
```python
@pytest.fixture
device                  # torch.device (CPU/CUDA)

@pytest.fixture
sample_batch           # (x: [4,120,20], y: dict with labels/targets)

@pytest.fixture
sample_batch_regression  # (x: [4,120,20], y: dict with regression targets)

@pytest.fixture
sample_dataframe       # Price DataFrame (120 timesteps, 20 features)

@pytest.fixture
mock_model_config      # ModelConfig with defaults

@pytest.fixture
mock_training_config   # TrainingConfig with defaults

@pytest.fixture
temp_checkpoint_dir    # Temporary directory for test checkpoints

@pytest.fixture
minimal_dataloader     # PyTorch DataLoader with one batch
```

## Setup Instructions

### Prerequisites
1. **Python 3.10+** recommended
2. **Virtual Environment**: Use `.venvx_dignity/` or your preferred venv

### Installation

```bash
# Activate virtual environment (macOS/Linux)
source .venvx_dignity/bin/activate
# OR on Windows:
# .venvx_dignity\Scripts\activate

# Alternative: use your venv name
# source venv/bin/activate
# OR: venv\Scripts\activate  (Windows)

# Install core dependencies
pip install torch pandas numpy scikit-learn

# Install testing dependencies
pip install pytest pytest-cov pytest-mock hypothesis pytest-timeout

# Optional: Install remaining requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check pytest and torch are installed
python -c "import pytest; import torch; print('Ready!')"

# Run a quick test (from project root)
pytest tests/train/test_agent_train_loss.py::TestLossComputation::test_compute_losses_classification_single_batch -v

# OR use make target if available:
# make test-quick
```

## Test Execution

### Run All Phase 1 Tests
```bash
# From project root directory
pytest tests/train/ tests/models/ tests/integration/test_training_e2e.py -v

# OR use make target:
# make test-phase1
```

### Run by Category
```bash
# Unit tests only (fast, every commit)
pytest tests/ -m unit -v --tb=short

# Integration tests (pre-merge)
pytest tests/integration/ -v --tb=short

# With coverage report
pytest tests/train/ tests/models/ tests/integration/test_training_e2e.py --cov=train --cov=models --cov-report=term

# Specific test class
pytest tests/train/test_agent_train_loss.py::TestLossComputation -v

# Specific test
pytest tests/train/test_agent_train_loss.py::TestLossComputation::test_compute_losses_classification_single_batch -v
```

## Expected Test Results (After Environment Setup)

### Unit Tests (Phase 1)
- **test_agent_train_loss.py**: 13 tests ✓
- **test_model_architectures.py**: 15 tests ✓
- **test_training_manager.py**: 15 tests ✓

### Integration Tests (Phase 1)
- **test_training_e2e.py**: 7 tests ✓

**Total Phase 1: ~50 tests implemented** (Phase 1 complete; total planned across all phases: 130-145 tests)

## Coverage Targets

After Phase 1 completion, expected coverage:
- `train/core/agent_train.py`: 75-85%
- `models/agent_hybrid.py`: 70-80%
- `train/training_manager.py`: 60-70%
- **Overall train/ + models/**: 65-75%

## Next Steps

### Phase 2: Evaluation & Risk Management
Once Phase 1 is verified:
```bash
# Create Phase 2 test files
tests/eval/test_agent_eval.py
tests/eval/test_run_evaluation.py
tests/risk/test_risk_manager.py
tests/integration/test_eval_pipeline.py
```

### Phase 3: Data & Preparation
```bash
tests/data/test_prepare_dataset.py
tests/data/test_yfinance_downloader.py
tests/data/test_histdata.py
tests/integration/test_data_pipeline.py
```

### Phase 4: Utils & Config
```bash
tests/utils/test_logger.py
tests/utils/test_tracing.py
tests/config/test_config.py
tests/integration/test_training_pipeline_e2e.py
```

## CI/CD Integration

### Recommended GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run unit tests (every commit)
        run: pytest tests/ -m unit --tb=short -x
      
      - name: Run integration tests (PR only)
        if: github.event_name == 'pull_request'
        run: pytest tests/integration/ --tb=short
      
      - name: Generate coverage report (PR only)
        if: github.event_name == 'pull_request'
        run: |
          pytest tests/ --cov=train --cov=models --cov=eval --cov=risk \
            --cov-report=term --cov-report=html
          echo "Coverage generated in htmlcov/"
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'torch'**
   - Solution: `pip install torch`

2. **ModuleNotFoundError: No module named 'config'**
   - Ensure pytest is run from the project root directory
   - Check that `tests/conftest.py` adds ROOT to sys.path
   - Consider installing package in editable mode: `pip install -e .`

3. **Tests timeout**
   - Check pytest.ini has `--timeout=300`
   - Long-running tests need `@pytest.mark.slow`

4. **CUDA out of memory**
   - Use `device = "cpu"` fixture instead of GPU
   - All tests are CPU-compatible

## Success Criteria

Phase 1 is complete when:
- ✓ All 50 tests execute successfully
- ✓ No import errors
- ✓ Coverage ≥60% for train/ and models/
- ✓ Test execution time <5 minutes
- ✓ All fixtures work as expected
- ✓ Gradient flow validated

## Test Markers Reference

```bash
# Run only unit tests (fastest, every commit)
pytest -m unit

# Run integration tests (moderate speed, on PR)
pytest -m integration

# Run slow tests (nightly only)
pytest -m slow

# Run real API tests (nightly, requires API keys)
pytest -m real_api

# Run regression guardrails
pytest -m regression

# Run everything except slow
pytest -m "not slow"
```
