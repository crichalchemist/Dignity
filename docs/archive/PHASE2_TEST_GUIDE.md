# Phase 2 Test Quick Reference Guide

## Test Files Created (4 new files, 93 tests total)

### 1. tests/eval/test_agent_eval.py (17 tests)
**Covers**: Evaluation metrics and model scoring

**Test Classes**:
- `TestCollectOutputs` - Gather predictions from model
- `TestClassificationMetrics` - Accuracy, precision, recall, F1
- `TestRegressionMetrics` - MSE, RMSE, MAE, R²
- `TestEvaluateModel` - Full evaluation framework
- `TestEvaluationWithDifferentDataSizes` - Scalability

**Key Tests**:
```bash
# Run all evaluation tests
pytest tests/eval/test_agent_eval.py -v

# Run specific test class
pytest tests/eval/test_agent_eval.py::TestClassificationMetrics -v

# Run single test
pytest tests/eval/test_agent_eval.py::TestClassificationMetrics::test_perfect_accuracy -v
```

---

### 2. tests/risk/test_risk_manager.py (32 tests)
**Covers**: Risk management gates and position limits

**Test Classes**:
- `TestRiskConfig` - Configuration validation
- `TestRiskManagerInitialization` - Setup and initialization
- `TestEquityTracking` - Peak equity and current equity
- `TestDrawdownDetection` - Drawdown calculation
- `TestPositionLimits` - Max open positions
- `TestNoTradeWindow` - Market hours restrictions
- `TestVolatilityThrottle` - Volatility gates
- `TestSpreadCheck` - Spread limitations
- `TestActiveGates` - Gate aggregation

**Key Tests**:
```bash
# Run all risk manager tests
pytest tests/risk/test_risk_manager.py -v

# Run drawdown tests
pytest tests/risk/test_risk_manager.py::TestDrawdownDetection -v

# Run with specific keyword
pytest tests/risk/test_risk_manager.py -k "drawdown" -v
```

---

### 3. tests/eval/test_run_evaluation.py (19 tests)
**Covers**: Evaluation script orchestration

**Test Classes**:
- `TestArgumentParsing` - CLI argument handling
- `TestDeviceHandling` - GPU/CPU selection
- `TestPairProcessing` - Currency pair parsing
- `TestModelCheckpointLoading` - Model loading
- `TestRiskManagerInitialization` - Risk setup
- `TestSignalPolicyPathFormatting` - File path handling
- `TestResultsCollection` - Results aggregation
- `TestEvaluationPipelineIntegration` - Full pipeline

**Key Tests**:
```bash
# Run all run_evaluation tests
pytest tests/eval/test_run_evaluation.py -v

# Run argument parsing tests only
pytest tests/eval/test_run_evaluation.py::TestArgumentParsing -v

# Run with coverage
pytest tests/eval/test_run_evaluation.py --cov=eval.run_evaluation -v
```

---

### 4. tests/eval/test_eval_pipeline.py (10 integration tests)
**Covers**: End-to-end evaluation workflows

**Test Classes**:
- `TestEvaluationWithRiskManager` - Integration with risk manager
- `TestEvaluationPipelineFlow` - Data → Model → Evaluation
- `TestMultiplePairEvaluation` - Multi-pair workflows
- `TestRiskManagerIntegrationWithEvaluation` - Risk gates in eval
- `TestEvaluationOutputFormatting` - Output structure
- `TestCheckpointLoading` - Model checkpoint persistence

**Key Tests**:
```bash
# Run all integration tests
pytest tests/eval/test_eval_pipeline.py -v

# Run only TestEvaluationPipelineFlow
pytest tests/eval/test_eval_pipeline.py::TestEvaluationPipelineFlow -v

# Run integration tests only (all Phase 2)
pytest tests/eval tests/risk -m integration -v
```

---

## Running Tests

### Quick Start
```bash
# Run all Phase 2 tests
pytest tests/eval tests/risk -v

# Run with summary
pytest tests/eval tests/risk -v --tb=short

# Run with coverage report
pytest tests/eval tests/risk --cov=eval --cov=risk --cov-report=term-missing
```

### By Type
```bash
# Unit tests only (fast, ~5s)
pytest tests/eval tests/risk -m unit

# Integration tests only
pytest tests/eval tests/risk -m integration

# All tests
pytest tests/eval tests/risk
```

### By Module
```bash
# Just evaluation
pytest tests/eval -v

# Just risk management
pytest tests/risk -v

# Specific file
pytest tests/eval/test_agent_eval.py -v
```

### Debug Mode
```bash
# Show print statements
pytest tests/eval tests/risk -v -s

# Stop on first failure
pytest tests/eval tests/risk -x

# Show local variables on failure
pytest tests/eval tests/risk -l

# Run with pdb on failure
pytest tests/eval tests/risk --pdb
```

---

## Test Structure

### Standard Pattern Used
```python
@pytest.mark.unit  # or @pytest.mark.integration
class TestFeatureName:
    """Description of test class."""
    
    def test_specific_behavior(self, fixture1, fixture2):
        """Specific behavior being tested."""
        # Arrange
        component = setup()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected
```

### Available Fixtures (from conftest.py)
- `device` - PyTorch device (CPU/GPU)
- `sample_batch` - Standard (X, y) tensor pair
- `model_config` - Mock model configuration
- `training_config` - Mock training configuration

---

## Common Test Patterns

### Classification Evaluation
```python
def test_classification_evaluation(self):
    """Test classification metrics."""
    metrics = classification_metrics(outputs, targets)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
```

### Risk Gate Activation
```python
def test_gate_activation(self):
    """Test risk gate."""
    context = {"volatility": 0.03}
    active_gates = rm._active_gates(context)
    assert "volatility_throttle" in active_gates
```

### Model Evaluation Pipeline
```python
def test_evaluation_pipeline(self):
    """Test full evaluation."""
    metrics = evaluate_model(model, test_loader, task_type="classification")
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
```

---

## Debugging Failed Tests

### If torch import fails
```bash
# Install torch in environment
pip install torch torchvision torchaudio

# Or use conda
conda install pytorch::pytorch torchvision torchaudio -c pytorch
```

### If fixture not found
```bash
# Ensure conftest.py is in tests/ directory
ls -la tests/conftest.py

# Verify pytest can see conftest
pytest --fixtures | grep sample_batch
```

### If module import fails
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Run from repository root
cd <project_root>  # Navigate to your repository root directory
pytest tests/eval tests/risk -v
```

---

## Metrics Explained

### Classification Metrics
- **Accuracy**: Correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1**: Harmonic mean of precision and recall

### Regression Metrics
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination (0-1, higher is better)

### Risk Metrics
- **Drawdown**: (Peak - Current) / Peak percentage
- **Volatility**: Standard deviation of returns
- **Spread**: Bid-ask spread in market
- **Position Limit**: Max open positions

---

## Next Steps

After Phase 2 completion:
1. Run full Phase 2 test suite
2. Measure coverage (target: 85% for eval/risk)
3. Document any failing tests
4. Proceed to Phase 3: Data downloaders (42-50 tests)

---

*Phase 2 Complete: 93 tests covering evaluation and risk management*
*Estimated Phase 1+2 Coverage: 40-45% of codebase*
*Total Tests (Phase 1+2): 143 tests*
