# Tracing Quick Reference

## 🚀 Start Tracing in 3 Steps

### 1. Open Trace Collector

```
VS Code Command: ai-mlstudio.tracing.open
```

### 2. Run Training (Auto-instrumented)

```bash
python train/run_training.py --pairs gbpusd --epochs 3
# or
python train/run_training_multitask.py --pairs gbpusd --epochs 3
```

### 3. View Traces

Look in AI Toolkit's trace viewer to see:

- Training epochs with duration
- Per-batch loss and metrics
- Validation performance
- Error details (if any)

---

## 📊 Trace Structure

```
Service: sequence-training
├── training_epoch (1/3)
│   ├── batch_processing (1)
│   │   └── loss: 0.8234
│   ├── batch_processing (2)
│   │   └── loss: 0.7981
│   └── validation
│       ├── val_loss: 0.6234
│       └── val_metric: 0.823
├── training_epoch (2/3)
│   └── ...
└── training_epoch (3/3)
    └── ...
```

---

## 💻 Manual Tracing

### In Your Code

```python
from utils.tracing import get_tracer, TracingContext

tracer = get_tracer(__name__)

# Simple span
with TracingContext(tracer, "operation_name", {"key": "value"}) as span:
    result = expensive_operation()
    span.set_attribute("result", result)

# Decorator
from utils.tracing import trace_function

@trace_function(tracer, func_name="my_op")
def my_function(x, y):
    return x + y
```

---

## 📋 Instrumented Components

✅ **Training Loops**

- Single-task training (`train/core/agent_train.py`)
- Multi-task training (`train/core/agent_train_multitask.py`)

✅ **CLI Runners**

- `train/run_training.py`
- `train/run_training_multitask.py`

**Available for instrumentation:**

- Data loading (`data/prepare_dataset.py`)
- Feature engineering (`features/agent_features.py`)
- Evaluation (`eval/run_evaluation.py`)

---

## 🔧 Configuration

### Environment Variables

```bash
# Optional: change OTLP endpoint
OTEL_EXPORTER_OTLP_ENDPOINT=http://your-collector:4318

# Optional: trace sampling (0.0-1.0)
OTEL_TRACES_SAMPLER=traceidratio
OTEL_TRACES_SAMPLER_ARG=0.5
```

### Python Code

```python
from utils.tracing import setup_tracing

setup_tracing(
    service_name="my-service",
    otlp_endpoint="http://localhost:4318",
    environment="production"
)
```

---

## 🐛 Troubleshooting

| Problem              | Solution                                                                      |
|----------------------|-------------------------------------------------------------------------------|
| Traces not appearing | Check `ai-mlstudio.tracing.open` is running                                   |
| Import error         | `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp` |
| High latency         | Collector may be slow; check network                                          |
| Missing metrics      | Ensure spans use `set_attribute()`                                            |

---

## 📚 Full Documentation

See `docs/TRACING_SETUP.md` for:

- Complete setup guide
- All available utilities
- Advanced usage patterns
- Best practices
- Integration examples

See `TRACING_IMPLEMENTATION.md` for:

- What was implemented
- Files modified
- Integration architecture
- Metrics captured

---

## 🎯 Key Spans

| Span                  | Attributes                                | Use Case                       |
|-----------------------|-------------------------------------------|--------------------------------|
| `training_epoch`      | epoch, total_epochs, train_loss, val_loss | Track training progress        |
| `batch_processing`    | batch_index, batch_size, loss             | Debug convergence issues       |
| `validation`          | val_loss, val_metric                      | Monitor validation performance |
| `data_loading`        | pair, num_samples                         | Identify data bottlenecks      |
| `feature_engineering` | pair, feature_count                       | Profile feature computation    |

---

## ⚡ Performance

- **Overhead:** <1% with batch export
- **Batch size:** 512 spans (default)
- **Export interval:** 5 seconds
- **Memory:** ~MB per 1000 spans

---

**Last Updated:** 2025-12-06
