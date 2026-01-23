# Google Colab Setup Guide

This guide helps you run the Sequence trading system on Google Colab.

## Quick Start

### 1. Clone the Repository

```python
# Run this cell first
!git clone https://github.com/YOUR_USERNAME/Sequence.git
%cd Sequence
```

### 2. Install Dependencies

```python
# Install required packages
!pip install -q -r requirements.txt

# Install TimesFM foundation model (editable install)
!pip install -q -e ./models/timesFM
```

### 3. Setup Python Path (CRITICAL)

**IMPORTANT**: Run this cell in every new Colab session before importing any Sequence modules:

```python
import sys
from pathlib import Path

# Add Sequence root and run/ to Python path
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

print(f"✅ Python path configured:")
print(f"   - ROOT: {ROOT}")
print(f"   - run/: {ROOT / 'run'}")
```

### 4. Verify Setup

```python
# Test imports
try:
    from config.config import ModelConfig, DataConfig
    from utils.logger import get_logger
    from train.features.agent_features import build_feature_frame
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you ran the 'Setup Python Path' cell above")
```

## Data Storage Options

### Option A: Use Google Drive (Recommended for persistence)

```python
from google.colab import drive
drive.mount('/content/drive')

# Set data directory in your Drive
import os
os.environ['SEQUENCE_DATA_DIR'] = '/content/drive/MyDrive/Sequence/data'

# Create data directory
!mkdir -p "/content/drive/MyDrive/Sequence/data"
```

### Option B: Use Colab's Temporary Storage (Faster but ephemeral)

```python
# Data will be lost when runtime disconnects
!mkdir -p /content/data
```

## Common Workflows

### Workflow 1: Data Preparation Only

```python
# Setup (run cells 1-3 first)

# Prepare dataset with NPY format (fast loading)
!python data/prepare_dataset.py \
  --pairs gbpusd \
  --years 2023,2024 \
  --t-in 120 \
  --t-out 10 \
  --task-type classification \
  --input-root data/data
```

### Workflow 2: Complete Training Pipeline

```python
# Run complete pipeline: download → prepare → train
!python run/training_pipeline.py \
  --pairs gbpusd \
  --run-histdata-download \
  --years 2023,2024 \
  --t-in 120 \
  --t-out 10 \
  --epochs 10 \
  --batch-size 64 \
  --checkpoint-dir models
```

### Workflow 3: Training with GDELT Sentiment (BigQuery)

```python
# GDELT via BigQuery (recommended for Colab - no file downloads)
!python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --include-sentiment \
  --use-bigquery-gdelt \
  --gdelt-themes "WB_1427_BUSINESS_FINANCE,WB_2327_BUSINESS_FINANCIAL_MARKETS"
```

Note: BigQuery requires authentication. See [GDELT BigQuery Setup](#gdelt-bigquery-setup) below.

### Workflow 4: RL Training with Backtesting

```python
# Train RL agent on historical data
!python rl/run_a3c_training.py \
  --pair gbpusd \
  --env-mode backtesting \
  --historical-data data/data/gbpusd/gbpusd_prepared.csv \
  --num-workers 2 \
  --total-steps 100000
```

## Performance Optimizations for Colab

### 1. Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 or A100 if available)
3. Verify GPU is active:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 2. Use NPY Format for Large Datasets

The codebase now supports NPY format (30-50x faster than CSV):

```python
# Data preparation automatically saves both NPY and CSV
!python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10

# Training pipeline will auto-detect and use NPY format
!python run/training_pipeline.py --pairs gbpusd --epochs 10
```

### 3. Reduce Memory Usage

For large datasets or limited RAM:

```python
# Use smaller batch size
!python run/training_pipeline.py --pairs gbpusd --batch-size 32

# Process fewer pairs at once
!python run/training_pipeline.py --pairs gbpusd --epochs 10  # Not gbpusd,eurusd,gbpjpy
```

## GDELT BigQuery Setup

### 1. Enable BigQuery API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **BigQuery API**
4. Create service account key (JSON)

### 2. Authenticate in Colab

```python
from google.colab import auth
auth.authenticate_user()

# Set project ID
import os
os.environ['GOOGLE_CLOUD_PROJECT'] = 'your-project-id'
```

### 3. Test GDELT Query

```python
!python data/prepare_dataset.py \
  --pairs gbpusd \
  --t-in 120 \
  --t-out 10 \
  --include-sentiment \
  --use-bigquery-gdelt \
  --years 2024
```

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'config'`

**Solution**: Make sure you ran the "Setup Python Path" cell (step 3)

```python
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))
```

### Out of Memory

**Error**: `CUDA out of memory` or `RuntimeError: [enforce fail at alloc_cpu.cpp]`

**Solution**: Reduce batch size or use smaller datasets

```python
# Smaller batch size
!python run/training_pipeline.py --batch-size 32

# Use fewer years of data
!python data/prepare_dataset.py --years 2024 --pairs gbpusd
```

### Missing Data Files

**Error**: `FileNotFoundError: data/data/gbpusd/gbpusd_prepared.csv`

**Solution**: Run data preparation first

```python
!python data/prepare_dataset.py --pairs gbpusd --years 2023,2024
```

### Slow CSV Loading

**Issue**: Loading prepared data takes 30-60 seconds

**Solution**: The codebase now automatically uses NPY format (30-50x faster). Make sure you're using the latest version:

```python
# Re-prepare data to generate NPY files
!python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10

# Training will now load NPY format automatically
!python run/training_pipeline.py --pairs gbpusd --epochs 10
```

## File Structure on Colab

```
/content/Sequence/                    # Repository root
├── data/
│   ├── data/                         # Prepared datasets
│   │   └── gbpusd/
│   │       ├── gbpusd_prepared.npy       # Fast NPY format (auto-generated)
│   │       ├── gbpusd_prepared.csv       # CSV fallback (auto-generated)
│   │       └── gbpusd_prepared_metadata.json  # Metadata
│   └── prepare_dataset.py            # Data preparation script
├── models/                           # Model checkpoints
│   └── gbpusd_best_model.pt
├── run/
│   ├── config/
│   │   └── config.py                 # Configuration dataclasses
│   └── training_pipeline.py          # Main training pipeline
└── train/
    └── run_training.py               # Training entry point
```

## Best Practices

1. **Always run the Python path setup cell first** in every new session
2. **Use Google Drive** for data persistence across sessions
3. **Enable GPU** for faster training (Runtime → Change runtime type → GPU)
4. **Monitor GPU memory** with `!nvidia-smi` to avoid OOM errors
5. **Save checkpoints frequently** to avoid losing progress on disconnects
6. **Use NPY format** for large datasets (automatically enabled in latest version)

## Example: Complete Colab Session

```python
# 1. Clone and setup
!git clone https://github.com/YOUR_USERNAME/Sequence.git
%cd Sequence
!pip install -q -r requirements.txt

# 2. Setup paths (CRITICAL)
import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "run"))

# 3. Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 4. Prepare data
!python data/prepare_dataset.py --pairs gbpusd --years 2024 --t-in 120 --t-out 10

# 5. Train model
!python run/training_pipeline.py --pairs gbpusd --epochs 20 --batch-size 64

# 6. Evaluate
!python eval/run_evaluation.py --pairs gbpusd --checkpoint-path models/gbpusd_best_model.pt
```

## Additional Resources

- **Main Documentation**: See [README.md](README.md)
- **Configuration Reference**: See [CLAUDE.md](CLAUDE.md)
- **Architecture Guide**: See [docs/ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)
- **Notebook Examples**: See [notebooks/](notebooks/)

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the error message carefully
3. Verify you ran the Python path setup cell
4. Check that data files exist in the expected locations
5. Open an issue on GitHub with error details
