# New Data Sources Integration

This document describes the three new fundamental data sources integrated into the Sequence trading system.

## Quick Start (5 minutes)

```bash
# 1. Install packages
bash run/scripts/install_data_sources.sh

# 2. Set up API keys
export FRED_API_KEY="your_fred_key_here"
export COMTRADE_API_KEY="your_comtrade_key_here"  # optional

# 3. Test installation
python run/scripts/test_data_sources.py

# 4. Collect data
python run/scripts/example_fundamental_integration.py \
    --pair EURUSD \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --output-dir data/fundamentals
```

**Get API Keys:**
- **FRED** (required): https://fred.stlouisfed.org/docs/api/api_key.html
- **Comtrade** (optional): https://comtradeplus.un.org/

---

## Overview

Three new data sources have been added to enhance forex prediction with macroeconomic fundamentals:

1. **UN Comtrade** - International trade data and trade balance indicators
2. **FRED (Federal Reserve Economic Data)** - Enhanced economic indicators
3. **ECB Monetary Policy Shocks** - Central bank policy surprise measures

## Installation

### Quick Install

```bash
bash run/scripts/install_data_sources.sh
```

### Manual Install

```bash
# Install UN Comtrade API
pip install -e ./new_data_sources/comtradeapicall

# Install FRED API
pip install -e ./new_data_sources/FRB

# ECB shocks: No installation needed (CSV data)
```

## API Keys Setup

### UN Comtrade API Key

1. Register at https://comtradeplus.un.org/
2. Get your subscription key
3. Set environment variable:
   ```bash
   export COMTRADE_API_KEY="your_key_here"
   ```

### FRED API Key

1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Get your API key
3. Set environment variable:
   ```bash
   export FRED_API_KEY="your_key_here"
   ```

### ECB Shocks

No API key required - data is loaded from CSV files in `new_data_sources/jkshocks_update_ecb/`.

## Usage Examples

### Collect All Fundamental Data

```python
from data.extended_data_collection import collect_all_forex_fundamentals
import os

# Collect all fundamental data for EUR/USD
data = collect_all_forex_fundamentals(
    currency_pair="EURUSD",
    start_date="2023-01-01",
    end_date="2023-12-31",
    comtrade_api_key=os.getenv("COMTRADE_API_KEY"),
    fred_api_key=os.getenv("FRED_API_KEY")
)

print(f"Trade balance records: {len(data['trade'])}")
print(f"Economic indicators: {len(data['economic'])}")
print(f"Monetary policy shocks: {len(data['shocks'])}")
```

### Individual Data Source Usage

#### UN Comtrade - Trade Data

```python
from data.downloaders.comtrade_downloader import get_trade_indicators_for_forex

# Get trade balance for EUR/USD countries
trade_data = get_trade_indicators_for_forex(
    currency_pair="EURUSD",
    start_year=2023,
    end_year=2023,
    subscription_key=os.getenv("COMTRADE_API_KEY")
)
```

#### FRED - Economic Indicators

```python
from data.downloaders.fred_downloader import get_forex_economic_indicators

# Get economic indicators for EUR/USD
economic_data = get_forex_economic_indicators(
    currency_pair="EURUSD",
    start_date="2023-01-01",
    end_date="2023-12-31",
    api_key=os.getenv("FRED_API_KEY"),
    indicators=["interest_rate", "inflation", "gdp"]  # Optional filter
)
```

#### ECB Monetary Policy Shocks

```python
from data.downloaders.ecb_shocks_downloader import load_ecb_shocks_daily

# Load daily ECB monetary policy shocks
shocks = load_ecb_shocks_daily()

# Filter for significant events
significant_shocks = shocks[abs(shocks['MP_median']) > 0.05]
print(f"Found {len(significant_shocks)} significant policy shocks")
```

### Save and Load Data

```python
from data.extended_data_collection import (
    collect_all_forex_fundamentals,
    save_fundamental_data
)

# Collect data
data = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")

# Save to parquet files
paths = save_fundamental_data(
    data=data,
    output_dir="data/fundamentals",
    currency_pair="EURUSD",
    file_format="parquet"
)

# Outputs:
# - data/fundamentals/EURUSD_trade.parquet
# - data/fundamentals/EURUSD_economic.parquet
# - data/fundamentals/EURUSD_shocks.parquet
```

### Merge with Price Data

```python
from data.extended_data_collection import merge_with_price_data
import pandas as pd

# Load price data
price_df = pd.read_parquet("data/prepared/EURUSD_1h.parquet")

# Collect fundamental data
fundamentals = collect_all_forex_fundamentals("EURUSD", "2023-01-01", "2023-12-31")

# Merge everything
merged_df = merge_with_price_data(fundamentals, price_df, date_column="datetime")

# Now merged_df contains:
# - All price columns (open, high, low, close, volume)
# - Trade balance indicators (prefixed with 'trade_')
# - Economic indicators (prefixed with 'economic_')
# - Monetary shocks (prefixed with 'shocks_')
```

## Data Sources Details

### 1. UN Comtrade - International Trade Data

**What it provides:**
- Monthly trade balance (exports - imports) by country
- Bilateral trade flows
- Relevant for forex pairs as trade balance affects currency demand

**Supported currency pairs:**
- EURUSD, GBPUSD, USDJPY, AUDUSD, EURJPY, EURGBP, USDCAD, USDCHF

**Data frequency:** Monthly

**Key features:**
- Trade balance is a leading indicator for currency strength
- Export/import imbalances drive currency flows
- Particularly relevant for commodity currencies (AUD, CAD)

### 2. FRED - Federal Reserve Economic Data

**What it provides:**
- Interest rates (Fed Funds, ECB Deposit Rate, etc.)
- Inflation (CPI for USD, EUR, GBP, JPY, AUD, CAD)
- GDP growth rates
- Unemployment rates
- Retail sales

**Supported currencies:**
- USD, EUR, GBP, JPY, AUD, CAD

**Data frequency:** Varies by series (daily to quarterly)

**Key features:**
- Central bank policy indicators (interest rates)
- Economic health metrics (GDP, unemployment)
- Inflation expectations
- Forward-filled to match price data frequency

### 3. ECB Monetary Policy Shocks

**What it provides:**
- Surprise changes in ECB policy stance
- Decomposition into Monetary Policy (MP) and Central Bank Information (CBI) shocks
- Event-based data (Governing Council meetings)

**Source:**
Jarocinski & Karadi (2020) - "Deconstructing Monetary Policy Surprises"

**Data frequency:** Daily (by meeting) or Monthly (aggregated)

**Key columns:**
- `MP_median`: Pure monetary policy shock (hawkish/dovish)
- `CBI_median`: Information shock (economic outlook signal)
- `pc1`: Overall policy surprise (first principal component)
- `STOXX50`: Market reaction during event window

**Use cases:**
- Identify surprise policy changes
- Distinguish between policy moves and information signals
- EUR pairs only (EURUSD, EURGBP, EURJPY, EURCHF)

## Integration with Training Pipeline

### Adding to Dataset Preparation

```python
from data.extended_data_collection import collect_all_forex_fundamentals, merge_with_price_data
from data.prepare_dataset import prepare_forex_dataset
import pandas as pd

# 1. Collect fundamentals
fundamentals = collect_all_forex_fundamentals(
    currency_pair="EURUSD",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# 2. Load existing price + sentiment data
price_sentiment_df = pd.read_parquet("data/prepared/EURUSD_1h.parquet")

# 3. Merge with fundamentals
full_dataset = merge_with_price_data(fundamentals, price_sentiment_df)

# 4. Save enhanced dataset
full_dataset.to_parquet("data/prepared/EURUSD_1h_with_fundamentals.parquet")
```

### Feature Engineering

The fundamental data can be used to create additional features:

```python
def add_fundamental_features(df):
    """Add derived features from fundamental data."""
    # Interest rate differential (for EURUSD)
    if 'economic_interest_rate' in df.columns:
        # Pivot to get one row per date with separate columns for each currency
        rate_pivot = df.pivot_table(
            index=df.index,
            columns='economic_currency',
            values='economic_interest_rate'
        )
        if 'EUR' in rate_pivot.columns and 'USD' in rate_pivot.columns:
            df['interest_rate_diff'] = rate_pivot['EUR'] - rate_pivot['USD']
    
    # Trade balance momentum
    if 'trade_trade_balance' in df.columns:
        df['trade_balance_change'] = df['trade_trade_balance'].diff()
    
    # Monetary shock intensity
    if 'shocks_MP_median' in df.columns:
        df['recent_shock_impact'] = df['shocks_MP_median'].rolling(window=30).sum()
    
    return df
```

## Cognee Integration

All downloaders include `format_for_cognee()` functions to create semantic search-friendly text:

```python
from data.downloaders.fred_downloader import get_forex_economic_indicators, format_for_cognee

# Download data
economic_data = get_forex_economic_indicators("EURUSD", "2023-01-01", "2023-12-31")

# Format for Cognee
formatted_data = format_for_cognee(economic_data)

# Each row now has a 'full_text' column like:
# "Federal Funds Rate for USD on 2023-01-01: 4.33. 
#  Indicator type: interest_rate. Series ID: FEDFUNDS."

# Ingest into Cognee
from data.cognee_processor import ingest_dataframe
ingest_dataframe(formatted_data, source="FRED_economic")
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Ensure packages are installed
pip install -e "new_data_sources/comtradeapicall"
pip install -e "new_data_sources/FRB"
```

### API Rate Limits

**UN Comtrade:**
- Free tier: 500 records per request
- Premium: 250,000 records per request
- Use `subscription_key` parameter for premium access

**FRED:**
- 120 requests per minute
- Use caching to avoid repeated calls

### Missing ECB Shock Data

Ensure CSV files exist:
```bash
ls "new_data_sources/jkshocks_update_ecb/"
# Should show:
# - shocks_ecb_mpd_me_d.csv
# - shocks_ecb_mpd_me_m.csv
```

## Data Quality Considerations

### Trade Data
- Monthly frequency may lag price movements
- Forward-fill to match higher frequency price data
- Some countries may have incomplete data

### Economic Indicators
- Different release schedules (some are lagging indicators)
- Revisions may occur (preliminary vs. final GDP)
- Seasonal adjustments vary by series

### ECB Shocks
- Event-based data (sparse time series)
- Zero values on non-meeting days
- EUR pairs only

## Next Steps

1. **Install packages**: Run `bash run/scripts/install_data_sources.sh`
2. **Set API keys**: Configure `COMTRADE_API_KEY` and `FRED_API_KEY`
3. **Test collection**: Run examples above to verify data access
4. **Integrate with training**: Modify `data/prepare_dataset.py` to include fundamentals
5. **Feature engineering**: Add fundamental-based features to your model

## References

- UN Comtrade API: https://comtradeplus.un.org/
- FRED API: https://fred.stlouisfed.org/docs/api/
- Jarocinski & Karadi (2020): https://doi.org/10.1257/mac.20180090
