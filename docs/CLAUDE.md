# CLAUDE.md - Internal Tool Context

## About This File

This file contains an auto-generated `<claude-mem-context>` block used by the claude-mem context system to populate memory snippets for the AI assistant. 

**Important**: 
- The `<claude-mem-context>` section below is **auto-generated** and should **not be manually edited**
- This section stores recent activity summaries to provide context across conversations
- The canonical source is the claude-mem internal tool system
- This file can be added to `.gitignore` if you prefer not to commit auto-generated content

---

## Setting Up Your Development Environment

### Environment Variables and Secrets Management

When working with external APIs (FRED, Comtrade, ECB data), you'll need API keys and credentials. Follow these security best practices:

**Variable Leakage Prevention:**
1. **Never log or print environment contents**: Load your `.env` file silently via `python-dotenv` or similar. Do not log the contents.
   ```python
   from dotenv import load_dotenv
   load_dotenv()  # Silent load
   api_key = os.getenv("FRED_API_KEY")  # Use, don't print
   ```

2. **Prevent environment visibility in process listings**: Avoid launching child processes with exposed environment variables:
   ```bash
   # ❌ BAD: env var visible in ps output
   FRED_API_KEY=secret_key python script.py
   
   # ✅ GOOD: use .env file or process manager masking
   python script.py  # reads from .env silently
   ```

3. **Use secure secret management**:
   - Prefer `python-dotenv-safe` or similar validation libraries
   - For production, use OS secret stores (AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
   - Never commit `.env` files to version control

4. **Verify `.gitignore` configuration**:
   - Ensure `.env` is listed in `.gitignore` to prevent accidental commits
   - Example entries: `.env`, `.env.local`, `.env.*.local`

5. **Placeholder examples only**:
   - When documenting API key usage, always show placeholders, never real values
   - Example: `export FRED_API_KEY='your_fred_api_key_here'` (not an actual key)
   - Example: `export COMTRADE_API_KEY='your_comtrade_key_here'` (not an actual key)

### Loading API Keys Safely

**Example Setup:**

1. Create `.env` file (in `.gitignore`):
   ```bash
   FRED_API_KEY=your_actual_key_here
   COMTRADE_API_KEY=your_actual_key_here
   ```

2. Load in your code:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()  # Load silently
   fred_key = os.getenv("FRED_API_KEY")
   comtrade_key = os.getenv("COMTRADE_API_KEY")
   # Use keys without logging them
   ```

### API Key Documentation

- **FRED_API_KEY**: Your Federal Reserve Economic Data API key (get from https://fred.stlouisfed.org/docs/api/)
- **COMTRADE_API_KEY**: Your UN Comtrade API key (for full access, otherwise preview mode applies)

---

## Monitoring Data Staleness

Fundamental data can become outdated, especially during high market volatility. The framework provides utilities to detect and flag stale data:

### Example: Validating Fundamental Data Freshness

```python
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def validate_fundamental_freshness(
    df: pd.DataFrame,
    max_age_days: int = 30,
    volatility_threshold: float = 0.02
) -> dict[str, int]:
    """
    Validate that fundamental data is fresh and identify high-volatility stale periods.
    
    Args:
        df: DataFrame with 'returns' column and 'last_fundamental_update' timestamp
        max_age_days: Maximum acceptable age of fundamental data in days
        volatility_threshold: Threshold for rolling volatility (std of returns)
    
    Returns:
        Dictionary with validation results including affected row count and max age
    
    Example:
        >>> df = pd.DataFrame({
        ...     'returns': [0.01, -0.02, 0.015, ...],
        ...     'last_fundamental_update': pd.date_range('2023-01-01', periods=len(df))
        ... })
        >>> result = validate_fundamental_freshness(df, max_age_days=30, volatility_threshold=0.02)
        >>> if result['flagged_rows'] > 0:
        ...     logger.warning(f"Found {result['flagged_rows']} stale high-volatility rows")
    
    """
    results = {
        'flagged_rows': 0,
        'max_age_days': 0,
        'high_volatility_periods': 0
    }
    
    if df.empty:
        return results
    
    # Make a copy to avoid mutating input DataFrame
    df = df.copy()
    
    # Compute rolling volatility (20-period standard deviation of returns)
    if 'returns' in df.columns:
        df['volatility'] = df['returns'].rolling(window=20).std()
        high_volatility = df['volatility'] > volatility_threshold
    else:
        high_volatility = pd.Series([False] * len(df), index=df.index)
    
    # Compute data age from last_fundamental_update
    if 'last_fundamental_update' in df.columns:
        df['last_fundamental_update'] = pd.to_datetime(df['last_fundamental_update'])
        now = pd.Timestamp.utcnow()
        df['data_age_days'] = (now - df['last_fundamental_update']).dt.days
        stale_data = df['data_age_days'] > max_age_days
    else:
        stale_data = pd.Series([False] * len(df), index=df.index)
        df['data_age_days'] = 0
    
    # Flag rows that are BOTH high-volatility AND stale
    flagged = high_volatility & stale_data
    results['flagged_rows'] = flagged.sum()
    results['max_age_days'] = df['data_age_days'].max() if 'data_age_days' in df.columns else 0
    results['high_volatility_periods'] = high_volatility.sum()
    
    # Emit warning if any rows are flagged
    if results['flagged_rows'] > 0:
        logger.warning(
            f"[Staleness Check] Found {results['flagged_rows']} high-volatility rows with "
            f"stale fundamentals (max age: {results['max_age_days']} days, "
            f"volatility_threshold: {volatility_threshold}). "
            f"Consider refreshing fundamental data before training."
        )
    
    return results
```

### Usage in Training Pipeline

Call this validation after loading fundamental data to catch potential issues early:

```python
fundamentals = collect_all_forex_fundamentals(pair, start_date, end_date)

if 'trade' in fundamentals and not fundamentals['trade'].empty:
    staleness_check = validate_fundamental_freshness(
        fundamentals['trade'],
        max_age_days=30,
        volatility_threshold=0.02
    )
    # Log warning if many stale rows detected
    stale_count = staleness_check.get('flagged_rows', 0)
    total_rows = len(fundamentals['trade'])
    if stale_count > 0:
        stale_ratio = stale_count / total_rows if total_rows > 0 else 0
        if stale_ratio > 0.2 or stale_count > 10:
            logger.warning(
                f"High stale data ratio detected: {stale_count}/{total_rows} "
                f"({stale_ratio:.1%}) rows flagged as high-volatility and stale. "
                f"Consider updating or validating data sources."
            )
```

---

<claude-mem-context>
# Recent Activity

<!-- This section is auto-generated by claude-mem. Edit content outside the tags. -->

*No recent activity*
</claude-mem-context>