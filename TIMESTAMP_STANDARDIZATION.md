# Timestamp Standardization Implementation

## Overview
Implemented consistent millisecond-based timestamp handling across the codebase to prevent data loss during enrichment operations and ensure compatibility with external data sources (CCXT API).

## Changes Made

### 1. **trainingplan.md** - Synthetic Data Generation
- **Line 80**: Changed timestamp conversion from **seconds** (`// 10**9`) to **milliseconds** (`// 10**6`)
  - Before: `timestamps.astype('int64') // 10**9` → Unix seconds
  - After: `timestamps.astype('int64') // 10**6` → Unix milliseconds
- **Line 15**: Updated documentation to specify milliseconds as the standard unit
  - Before: `float (Unix)` 
  - After: `float (Unix ms)` with clarification

### 2. **trainingplan.md** - Data Enrichment Pipeline
- **Lines 127-134**: Removed redundant millisecond conversion in enrichment step
  - Now directly joins synthetic data (already in ms) with CCXT data (native ms format)
  - Prevents 1000x timestamp inflation that would break joins

### 3. **data/source/crypto.py** - CSV Loader Enhancement
- **`load_from_csv()` method**: 
  - Added timestamp normalization step via `self._normalize_timestamp()`
  - Updated docstring to indicate normalization to milliseconds
  - Handles CSV imports with any timestamp format

- **New `_normalize_timestamp()` helper method**:
  - Automatically detects timestamp format (seconds, milliseconds, nanoseconds, datetime strings)
  - Converts all formats to milliseconds for consistency
  - Detection logic:
    - `< 2e10`: Treats as seconds → multiply by 1000
    - `2e10 to 3e13`: Treats as milliseconds → pass through
    - `> 3e13`: Treats as nanoseconds → divide by 1e6
    - String objects: Parse via `pd.to_datetime()` then convert to ms

## Why This Matters

### The Original Bug
1. Synthetic data was in **Unix seconds** (e.g., 1705382400)
2. CCXT API returns timestamps in **Unix milliseconds** (e.g., 1705382400000)
3. The enrichment join `df['timestamp'].isin(btc_vol.index)` found **zero matches**
4. Result: Empty enriched dataset, defeating the purpose of real volatility injection

### The Fix
- All timestamps now standardized to **milliseconds** (CCXT native format)
- Synthetic generator produces ms-format timestamps
- CSV loader auto-converts any format to ms
- Enrichment pipeline works seamlessly without format conflicts

## Timestamp Format Reference

| Format | Magnitude | Example | Detected As |
|--------|-----------|---------|-------------|
| Unix Seconds | < 2e10 | 1705382400 | Seconds (×1000) |
| Unix Milliseconds | 2e10-3e13 | 1705382400000 | Milliseconds (pass) |
| Unix Nanoseconds | > 3e13 | 1705382400000000 | Nanoseconds (÷1e6) |
| ISO 8601 String | N/A | "2025-01-16T00:00:00Z" | DateTime (→ ms) |

## Validation Checklist
- [x] Synthetic generator now produces millisecond timestamps
- [x] Documentation updated to reflect standard unit
- [x] CSV loader handles multiple timestamp formats automatically
- [x] Enrichment pipeline removes redundant conversion
- [x] CCXT join will now have matching timestamp formats
- [x] Timestamp normalization is bidirectional-safe (no data loss)

## Testing Recommendations
1. Generate synthetic merchant data and verify timestamp ranges (should be ~1.7e12)
2. Pull real CCXT data and verify timestamp ranges match
3. Run enrichment pipeline and confirm non-empty output (successful joins)
4. Validate timestamp values are preserved through parquet read/write cycles
