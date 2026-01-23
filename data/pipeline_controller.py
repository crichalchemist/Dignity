"""Unified data pipeline controller for collection, preprocessing, and validation."""
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("DataPipeline")

DB_PATH = Path("output_central/data_pipeline.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data collection configuration."""
    data_sources: list[str]  # GDELT, YFinance, HistData
    symbols: list[str]
    countries: list[str]
    start_date: str
    end_date: str
    resolution: str  # daily, 1hour, 5min, 1min
    preprocessing: dict[str, Any]


class DataPipelineController:
    """Unified interface for data collection, preprocessing, and validation."""

    def __init__(self):
        self.init_database()
        self.cache = {}
        self.validation_rules = {
            'min_rows': 100,
            'max_missing_percent': 0.05,
            'price_bounds': (0.001, 1000000),
            'volume_bounds': (0, 1000000000)
        }

    def init_database(self):
        """Initialize SQLite database for data pipeline tracking."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Data collection jobs
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS data_collections
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           collection_id
                           TEXT
                           UNIQUE,
                           data_source
                           TEXT,
                           symbol
                           TEXT,
                           start_date
                           TEXT,
                           end_date
                           TEXT,
                           resolution
                           TEXT,
                           rows_collected
                           INTEGER,
                           status
                           TEXT
                           DEFAULT
                           'PENDING',
                           error_message
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           completed_at
                           TIMESTAMP
                       )
                       ''')

        # Data preprocessing jobs
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS data_preprocessing
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           preprocessing_id
                           TEXT
                           UNIQUE,
                           collection_id
                           TEXT,
                           normalization_method
                           TEXT,
                           feature_engineering
                           TEXT,
                           rows_processed
                           INTEGER,
                           status
                           TEXT
                           DEFAULT
                           'PENDING',
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           completed_at
                           TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           collection_id
                       ) REFERENCES data_collections
                       (
                           collection_id
                       )
                           )
                       ''')

        # Data validation results
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS data_validation
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           validation_id
                           TEXT
                           UNIQUE,
                           collection_id
                           TEXT,
                           rows_checked
                           INTEGER,
                           missing_values
                           INTEGER,
                           missing_percent
                           REAL,
                           outliers_detected
                           INTEGER,
                           quality_score
                           REAL,
                           status
                           TEXT,
                           issues_json
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           FOREIGN
                           KEY
                       (
                           collection_id
                       ) REFERENCES data_collections
                       (
                           collection_id
                       )
                           )
                       ''')

        # Dataset versions for reproducibility
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS dataset_versions
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           version_id
                           TEXT
                           UNIQUE,
                           dataset_name
                           TEXT,
                           rows
                           INTEGER,
                           columns
                           INTEGER,
                           features_list
                           TEXT,
                           file_path
                           TEXT,
                           file_size
                           INTEGER,
                           checksum
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        conn.commit()
        conn.close()
        logger.info(f"Data pipeline database initialized at {DB_PATH}")

    def collect_data(
            self,
            config: DataConfig,
            data_source: str = None
    ) -> tuple[pd.DataFrame | None, str]:
        """Collect data from specified source."""
        try:
            collection_id = f"{datetime.now().isoformat()}_{config.symbols[0]}"

            # Record collection job
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            source = data_source or config.data_sources[0]

            cursor.execute('''
                           INSERT INTO data_collections
                           (collection_id, data_source, symbol, start_date, end_date,
                            resolution, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               collection_id, source, config.symbols[0],
                               config.start_date, config.end_date,
                               config.resolution, 'RUNNING'
                           ))

            conn.commit()
            conn.close()

            # Collect based on source
            if source == 'GDELT':
                data = self._collect_gdelt(config, collection_id)
            elif source == 'YFinance':
                data = self._collect_yfinance(config, collection_id)
            elif source == 'HistData':
                data = self._collect_histdata(config, collection_id)
            else:
                data = None

            if data is not None:
                self._update_collection_status(collection_id, 'COMPLETED', len(data))
                logger.info(f"Data collected: {collection_id} - {len(data)} rows")
                return data, collection_id
            else:
                self._update_collection_status(collection_id, 'FAILED', error="Data retrieval failed")
                return None, collection_id

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            return None, None

    def _collect_gdelt(self, config: DataConfig, collection_id: str) -> pd.DataFrame | None:
        """Collect GDELT data."""
        try:
            from datetime import datetime as dt

            from data.gdelt.consolidated_downloader import GDELTDownloader

            downloader = GDELTDownloader()
            start_dt = dt.strptime(config.start_date, '%Y-%m-%d')
            end_dt = dt.strptime(config.end_date, '%Y-%m-%d')

            data = downloader.download_daterange(
                start_dt, end_dt, config.countries, config.resolution
            )

            return data if data is not None and not data.empty else None

        except Exception as e:
            logger.error(f"GDELT collection failed: {e}")
            return None

    def _collect_yfinance(self, config: DataConfig, collection_id: str) -> pd.DataFrame | None:
        """Collect YFinance FX data."""
        try:

            import yfinance as yf

            all_data = []

            for symbol in config.symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=config.start_date,
                    end=config.end_date,
                    interval=self._convert_resolution(config.resolution)
                )

                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)

            if all_data:
                return pd.concat(all_data, ignore_index=False)
            return None

        except Exception as e:
            logger.error(f"YFinance collection failed: {e}")
            return None

    def _collect_histdata(self, config: DataConfig, collection_id: str) -> pd.DataFrame | None:
        """Collect HistData FX data."""
        try:
            # Implementation would connect to HistData API
            logger.warning("HistData collection not yet implemented")
            return None

        except Exception as e:
            logger.error(f"HistData collection failed: {e}")
            return None

    def collect_fundamental_data(
            self,
            currency_pair: str,
            start_date: str,
            end_date: str,
            comtrade_api_key: str | None = None,
            fred_api_key: str | None = None,
            include_sources: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Collect fundamental economic data for a forex pair.

        Args:
            currency_pair: Forex pair (e.g., "EURUSD")
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            comtrade_api_key: UN Comtrade API key
            fred_api_key: FRED API key
            include_sources: Optional list of sources ["trade", "economic", "shocks"]

        Returns:
            Dictionary with DataFrames for each source
        """
        # Generate collection_id once before try block for both success/failure paths
        collection_id = f"{datetime.now().isoformat()}_{currency_pair}_fundamentals"
        try:
            from data.extended_data_collection import collect_all_forex_fundamentals

            logger.info(f"Collecting fundamental data for {currency_pair}")

            data = collect_all_forex_fundamentals(
                currency_pair=currency_pair,
                start_date=start_date,
                end_date=end_date,
                comtrade_api_key=comtrade_api_key,
                fred_api_key=fred_api_key,
                include_sources=include_sources
            )

            # Record in database (use same collection_id for both success/failure)
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            total_rows = sum(len(df) for df in data.values() if isinstance(df, pd.DataFrame) and not df.empty)

            cursor.execute('''
                           INSERT INTO data_collections
                           (collection_id, data_source, symbol, start_date, end_date,
                            resolution, rows_collected, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               collection_id, 'FUNDAMENTALS', currency_pair,
                               start_date, end_date, 'mixed', total_rows, 'COMPLETED'
                           ))

            conn.commit()
            conn.close()

            logger.info(f"Collected {total_rows} fundamental records for {currency_pair}")
            return data

        except Exception as e:
            logger.error(f"Fundamental data collection failed: {e}")

            # Update database to reflect failure with error message
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO data_collections
                    (collection_id, data_source, symbol, start_date, end_date,
                     resolution, rows_collected, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        collection_id,
                        'FUNDAMENTALS', currency_pair, start_date, end_date,
                        'mixed', 0, 'FAILED', str(e)
                    ))
                conn.commit()
                conn.close()
            except Exception as db_err:
                logger.error(f"Failed to record collection failure in database: {db_err}")

            return {}

    def preprocess(
            self,
            data: pd.DataFrame,
            collection_id: str,
            config: dict[str, Any]
    ) -> tuple[pd.DataFrame | None, str]:
        """Preprocess collected data."""
        try:
            preprocessing_id = f"prep_{datetime.now().isoformat()}"

            # Record preprocessing job
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO data_preprocessing
                               (preprocessing_id, collection_id, status)
                           VALUES (?, ?, ?)
                           ''', (preprocessing_id, collection_id, 'RUNNING'))

            conn.commit()
            conn.close()

            # Apply preprocessing steps
            processed = data.copy()

            # Normalization
            if config.get('normalize'):
                processed = self._normalize(processed, config.get('normalization_method', 'zscore'))

            # Feature engineering
            if config.get('engineer_features'):
                processed = self._engineer_features(processed)

            # Handle missing values
            processed = processed.fillna(method='ffill').fillna(method='bfill')

            # Update status
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           UPDATE data_preprocessing
                           SET status         = ?,
                               rows_processed = ?,
                               completed_at   = ?
                           WHERE preprocessing_id = ?
                           ''', ('COMPLETED', len(processed), datetime.now().isoformat(), preprocessing_id))

            conn.commit()
            conn.close()

            logger.info(f"Preprocessing completed: {preprocessing_id}")
            return processed, preprocessing_id

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None, None

    def validate(
            self,
            data: pd.DataFrame,
            collection_id: str
    ) -> tuple[dict[str, Any], str]:
        """Validate data quality."""
        try:
            validation_id = f"val_{datetime.now().isoformat()}"

            issues = []

            # Check row count
            if len(data) < self.validation_rules['min_rows']:
                issues.append(f"Insufficient rows: {len(data)} < {self.validation_rules['min_rows']}")

            # Check missing values
            missing_percent = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_percent > self.validation_rules['max_missing_percent']:
                issues.append(f"Too many missing values: {missing_percent:.2%}")

            # Check data types and bounds
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'price' in col.lower() or 'close' in col.lower():
                    prices = data[col].dropna()
                    if (prices < self.validation_rules['price_bounds'][0]).any() or \
                            (prices > self.validation_rules['price_bounds'][1]).any():
                        issues.append(f"Price out of bounds in {col}")

            # Detect outliers (3-sigma rule)
            outliers = 0
            for col in numeric_cols:
                col_data = data[col].dropna()
                mean = col_data.mean()
                std = col_data.std()
                outliers += ((col_data < mean - 3 * std) | (col_data > mean + 3 * std)).sum()

            quality_score = max(0, 100 - (len(issues) * 10) - (outliers / len(data) * 100))
            status = 'PASS' if quality_score >= 70 else 'WARN' if quality_score >= 50 else 'FAIL'

            # Record validation
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO data_validation
                           (validation_id, collection_id, rows_checked, missing_values,
                            missing_percent, outliers_detected, quality_score, status, issues_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               validation_id, collection_id, len(data),
                               int(data.isnull().sum().sum()), missing_percent,
                               outliers, quality_score, status, json.dumps(issues)
                           ))

            conn.commit()
            conn.close()

            result = {
                "validation_id": validation_id,
                "rows_checked": len(data),
                "quality_score": quality_score,
                "status": status,
                "issues": issues
            }

            logger.info(f"Validation completed: {validation_id} - Score: {quality_score:.1f}")
            return result, validation_id

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}, None

    def _normalize(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Normalize data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if method == 'zscore':
            for col in numeric_cols:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    data[col] = (data[col] - mean) / std

        elif method == 'minmax':
            for col in numeric_cols:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)

        return data

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        # Add moving averages if price data exists
        price_cols = [col for col in data.columns if 'close' in col.lower()]

        for col in price_cols:
            if col in data.columns:
                data[f'{col}_ma_5'] = data[col].rolling(5).mean()
                data[f'{col}_ma_20'] = data[col].rolling(20).mean()
                data[f'{col}_volatility'] = data[col].rolling(20).std()

        return data

    def _convert_resolution(self, resolution: str) -> str:
        """Convert resolution format."""
        mapping = {
            'daily': '1d',
            '1hour': '1h',
            '5min': '5m',
            '1min': '1m'
        }
        return mapping.get(resolution, resolution)

    def _update_collection_status(
            self,
            collection_id: str,
            status: str,
            rows: int = None,
            error: str = None
    ):
        """Update collection job status."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            if error:
                cursor.execute('''
                               UPDATE data_collections
                               SET status        = ?,
                                   error_message = ?,
                                   completed_at  = ?
                               WHERE collection_id = ?
                               ''', (status, error, datetime.now().isoformat(), collection_id))
            else:
                cursor.execute('''
                               UPDATE data_collections
                               SET status         = ?,
                                   rows_collected = ?,
                                   completed_at   = ?
                               WHERE collection_id = ?
                               ''', (status, rows, datetime.now().isoformat(), collection_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update collection status: {e}")

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get overall pipeline status."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM data_collections WHERE status = 'COMPLETED'")
            completed_collections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM data_preprocessing WHERE status = 'COMPLETED'")
            completed_preprocessing = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM data_validation WHERE status = 'PASS'")
            passed_validations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM dataset_versions")
            dataset_versions = cursor.fetchone()[0]

            conn.close()

            return {
                "collections_completed": completed_collections,
                "preprocessing_completed": completed_preprocessing,
                "validations_passed": passed_validations,
                "dataset_versions": dataset_versions,
                "pipeline_status": "OPERATIONAL"
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {}


# Global controller instance
controller = DataPipelineController()
