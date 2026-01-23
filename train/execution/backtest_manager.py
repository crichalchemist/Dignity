"""Backtesting manager with comparison and result storage."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from backtesting import Backtest

from config.constants import DEFAULT_BACKTEST_CASH, DEFAULT_COMMISSION_RATE

logger = logging.getLogger("BacktestManager")

DB_PATH = Path("output_central/backtest_results.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class BacktestManager:
    """Manage backtesting with comparison and storage."""

    def __init__(self):
        self.init_database()
        self.results_cache = {}

    def init_database(self):
        """Initialize SQLite database for backtest results."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS backtest_runs
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           run_id
                           TEXT
                           UNIQUE,
                           strategy_name
                           TEXT
                           NOT
                           NULL,
                           symbol
                           TEXT
                           NOT
                           NULL,
                           timeframe
                           TEXT,
                           start_date
                           TEXT,
                           end_date
                           TEXT,
                           cash
                           INTEGER,
                           commission
                           REAL,
                           total_return
                           REAL,
                           sharpe_ratio
                           REAL,
                           sortino_ratio
                           REAL,
                           max_drawdown
                           REAL,
                           win_rate
                           REAL,
                           trades_count
                           INTEGER,
                           avg_trade_duration
                           TEXT,
                           best_trade
                           REAL,
                           worst_trade
                           REAL,
                           profit_factor
                           REAL,
                           equity_peak
                           REAL,
                           result_json
                           TEXT,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       """)

        # Comparison table for side-by-side analysis
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS backtest_comparisons
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           comparison_id
                           TEXT
                           UNIQUE,
                           strategy_1_id
                           INTEGER,
                           strategy_2_id
                           INTEGER,
                           comparison_date
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           winner
                           TEXT,
                           delta_return
                           REAL,
                           delta_sharpe
                           REAL,
                           delta_drawdown
                           REAL,
                           FOREIGN
                           KEY
                       (
                           strategy_1_id
                       ) REFERENCES backtest_runs
                       (
                           id
                       ),
                           FOREIGN KEY
                       (
                           strategy_2_id
                       ) REFERENCES backtest_runs
                       (
                           id
                       )
                           )
                       """)

        conn.commit()
        conn.close()
        logger.info(f"Backtest database initialized at {DB_PATH}")

    def run_backtest(
            self,
            data: pd.DataFrame,
            strategy_class: type,
            strategy_name: str,
            symbol: str,
            cash: int = DEFAULT_BACKTEST_CASH,
            commission: float = DEFAULT_COMMISSION_RATE,
            **strategy_params,
    ) -> dict[str, Any]:
        """Run backtest with strategy."""
        """Run backtest with strategy."""
        try:
            bt = Backtest(data, strategy_class, cash=cash, commission=commission)
            stats = bt.run(**strategy_params)

            # Extract metrics
            result = {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "total_return": float(stats["Return [%]"]) if "Return [%]" in stats else 0,
                "sharpe_ratio": float(stats["Sharpe Ratio"]) if "Sharpe Ratio" in stats else 0,
                "sortino_ratio": float(stats["Sortino Ratio"]) if "Sortino Ratio" in stats else 0,
                "max_drawdown": float(stats["Max. Drawdown [%]"])
                if "Max. Drawdown [%]" in stats
                else 0,
                "win_rate": float(stats["Win Rate [%]"]) if "Win Rate [%]" in stats else 0,
                "trades": int(stats["# Trades"]) if "# Trades" in stats else 0,
                "equity_final": float(stats["Equity Final [$]"])
                if "Equity Final [$]" in stats
                else 0,
                "_stats": stats,  # Store full stats for detailed analysis
            }

            logger.info(
                f"Backtest complete: {strategy_name} {symbol} - Return: {result['total_return']}%"
            )
            return result

        except (KeyError, ValueError) as e:
            logger.error(f"Backtest data error for {strategy_name} on {symbol}: {e}")
            return None
        except (AttributeError, TypeError) as e:
            logger.error(f"Backtest configuration error for {strategy_name}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected backtest error for {strategy_name} on {symbol}: {e}")
            return None

    def save_result(
            self,
            run_id: str,
            strategy_name: str,
            symbol: str,
            timeframe: str,
            start_date: str,
            end_date: str,
            cash: int,
            commission: float,
            result: dict[str, Any],
    ) -> bool:
        """Save backtest result to database."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO backtest_runs
                (run_id, strategy_name, symbol, timeframe, start_date, end_date,
                 cash, commission, total_return, sharpe_ratio, sortino_ratio,
                 max_drawdown, win_rate, trades_count, equity_peak, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    strategy_name,
                    symbol,
                    timeframe,
                    start_date,
                    end_date,
                    cash,
                    commission,
                    result.get("total_return", 0),
                    result.get("sharpe_ratio", 0),
                    result.get("sortino_ratio", 0),
                    result.get("max_drawdown", 0),
                    result.get("win_rate", 0),
                    result.get("trades", 0),
                    result.get("equity_final", 0),
                    json.dumps(result),
                ),
            )

            conn.commit()
            conn.close()

            self.results_cache[run_id] = result
            logger.info(f"Result saved: {run_id}")
            return True

        except sqlite3.IntegrityError as e:
            logger.error(
                f"Database integrity error saving result {run_id}: {e} (duplicate run_id?)"
            )
            return False
        except sqlite3.OperationalError as e:
            logger.error(f"Database operational error saving result {run_id}: {e}")
            return False
        except (OSError, PermissionError) as e:
            logger.error(f"File system error accessing database: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error saving result {run_id}: {e}")
            return False

    def compare_strategies(self, run_id_1: str, run_id_2: str) -> dict[str, Any]:
        """Compare two backtest results."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Get both results
            cursor.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id_1,))
            result1 = cursor.fetchone()

            cursor.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id_2,))
            result2 = cursor.fetchone()

            if not result1 or not result2:
                logger.warning(f"One or both run_ids not found: {run_id_1}, {run_id_2}")
                return None

            # Extract key metrics
            r1_return = float(result1[9])
            r2_return = float(result2[9])

            r1_sharpe = float(result1[10])
            r2_sharpe = float(result2[10])

            r1_drawdown = float(result1[12])
            r2_drawdown = float(result2[12])

            # Determine winner (higher return, higher Sharpe, lower drawdown)
            winner_score_1 = (r1_return * 0.4) + (r1_sharpe * 0.4) + ((100 - r1_drawdown) * 0.2)
            winner_score_2 = (r2_return * 0.4) + (r2_sharpe * 0.4) + ((100 - r2_drawdown) * 0.2)

            winner = result1[2] if winner_score_1 > winner_score_2 else result2[2]

            comparison = {
                "strategy_1": result1[2],
                "strategy_2": result2[2],
                "symbol": result1[4],
                "winner": winner,
                "comparison_date": datetime.now().isoformat(),
                "metrics": {
                    "return": {
                        "strategy_1": r1_return,
                        "strategy_2": r2_return,
                        "delta": r1_return - r2_return,
                    },
                    "sharpe_ratio": {
                        "strategy_1": r1_sharpe,
                        "strategy_2": r2_sharpe,
                        "delta": r1_sharpe - r2_sharpe,
                    },
                    "max_drawdown": {
                        "strategy_1": r1_drawdown,
                        "strategy_2": r2_drawdown,
                        "delta": r1_drawdown - r2_drawdown,
                    },
                    "trades": {"strategy_1": int(result1[14]), "strategy_2": int(result2[14])},
                },
            }

            conn.close()
            logger.info(f"Comparison complete: {result1[2]} vs {result2[2]}")
            return comparison

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Database error during comparison: {e}")
            return None
        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Data extraction error during comparison: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error comparing {run_id_1} vs {run_id_2}: {e}")
            return None

    def get_results_dataframe(self, symbol: str = None, limit: int = 20) -> pd.DataFrame:
        """Get results as DataFrame for analysis."""
        try:
            conn = sqlite3.connect(DB_PATH)

            query = "SELECT * FROM backtest_runs"
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            return df

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Database error retrieving results: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"Unexpected error retrieving results: {e}")
            return pd.DataFrame()

    def export_comparison_csv(self, run_id_1: str, run_id_2: str, output_path: str) -> bool:
        """Export comparison to CSV."""
        try:
            comparison = self.compare_strategies(run_id_1, run_id_2)
            if not comparison:
                return False

            df = pd.DataFrame([comparison])
            df.to_csv(output_path, index=False)
            logger.info(f"Comparison exported to {output_path}")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"File system error exporting to {output_path}: {e}")
            return False
        except ValueError as e:
            logger.error(f"Data conversion error during export: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error exporting comparison: {e}")
            return False

    def get_portfolio_stats(self) -> dict[str, Any]:
        """Get aggregated portfolio statistics."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM backtest_runs")
            total_runs = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(total_return) FROM backtest_runs")
            avg_return = cursor.fetchone()[0] or 0

            cursor.execute("SELECT AVG(sharpe_ratio) FROM backtest_runs")
            avg_sharpe = cursor.fetchone()[0] or 0

            cursor.execute("SELECT AVG(max_drawdown) FROM backtest_runs")
            avg_drawdown = cursor.fetchone()[0] or 0

            cursor.execute("SELECT DISTINCT symbol FROM backtest_runs")
            symbols = [row[0] for row in cursor.fetchall()]

            cursor.execute("SELECT DISTINCT strategy_name FROM backtest_runs")
            strategies = [row[0] for row in cursor.fetchall()]

            conn.close()

            return {
                "total_runs": total_runs,
                "avg_return": avg_return,
                "avg_sharpe": avg_sharpe,
                "avg_drawdown": avg_drawdown,
                "unique_symbols": len(symbols),
                "unique_strategies": len(strategies),
                "symbols": symbols,
                "strategies": strategies,
            }

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.error(f"Database error retrieving portfolio stats: {e}")
            return {}
        except Exception as e:
            logger.exception(f"Unexpected error retrieving portfolio stats: {e}")
            return {}


# Global manager instance
manager = BacktestManager()
