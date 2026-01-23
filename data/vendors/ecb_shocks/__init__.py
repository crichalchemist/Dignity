"""
ECB Monetary Policy Shocks - Vendored Data Package

This package provides access to the ECB monetary policy shock data from
Jarociński & Karadi (2020), "Deconstructing Monetary Policy Surprises—The Role
of Information Shocks", American Economic Journal: Macroeconomics 12(2): 1-43.

The data includes:
- Daily shocks (shocks_ecb_mpd_me_d.csv)
- Monthly shocks (shocks_ecb_mpd_me_m.csv)
"""

from pathlib import Path

__version__ = "2020.1"  # Based on Jarociński & Karadi (2020) dataset
__author__ = "Marek Jarociński and Peter Karadi"
__citation__ = (
    "Jarociński, M., & Karadi, P. (2020). Deconstructing Monetary Policy Surprises—"
    "The Role of Information Shocks. American Economic Journal: Macroeconomics, 12(2), 1-43. "
    "https://doi.org/10.1257/mac.20180090"
)


def get_data_dir() -> Path:
    """
    Get the directory containing ECB shocks CSV files.

    Returns:
        Path: Absolute path to the data directory containing vendored CSV files.
              The path is resolved to ensure it is absolute regardless of the
              current working directory.
    """
    return (Path(__file__).parent / "data").resolve()


def get_daily_shocks_path() -> Path:
    """
    Get path to daily ECB monetary policy shocks CSV file.

    Returns:
        Path: Absolute path to shocks_ecb_mpd_me_d.csv
    """
    return get_data_dir() / "shocks_ecb_mpd_me_d.csv"


def get_monthly_shocks_path() -> Path:
    """
    Get path to monthly ECB monetary policy shocks CSV file.

    Returns:
        Path: Absolute path to shocks_ecb_mpd_me_m.csv
    """
    return get_data_dir() / "shocks_ecb_mpd_me_m.csv"


def verify_data_exists() -> bool:
    """
    Verify that all required CSV files exist in the vendored data directory.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    return get_daily_shocks_path().exists() and get_monthly_shocks_path().exists()


# Convenience exports
__all__ = [
    "get_data_dir",
    "get_daily_shocks_path",
    "get_monthly_shocks_path",
    "verify_data_exists",
    "__version__",
    "__author__",
    "__citation__",
]
