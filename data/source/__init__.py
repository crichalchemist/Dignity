"""Data source interfaces."""

from .crypto import CryptoSource
from .synthetic import SyntheticGenerator

__all__ = ["SyntheticGenerator", "CryptoSource"]
