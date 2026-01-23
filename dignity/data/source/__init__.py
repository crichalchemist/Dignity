"""Data source interfaces."""
from .synthetic import SyntheticGenerator
from .crypto import CryptoSource

__all__ = ["SyntheticGenerator", "CryptoSource"]
