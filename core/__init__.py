"""Core utilities: config, signals, privacy."""
from .config import DignityConfig
from .signals import SignalProcessor
from .privacy import PrivacyManager

__all__ = ["DignityConfig", "SignalProcessor", "PrivacyManager"]
