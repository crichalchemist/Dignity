"""Data processors for alignment and feature engineering.

This package contains processor components for the Dignity Foundation Model data pipeline.
The legacy TransactionPipeline is in data/pipeline.py at the parent level.
"""

from .alignment import TimestampAligner

__all__ = ['TimestampAligner']
