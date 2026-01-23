"""Centralised logging utilities for the Sequence codebase.

The project currently uses many ``print`` statements for diagnostics.  The
roadmap calls for a proper ``logging`` framework with configurable verbosity.
This module provides a ``get_logger`` helper that creates a module‑level logger
named after the caller's ``__name__``.  The logger is configured on first use
with a stream handler that respects the ``SEQ_LOG_LEVEL`` environment variable:

* ``CRITICAL`` – default level, shows only critical errors.
* ``ERROR`` – shows errors.
* ``WARNING`` – shows warnings.
* ``INFO`` – typical informational messages (replaces most ``print`` calls).
* ``DEBUG`` – very verbose output for debugging.

Usage example:

>>> from utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("Data loading started")

All existing modules should replace direct ``print`` calls with the appropriate
logger method (``debug``, ``info``, ``warning``, ``error``).
"""

from __future__ import annotations

import logging
import os
from typing import Final

_DEFAULT_LEVEL: Final[int] = logging.CRITICAL

_LOGGER_CACHE: dict[str, logging.Logger] = {}


def _initial_level() -> int:
    """Resolve the initial log level from ``SEQ_LOG_LEVEL``.

    The environment variable is optional; if it is missing or invalid the
    function falls back to ``_DEFAULT_LEVEL``.
    """
    level_name = os.getenv("SEQ_LOG_LEVEL", "CRITICAL").upper()
    return getattr(logging, level_name, _DEFAULT_LEVEL)


def _configure_root_logger() -> None:
    """Configure the root logger exactly once.

    The function sets a simple ``StreamHandler`` with a concise format.  It is
    idempotent – subsequent calls are no‑ops.
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured.
        return
    root.setLevel(_initial_level())
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a cached logger for *name*.

    The first call configures the global logging system via ``_configure_root_logger``.
    Subsequent calls retrieve the logger from an internal cache to avoid duplicate
    handlers.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    _configure_root_logger()
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    return logger
