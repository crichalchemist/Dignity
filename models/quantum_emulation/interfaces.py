"""Interface definitions for quantum emulation modules."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class EmulationMetadata:
    """Lightweight metadata for emulation outputs."""

    backend: str
    n_features: int
    notes: str | None = None


class LatentStateModel(ABC):
    """Transforms sequential inputs into latent state features."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LatentStateModel":
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y=y)
        return self.transform(X)


class DistributionModel(ABC):
    """Generative distribution model for scenario simulation."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> "DistributionModel":
        raise NotImplementedError

    @abstractmethod
    def sample(self, context: dict[str, Any] | None, n_paths: int, horizon: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pdf(self, context: dict[str, Any] | None, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DiscreteOptimizer(Protocol):
    """Protocol for discrete optimization backends."""

    def solve(self, objective: np.ndarray, constraints: dict[str, Any]) -> dict[str, Any]:
        ...


__all__ = [
    "EmulationMetadata",
    "LatentStateModel",
    "DistributionModel",
    "DiscreteOptimizer",
]
