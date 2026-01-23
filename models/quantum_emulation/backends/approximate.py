"""Approximate reservoir backend for larger qubit counts.

Uses a fixed random recurrent map as a fallback when statevector simulation
would be too expensive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ApproximateConfig:
    n_units: int
    spectral_radius: float = 0.9
    input_scale: float = 0.5
    seed: int = 42


class ApproximateReservoir:
    """Random recurrent map emulating reservoir dynamics."""

    def __init__(self, cfg: ApproximateConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        w = rng.normal(scale=1.0, size=(cfg.n_units, cfg.n_units))
        # Scale by spectral radius.
        eigenvalues = np.linalg.eigvals(w)
        max_radius = np.max(np.abs(eigenvalues))
        self.w = (w / max_radius) * cfg.spectral_radius
        self.state = np.zeros(cfg.n_units, dtype=np.float64)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, inputs: np.ndarray) -> np.ndarray:
        drive = self.cfg.input_scale * inputs
        if drive.shape[0] != self.state.shape[0]:
            drive = np.resize(drive, self.state.shape)
        self.state = np.tanh(self.w @ self.state + drive)
        return self.state.copy()


__all__ = ["ApproximateConfig", "ApproximateReservoir"]
