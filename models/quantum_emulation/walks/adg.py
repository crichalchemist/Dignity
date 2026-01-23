"""Quantum-walk inspired adaptive distribution generator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.quantum_emulation.interfaces import DistributionModel


@dataclass
class QWADGConfig:
    n_positions: int = 51
    steps: int = 20
    n_iters: int = 60
    candidate_scale: float = 0.2
    seed: int = 42


class SplitStepQuantumWalkADG(DistributionModel):
    """Split-step quantum walk distribution generator (emulated)."""

    def __init__(self, cfg: QWADGConfig | None = None):
        self.cfg = cfg or QWADGConfig()
        if self.cfg.n_positions % 2 == 0:
            raise ValueError("n_positions must be odd for symmetric walk")
        self._rng = np.random.default_rng(self.cfg.seed)
        self.theta1 = 0.1
        self.theta2 = 0.1
        self.target_probs: np.ndarray | None = None
        self.position_scale: float = 1.0

    def _coin(self, theta: float) -> np.ndarray:
        cos = np.cos(theta)
        sin = np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)

    def _simulate(self, theta1: float, theta2: float) -> np.ndarray:
        n = self.cfg.n_positions
        state = np.zeros((n, 2), dtype=np.complex128)
        center = n // 2
        state[center, 0] = 1.0 + 0.0j
        coin1 = self._coin(theta1)
        coin2 = self._coin(theta2)

        for _ in range(self.cfg.steps):
            state = state @ coin1.T
            state = self._shift(state)
            state = state @ coin2.T
            state = self._shift(state)
        probs = np.abs(state) ** 2
        probs = probs.sum(axis=1)
        total = probs.sum()
        return probs / total if total > 0 else probs

    def _shift(self, state: np.ndarray) -> np.ndarray:
        n = state.shape[0]
        shifted = np.zeros_like(state)
        # coin 0 moves left, coin 1 moves right
        shifted[1:, 0] = state[:-1, 0]
        shifted[:-1, 1] = state[1:, 1]
        # wrap edges
        shifted[0, 0] += state[0, 0]
        shifted[-1, 1] += state[-1, 1]
        return shifted

    def fit(self, data: np.ndarray) -> "SplitStepQuantumWalkADG":
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 1:
            data = data.reshape(-1)
        self.position_scale = np.std(data) if np.std(data) > 0 else 1.0
        bins = self.cfg.n_positions
        hist, _ = np.histogram(data, bins=bins, range=(-self._max_return(), self._max_return()), density=True)
        self.target_probs = hist / hist.sum()

        best_theta1, best_theta2 = self.theta1, self.theta2
        best_loss = np.inf
        for _ in range(self.cfg.n_iters):
            cand1 = best_theta1 + self._rng.normal(scale=self.cfg.candidate_scale)
            cand2 = best_theta2 + self._rng.normal(scale=self.cfg.candidate_scale)
            probs = self._simulate(cand1, cand2)
            loss = self._kl_divergence(self.target_probs, probs)
            if loss < best_loss:
                best_loss = loss
                best_theta1, best_theta2 = cand1, cand2
        self.theta1, self.theta2 = best_theta1, best_theta2
        return self

    def sample(self, context: dict | None, n_paths: int, horizon: int) -> np.ndarray:
        probs = self._simulate(self.theta1, self.theta2)
        positions = np.arange(self.cfg.n_positions) - (self.cfg.n_positions // 2)
        returns = positions * self._step_return()
        samples = self._rng.choice(returns, size=(n_paths, horizon), p=probs)
        return samples.astype(np.float64)

    def pdf(self, context: dict | None, grid: np.ndarray) -> np.ndarray:
        probs = self._simulate(self.theta1, self.theta2)
        positions = np.arange(self.cfg.n_positions) - (self.cfg.n_positions // 2)
        support = positions * self._step_return()
        return np.interp(grid, support, probs, left=0.0, right=0.0)

    def _step_return(self) -> float:
        return self._max_return() / (self.cfg.n_positions // 2)

    def _max_return(self) -> float:
        return 3.0 * self.position_scale

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-9
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))


__all__ = ["QWADGConfig", "SplitStepQuantumWalkADG"]
