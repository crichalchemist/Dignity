"""Discrete selection utilities inspired by QUBO formulations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GreedyQUBOConfig:
    k: int
    risk_aversion: float = 0.5


class GreedyQUBOSelector:
    """Greedy selector that approximates a QUBO objective."""

    def __init__(self, cfg: GreedyQUBOConfig):
        self.cfg = cfg

    def solve(self, objective: np.ndarray, constraints: dict[str, object]) -> dict[str, object]:
        returns = np.asarray(objective, dtype=np.float64)
        cov = np.asarray(constraints.get("covariance")) if "covariance" in constraints else None
        k = int(constraints.get("k", self.cfg.k))
        selected: list[int] = []

        if cov is None:
            ranked = np.argsort(returns)[::-1][:k]
            return {"selected": ranked.tolist(), "score": float(returns[ranked].sum())}

        scores = returns.copy()
        for _ in range(k):
            best_idx = None
            best_score = -np.inf
            for i in range(len(returns)):
                if i in selected:
                    continue
                penalty = cov[i, i]
                if selected:
                    penalty += 2.0 * cov[i, selected].sum()
                candidate_score = scores[i] - self.cfg.risk_aversion * penalty
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
        total_score = float(returns[selected].sum()) if selected else 0.0
        return {"selected": selected, "score": total_score}


__all__ = ["GreedyQUBOConfig", "GreedyQUBOSelector"]
