"""N-choose-K constraint utility for discrete selection."""
from __future__ import annotations

import itertools

import numpy as np


class NChooseKOptimizer:
    """Brute-force N-choose-K for small universes, fallback to greedy."""

    def __init__(self, max_bruteforce: int = 15):
        self.max_bruteforce = max_bruteforce

    def solve(self, objective: np.ndarray, constraints: dict[str, object]) -> dict[str, object]:
        returns = np.asarray(objective, dtype=np.float64)
        k = int(constraints.get("k", min(5, len(returns))))
        cov = constraints.get("covariance")
        n = len(returns)
        if n <= self.max_bruteforce:
            best_score = -np.inf
            best_sel = None
            for combo in itertools.combinations(range(n), k):
                score = returns[list(combo)].sum()
                if cov is not None:
                    score -= 0.5 * np.sum(np.asarray(cov)[np.ix_(combo, combo)])
                if score > best_score:
                    best_score = score
                    best_sel = combo
            return {"selected": list(best_sel) if best_sel else [], "score": float(best_score)}
        # fallback greedy
        ranked = np.argsort(returns)[::-1][:k]
        return {"selected": ranked.tolist(), "score": float(returns[ranked].sum())}


__all__ = ["NChooseKOptimizer"]
