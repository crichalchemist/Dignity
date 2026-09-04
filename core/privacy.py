"""Privacy-preserving utilities for transaction data.

Two families of primitives, each backed by a test in tests/test_privacy.py:

- Keyed pseudonymization (HMAC-SHA256): same key -> linkable, different key ->
  unlinkable.
- Input-level differential privacy: bounded Laplace noise spent against an
  epsilon ledger, and k-anonymous generalization by quantile binning.

Nothing here claims more than its test proves. docs/PRIVACY.md and
docs/THREAT-MODEL.md carry the exact wording of each guarantee.
"""

import hashlib
import hmac
import math
import secrets

import numpy as np

_MIN_KEY_BYTES = 16


class BudgetExhausted(ValueError):  # noqa: N818
    """Raised when a spend would exceed the configured epsilon total."""


class PrivacyBudget:
    """Sequential-composition ledger for epsilon-differential privacy.

    Epsilons add across releases. A spend that would push the total past
    ``epsilon_total`` raises instead of degrading the guarantee silently.
    """

    def __init__(self, epsilon_total: float):
        if epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")
        self.epsilon_total = float(epsilon_total)
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.epsilon_total - self._spent

    def spend(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self._spent + epsilon > self.epsilon_total + 1e-12:
            raise BudgetExhausted(
                f"spending {epsilon} would exceed budget: "
                f"{self._spent:.6f} spent of {self.epsilon_total:.6f}"
            )
        self._spent += epsilon


class PrivacyManager:
    """Privacy primitives bound to one budget, one optional key, and one RNG.

    ``rng`` is any object exposing ``random() -> float`` in [0, 1). Production
    uses ``secrets.SystemRandom``; tests inject ``random.Random(seed)`` or a
    constant for determinism without weakening the default.
    """

    def __init__(
        self,
        budget: PrivacyBudget,
        key: bytes | str | None = None,
        rng=None,
    ):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if key is not None and len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"key must be at least {_MIN_KEY_BYTES} bytes")
        self.budget = budget
        self._key = key
        self._rng = rng if rng is not None else secrets.SystemRandom()

    # -- keyed pseudonymization ---------------------------------------------

    def pseudonymize(self, identifier: str) -> str:
        """HMAC-SHA256 of ``identifier`` under the manager's key, as hex.

        Same key -> same pseudonym (linkable). Different key -> unrelated
        pseudonym (unlinkable). Requires a key; never falls back to unkeyed.
        """
        if self._key is None:
            raise ValueError("pseudonymize requires a key")
        return hmac.new(
            self._key, identifier.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def pseudonymize_many(self, identifiers: list[str]) -> list[str]:
        return [self.pseudonymize(i) for i in identifiers]

    # -- differential privacy -------------------------------------------------

    def add_laplace_noise(
        self,
        values: np.ndarray,
        *,
        epsilon: float,
        bounds: tuple[float, float],
    ) -> np.ndarray:
        """Clip ``values`` to ``bounds`` and add Laplace noise for epsilon-DP.

        Sensitivity is the width of ``bounds``; the caller cannot understate it
        by omission. ``epsilon`` is spent from the budget before any sampling,
        so a refused spend releases nothing. Floating-point implementation:
        see docs/THREAT-MODEL.md for the Mironov (2012) caveat.

        Returns a new array; ``values`` is not modified.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        lo, hi = bounds
        if lo >= hi:
            raise ValueError("bounds must satisfy lo < hi")
        self.budget.spend(epsilon)
        scale = (hi - lo) / epsilon
        clipped = np.clip(np.asarray(values, dtype=float), lo, hi)
        noise = np.fromiter(
            (self._laplace(scale) for _ in range(clipped.size)),
            dtype=float,
            count=clipped.size,
        ).reshape(clipped.shape)
        return clipped + noise

    def _laplace(self, scale: float) -> float:
        """One Laplace(0, scale) draw by inverse CDF from a uniform in [0, 1)."""
        while True:
            u = self._rng.random() - 0.5
            if abs(u) < 0.5:
                break
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    # -- k-anonymity ----------------------------------------------------------

    @staticmethod
    def generalize_amounts(
        values: np.ndarray,
        *,
        bins: int = 10,
        k: int = 5,
    ) -> np.ndarray:
        """Generalize ``values`` so every output value is shared by >= ``k`` records.

        Quantile (equal-frequency) edges give balanced bins; any bin with fewer
        than ``k`` members is merged into its smaller neighbour until none
        remain. Each record is replaced by the midpoint of its class's min and
        max, a function of the class alone. This is k-anonymity for this column
        as released; features derived from it downstream carry no k claim.
        """
        if k < 2:
            raise ValueError("k must be at least 2")
        if bins < 2:
            raise ValueError("bins must be at least 2")
        x = np.asarray(values, dtype=float).ravel()
        n = x.size
        if n < k:
            raise ValueError(f"need at least k={k} records, got {n}")

        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 2:
            # every value identical: one class of size n >= k
            return np.full(np.shape(values), x[0])

        # bin i covers [edges[i], edges[i+1]); the last bin is closed on the right
        idx = np.searchsorted(edges, x, side="right") - 1
        idx = np.clip(idx, 0, edges.size - 2)

        while True:
            counts = np.bincount(idx, minlength=edges.size - 1)
            small = np.flatnonzero((counts > 0) & (counts < k))
            if small.size == 0:
                break
            i = small[0]
            nonempty = np.flatnonzero(counts > 0)
            lower = nonempty[nonempty < i]
            upper = nonempty[nonempty > i]
            candidates = []
            if lower.size:
                candidates.append(lower[-1])
            if upper.size:
                candidates.append(upper[0])
            target = min(candidates, key=lambda b: (counts[b], b))
            idx[idx == i] = target

        out = np.empty(n)
        for b in np.unique(idx):
            members = idx == b
            out[members] = (x[members].min() + x[members].max()) / 2.0
        return out.reshape(np.shape(values))
