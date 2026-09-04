"""Subject-layer privacy primitives. Every test here backs one sentence in docs/PRIVACY.md."""

import random
import secrets

import numpy as np
import pytest

from core.privacy import BudgetExhausted, PrivacyBudget, PrivacyManager


class TestPrivacyBudget:
    """Sequential composition: epsilons add, and overspending refuses."""

    def test_refuses_to_overspend(self):
        budget = PrivacyBudget(epsilon_total=1.0)
        budget.spend(0.6)
        with pytest.raises(BudgetExhausted):
            budget.spend(0.5)
        # a refused spend must not be recorded
        assert budget.spent == pytest.approx(0.6)

    def test_allows_spending_exactly_to_total(self):
        budget = PrivacyBudget(epsilon_total=1.0)
        budget.spend(0.5)
        budget.spend(0.5)
        assert budget.remaining == pytest.approx(0.0)

    def test_composes_sequentially(self):
        budget = PrivacyBudget(epsilon_total=2.0)
        for eps in (0.25, 0.5, 0.75):
            budget.spend(eps)
        assert budget.spent == pytest.approx(1.5)
        assert budget.remaining == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_total_and_spend(self, bad):
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon_total=bad)
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon_total=1.0).spend(bad)


def _manager(key=b"0123456789abcdef", total=1.0, rng=None):
    return PrivacyManager(PrivacyBudget(epsilon_total=total), key=key, rng=rng)


class TestPseudonymize:
    """Keyed pseudonymization: linkable under one key, unlinkable across keys."""

    def test_same_key_is_deterministic(self):
        pm = _manager()
        assert pm.pseudonymize("0xabc") == pm.pseudonymize("0xabc")
        assert len(pm.pseudonymize("0xabc")) == 64

    def test_different_keys_are_unlinkable(self):
        a = _manager(key=b"key-one-is-16-bytes!")
        b = _manager(key=b"key-two-is-16-bytes!")
        assert a.pseudonymize("0xabc") != b.pseudonymize("0xabc")

    def test_rejects_short_key(self):
        with pytest.raises(ValueError, match="16 bytes"):
            _manager(key=b"short")

    def test_requires_key_at_call_time(self):
        pm = _manager(key=None)  # construction is allowed without a key
        with pytest.raises(ValueError, match="key"):
            pm.pseudonymize("0xabc")

    def test_distinct_key_id_pairs_never_collide_on_concatenation(self):
        # Under salt + identifier concatenation these two pairs hash identically.
        # HMAC keeps key and message in separate domains.
        a = PrivacyManager(PrivacyBudget(1.0), key=b"aaaaaaaaaaaaaaaab")
        b = PrivacyManager(PrivacyBudget(1.0), key=b"aaaaaaaaaaaaaaaa")
        assert a.pseudonymize("cd") != b.pseudonymize("bcd")

    def test_pseudonymize_many_preserves_order_and_length(self):
        pm = _manager()
        ids = ["x", "y", "x"]
        out = pm.pseudonymize_many(ids)
        assert len(out) == 3
        assert out[0] == out[2] != out[1]

    def test_str_key_is_utf8_encoded(self):
        assert _manager(key="0123456789abcdef").pseudonymize("q") == _manager(
            key=b"0123456789abcdef"
        ).pseudonymize("q")


class _ZeroNoise:
    """rng whose uniform is always 0.5 -> Laplace inverse CDF at u=0 -> exactly 0 noise."""

    def random(self) -> float:
        return 0.5


class TestAddLaplaceNoise:
    """Input-level epsilon-DP: bounded, ledgered, non-recoverable noise."""

    def test_clips_to_bounds_before_noising(self):
        pm = _manager(rng=_ZeroNoise())
        values = np.array([-50.0, 0.0, 500.0, 1500.0])
        out = pm.add_laplace_noise(values, epsilon=1.0, bounds=(0.0, 1000.0))
        np.testing.assert_array_equal(out, [0.0, 0.0, 500.0, 1000.0])
        # input untouched
        np.testing.assert_array_equal(values, [-50.0, 0.0, 500.0, 1500.0])

    def test_noise_scale_is_sensitivity_over_epsilon(self):
        # Laplace(0, b) has mean |x| = b. With bounds width 100 and eps 0.5, b = 200.
        pm = _manager(total=10.0, rng=random.Random(42))
        out = pm.add_laplace_noise(np.zeros(20_000), epsilon=0.5, bounds=(0.0, 100.0))
        assert np.mean(np.abs(out)) == pytest.approx(200.0, rel=0.05)
        # symmetric around zero
        assert np.mean(out) == pytest.approx(0.0, abs=10.0)

    def test_spends_budget(self):
        pm = _manager(total=1.0, rng=_ZeroNoise())
        pm.add_laplace_noise(np.ones(3), epsilon=0.4, bounds=(0.0, 1.0))
        assert pm.budget.spent == pytest.approx(0.4)
        pm.add_laplace_noise(np.ones(3), epsilon=0.6, bounds=(0.0, 1.0))
        with pytest.raises(BudgetExhausted):
            pm.add_laplace_noise(np.ones(3), epsilon=0.1, bounds=(0.0, 1.0))

    def test_budget_is_spent_before_any_release(self):
        # If the spend fails, nothing may be sampled: the rng must never be consulted.
        class _Counting:
            def __init__(self):
                self.calls = 0

            def random(self) -> float:
                self.calls += 1
                return 0.5

        rng = _Counting()
        pm = _manager(total=0.5, rng=rng)
        with pytest.raises(BudgetExhausted):
            pm.add_laplace_noise(np.ones(3), epsilon=1.0, bounds=(0.0, 1.0))
        assert pm.budget.spent == 0.0
        assert rng.calls == 0

    @pytest.mark.parametrize("eps", [0.0, -0.1])
    def test_rejects_nonpositive_epsilon(self, eps):
        with pytest.raises(ValueError):
            _manager().add_laplace_noise(np.ones(2), epsilon=eps, bounds=(0.0, 1.0))

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lo < hi"):
            _manager().add_laplace_noise(np.ones(2), epsilon=1.0, bounds=(1.0, 1.0))

    def test_default_rng_is_system_random(self):
        pm = PrivacyManager(PrivacyBudget(1.0))
        assert isinstance(pm._rng, secrets.SystemRandom)

    def test_preserves_shape(self):
        pm = _manager(rng=_ZeroNoise())
        out = pm.add_laplace_noise(np.zeros((3, 4)), epsilon=1.0, bounds=(0.0, 1.0))
        assert out.shape == (3, 4)

    def test_uniform_at_exactly_zero_is_resampled(self):
        # rng.random() == 0.0 maps to u = -0.5 -> log(0). Must resample, not return -inf.
        class _ZeroThenHalf:
            def __init__(self):
                self.calls = 0

            def random(self):
                self.calls += 1
                return 0.0 if self.calls == 1 else 0.5

        pm = _manager(rng=_ZeroThenHalf())
        out = pm.add_laplace_noise(np.zeros(1), epsilon=1.0, bounds=(0.0, 1.0))
        assert np.isfinite(out).all()
