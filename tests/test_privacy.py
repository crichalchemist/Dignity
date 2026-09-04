"""Subject-layer privacy primitives. Every test here backs one sentence in docs/PRIVACY.md."""

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
