"""Subject-layer privacy primitives. Every test here backs one sentence in docs/PRIVACY.md."""

import pytest

from core.privacy import BudgetExhausted, PrivacyBudget


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
