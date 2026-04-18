"""Test model components."""

import pytest
import torch

from models.backbone.attention import AdditiveAttention
from models.backbone.cnn1d import CNN1D
from models.backbone.hybrid import DignityBackbone
from models.backbone.lstm import StackedLSTM
from models.dignity import Dignity
from models.head.alpha import AlphaHead
from models.head.forecast import ForecastHead
from models.head.policy import PolicyHead
from models.head.regime import RegimeHead
from models.head.risk import RiskHead


class TestBackboneComponents:
    """Test individual backbone components."""

    def test_cnn1d(self):
        """Test 1D-CNN component."""
        model = CNN1D(input_size=32, hidden_size=64, kernel_size=3)
        x = torch.randn(4, 100, 32)  # [B, T, F]

        out = model(x)

        assert out.shape == (4, 100, 64)  # [B, T, H]

    def test_stacked_lstm(self):
        """Test LSTM component."""
        model = StackedLSTM(input_size=64, hidden_size=128, num_layers=2)
        x = torch.randn(4, 100, 64)

        out, (h_n, c_n) = model(x)

        assert out.shape == (4, 100, 128)  # [B, T, H]
        assert h_n.shape == (2, 4, 128)  # [L, B, H]
        assert c_n.shape == (2, 4, 128)

    def test_additive_attention(self):
        """Test attention mechanism."""
        model = AdditiveAttention(hidden_size=128)
        x = torch.randn(4, 100, 128)

        context, weights = model(x)

        assert context.shape == (4, 128)  # [B, H]
        assert weights.shape == (4, 100)  # [B, T]

        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_dignity_backbone(self):
        """Test full backbone."""
        model = DignityBackbone(input_size=32, hidden_size=256, n_layers=2)
        x = torch.randn(4, 100, 32)

        context, attn_weights = model(x)

        assert context.shape == (4, 256)
        assert attn_weights.shape == (4, 100)


class TestTaskHeads:
    """Test task-specific heads."""

    def test_risk_head(self):
        """Test risk scoring head — returns (var_estimate, position_limit) tuple."""
        head = RiskHead(input_size=256, hidden_size=128)
        context = torch.randn(4, 256)

        var_est, pos_lim = head(context)

        assert var_est.shape == (4, 1)
        assert pos_lim.shape == (4, 1)
        assert torch.all(var_est >= 0) and torch.all(var_est <= 1)
        assert torch.all(pos_lim >= 0) and torch.all(pos_lim <= 1)

    def test_forecast_head(self):
        """Test forecasting head."""
        head = ForecastHead(input_size=256, pred_len=5, num_features=3)
        context = torch.randn(4, 256)

        predictions = head(context)

        assert predictions.shape == (4, 5, 3)  # [B, pred_len, features]

    def test_policy_head(self):
        """Test policy head."""
        head = PolicyHead(input_size=256, n_actions=3)
        context = torch.randn(4, 256)

        logits, value = head(context)

        assert logits.shape == (4, 3)  # [B, actions]
        assert value.shape == (4, 1)  # [B, 1]

        # Test action sampling
        action, log_prob, value = head.sample_action(context)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)


class TestDignityModel:
    """Test complete Dignity model."""

    def test_risk_model(self):
        """Test risk scoring model — head returns (var_estimate, position_limit)."""
        model = Dignity(task="risk", input_size=32, hidden_size=128, n_layers=2)
        x = torch.randn(4, 100, 32)

        (var_est, pos_lim), attn = model(x)

        assert var_est.shape == (4, 1)
        assert pos_lim.shape == (4, 1)
        assert attn.shape == (4, 100)

    def test_forecast_model(self):
        """Test forecasting model."""
        model = Dignity(
            task="forecast",
            input_size=32,
            hidden_size=128,
            n_layers=2,
            pred_len=5,
            num_features=3,
        )
        x = torch.randn(4, 100, 32)

        predictions, attn = model(x)

        assert predictions.shape == (4, 5, 3)

    def test_model_summary(self):
        """Test model summary."""
        model = Dignity(task="risk", input_size=32, hidden_size=128)
        summary = model.summary()

        assert "Dignity" in summary
        assert "128" in summary  # hidden size
        # Parameter count is formatted with commas, so just check it exists
        assert "Total parameters" in summary

    def test_predict(self):
        """Test inference mode — risk predict returns (var_estimate, position_limit)."""
        model = Dignity(task="risk", input_size=32, hidden_size=64)
        x = torch.randn(2, 50, 32)

        var_est, pos_lim = model.predict(x)

        assert var_est.shape == (2, 1)
        assert pos_lim.shape == (2, 1)
        assert not model.training  # Should be in eval mode


# ---------------------------------------------------------------------------
# Phase 4: Cascade model — forward_cascade() + cascade_loss()
# ---------------------------------------------------------------------------


class TestCascadeForward:
    """Dignity(task='cascade') forward pass through all 4 heads."""

    def _make_model(self):
        return Dignity(task="cascade", input_size=32, hidden_size=256, n_layers=2)

    def test_instantiates_without_error(self):
        self._make_model()

    def test_forward_returns_dict(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert isinstance(out, dict)

    def test_all_keys_present(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        expected = {
            "regime_probs",
            "var_estimate",
            "position_limit",
            "alpha_score",
            "action_logits",
            "value",
            "attention_weights",
        }
        assert expected == set(out.keys())

    def test_regime_shape(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["regime_probs"].shape == (4, 4)

    def test_regime_sums_to_one(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert torch.allclose(out["regime_probs"].sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_var_estimate_shape_and_range(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["var_estimate"].shape == (4, 1)
        assert (out["var_estimate"] >= 0).all() and (out["var_estimate"] <= 1).all()

    def test_position_limit_shape_and_range(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["position_limit"].shape == (4, 1)
        assert (out["position_limit"] >= 0).all() and (out["position_limit"] <= 1).all()

    def test_alpha_shape_and_range(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["alpha_score"].shape == (4, 1)
        assert (out["alpha_score"] >= -1).all() and (out["alpha_score"] <= 1).all()

    def test_action_logits_shape(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["action_logits"].shape == (4, 3)

    def test_value_shape(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["value"].shape == (4, 1)

    def test_attention_shape(self, sample_sequence_32):
        model = self._make_model()
        out = model(sample_sequence_32)
        assert out["attention_weights"].shape == (4, 100)

    def test_single_head_forward_unaffected(self, sample_sequence):
        """The non-cascade forward() path must still work after refactor."""
        model = Dignity(
            task="forecast", input_size=9, hidden_size=128, n_layers=2, pred_len=5, num_features=3
        )
        preds, attn = model(sample_sequence)
        assert preds.shape == (4, 5, 3)


class TestCascadeLoss:
    """cascade_loss() — Guided Learning weighted sum."""

    def _make_model_and_data(self, B: int = 4):
        model = Dignity(task="cascade", input_size=32, hidden_size=256)
        x = torch.randn(B, 100, 32)
        labels = {
            "regime": torch.randint(0, 4, (B,)),
            "var": torch.rand(B, 1),
            "alpha": torch.randn(B, 1).clamp(-1, 1),
            "action": torch.randint(0, 3, (B,)),
        }
        return model, x, labels

    def test_returns_total_and_per_head_dict(self):
        model, x, labels = self._make_model_and_data()
        outputs = model(x)
        weights = {"regime": 0.2, "risk": 0.3, "alpha": 0.3, "policy": 0.2}
        total, per_head = Dignity.cascade_loss(outputs, labels, weights)
        assert isinstance(total, torch.Tensor) and total.ndim == 0
        assert set(per_head.keys()) == {"regime", "risk", "alpha", "policy"}

    def test_total_is_weighted_sum(self):
        model, x, labels = self._make_model_and_data()
        outputs = model(x)
        weights = {"regime": 0.2, "risk": 0.3, "alpha": 0.3, "policy": 0.2}
        total, per_head = Dignity.cascade_loss(outputs, labels, weights)
        expected = sum(weights[k] * per_head[k] for k in weights)
        assert torch.isclose(total, expected, atol=1e-5)

    def test_total_loss_positive(self):
        model, x, labels = self._make_model_and_data()
        outputs = model(x)
        weights = {"regime": 0.2, "risk": 0.3, "alpha": 0.3, "policy": 0.2}
        total, _ = Dignity.cascade_loss(outputs, labels, weights)
        assert total.item() > 0

    def test_gradients_reach_backbone(self):
        """Guided Learning guarantee: loss must propagate gradients back through backbone."""
        model, x, labels = self._make_model_and_data()
        outputs = model(x)
        weights = {"regime": 0.2, "risk": 0.3, "alpha": 0.3, "policy": 0.2}
        total, _ = Dignity.cascade_loss(outputs, labels, weights)
        total.backward()
        # Check that backbone CNN first-layer weight has a gradient
        first_param = next(model.backbone.parameters())
        assert first_param.grad is not None and first_param.grad.abs().sum() > 0

    def test_equal_weights_means_equal_influence(self):
        """With equal weights each head contributes equally to total."""
        model, x, labels = self._make_model_and_data()
        outputs = model(x)
        weights = {"regime": 0.25, "risk": 0.25, "alpha": 0.25, "policy": 0.25}
        total, per_head = Dignity.cascade_loss(outputs, labels, weights)
        reconstructed = 0.25 * sum(per_head.values())
        assert torch.isclose(total, reconstructed, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase 3: New and modified heads
# ---------------------------------------------------------------------------


class TestRegimeHead:
    """RegimeHead: context [B, 256] → regime_probs [B, 4] (softmax)."""

    def test_output_shape(self):
        head = RegimeHead(input_size=256)
        out = head(torch.randn(4, 256))
        assert out.shape == (4, 4)

    def test_softmax_sums_to_one(self):
        head = RegimeHead(input_size=256)
        out = head(torch.randn(8, 256))
        assert torch.allclose(out.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_all_probs_non_negative(self):
        head = RegimeHead(input_size=256)
        out = head(torch.randn(4, 256))
        assert (out >= 0).all()

    def test_configurable_n_regimes(self):
        head = RegimeHead(input_size=256, n_regimes=6)
        out = head(torch.randn(4, 256))
        assert out.shape == (4, 6)

    def test_output_dtype_float32(self):
        head = RegimeHead(input_size=256)
        out = head(torch.randn(4, 256))
        assert out.dtype == torch.float32


class TestAlphaHead:
    """AlphaHead: [B, 260] → alpha_score [B, 1] (tanh, range [-1, 1])."""

    def test_output_shape(self):
        head = AlphaHead(input_size=260)
        out = head(torch.randn(4, 260))
        assert out.shape == (4, 1)

    def test_output_in_tanh_range(self):
        head = AlphaHead(input_size=260)
        out = head(torch.randn(32, 260))
        assert (out >= -1).all() and (out <= 1).all()

    def test_output_dtype_float32(self):
        head = AlphaHead(input_size=260)
        out = head(torch.randn(4, 260))
        assert out.dtype == torch.float32

    def test_gradients_flow(self):
        head = AlphaHead(input_size=260)
        x = torch.randn(4, 260, requires_grad=True)
        loss = head(x).sum()
        loss.backward()
        assert x.grad is not None


class TestRiskHeadCascade:
    """RiskHead (cascade variant): input [B, 260] → (var_estimate [B,1], position_limit [B,1])."""

    def test_returns_tuple(self):
        head = RiskHead(input_size=260)
        out = head(torch.randn(4, 260))
        assert isinstance(out, tuple) and len(out) == 2

    def test_var_estimate_shape(self):
        head = RiskHead(input_size=260)
        var_est, _ = head(torch.randn(4, 260))
        assert var_est.shape == (4, 1)

    def test_position_limit_shape(self):
        head = RiskHead(input_size=260)
        _, pos_lim = head(torch.randn(4, 260))
        assert pos_lim.shape == (4, 1)

    def test_both_outputs_in_zero_one(self):
        head = RiskHead(input_size=260)
        var_est, pos_lim = head(torch.randn(16, 260))
        assert (var_est >= 0).all() and (var_est <= 1).all()
        assert (pos_lim >= 0).all() and (pos_lim <= 1).all()

    def test_legacy_256_input_still_works(self):
        """Backward compat: single-head path passes input_size=256."""
        head = RiskHead(input_size=256)
        out = head(torch.randn(4, 256))
        assert isinstance(out, tuple)

    def test_gradients_flow_to_both_outputs(self):
        head = RiskHead(input_size=260)
        x = torch.randn(4, 260, requires_grad=True)
        var_est, pos_lim = head(x)
        (var_est + pos_lim).sum().backward()
        assert x.grad is not None


class TestPolicyHeadCascade:
    """PolicyHead with cascade input_size=258 (context 256 + alpha 1 + var 1)."""

    def test_forward_with_cascade_input_size(self):
        head = PolicyHead(input_size=258, n_actions=3)
        logits, value = head(torch.randn(4, 258))
        assert logits.shape == (4, 3)
        assert value.shape == (4, 1)

    def test_sample_action_still_works(self):
        head = PolicyHead(input_size=258, n_actions=3)
        action, log_prob, value = head.sample_action(torch.randn(4, 258))
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert value.shape == (4, 1)

    def test_action_indices_valid(self):
        head = PolicyHead(input_size=258, n_actions=3)
        action, _, _ = head.sample_action(torch.randn(8, 258))
        assert ((action >= 0) & (action < 3)).all()
