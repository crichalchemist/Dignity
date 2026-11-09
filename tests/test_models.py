"""Test model components."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.config import PolicyConfig
from models.backbone.attention import AdditiveAttention
from models.backbone.cnn1d import CNN1D
from models.backbone.hybrid import DignityBackbone
from models.backbone.lstm import StackedLSTM
from models.dignity import Dignity
from models.head.forecast import ForecastHead
from models.head.policy import ACTIONS, PolicyHead, cost_matrix
from models.head.risk import RiskHead


class TestBackboneComponents:
    """Test individual backbone components."""

    def test_cnn1d(self):
        """Test 1D-CNN component."""
        model = CNN1D(input_size=9, hidden_size=64, kernel_size=3)
        x = torch.randn(4, 100, 9)  # [B, T, F]

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
        model = DignityBackbone(input_size=9, hidden_size=256, n_layers=2)
        x = torch.randn(4, 100, 9)

        context, attn_weights = model(x)

        assert context.shape == (4, 256)
        assert attn_weights.shape == (4, 100)


class TestTaskHeads:
    """Test task-specific heads."""

    def test_cost_matrix_rows_follow_actions_and_columns_index_the_true_state(self):
        C = cost_matrix(PolicyConfig().costs)
        assert C.dtype == np.float32
        assert C.tolist() == [[0.0, 10.0], [1.0, 1.0], [3.0, 0.5]]
        assert set(PolicyConfig().costs) == set(ACTIONS)

    def test_risk_head(self):
        """Test risk scoring head."""
        head = RiskHead(input_size=256, hidden_size=128)
        context = torch.randn(4, 256)

        risk = head(context)

        assert risk.shape == (4, 1)
        assert torch.all(risk >= 0) and torch.all(risk <= 1)  # Sigmoid output

    def test_forecast_probabilities_never_decrease_with_horizon(self):
        # y_h = 1 if an onset lies in (t, t+h]; that event only grows with h,
        # so a coherent early-warning score must too.
        head = ForecastHead(input_size=256, horizon=7)
        p = head(torch.randn(4, 256))
        assert p.shape == (4, 7)
        assert torch.all((p >= 0) & (p <= 1))
        assert torch.all(p[:, 1:] >= p[:, :-1])

    def test_policy_head_scores_allow_review_block_and_nothing_else(self):
        head = PolicyHead(input_size=256, n_actions=3)
        logits = head(torch.randn(4, 256))
        assert isinstance(logits, torch.Tensor) and logits.shape == (4, 3)
        assert not hasattr(head, "value") and not hasattr(head, "sample_action")
        assert ACTIONS == ("allow", "review", "block")

    def test_policy_head_rejects_an_action_count_the_cost_table_cannot_price(self):
        with pytest.raises(ValueError, match="ACTIONS"):
            PolicyHead(input_size=256, n_actions=4)


class TestDignityModel:
    """Test complete Dignity model."""

    def test_risk_model(self):
        """Test risk scoring model."""
        model = Dignity(task="risk", input_size=9, hidden_size=128, n_layers=2)
        x = torch.randn(4, 100, 9)

        predictions, attn = model(x)

        assert predictions.shape == (4, 1)
        assert attn.shape == (4, 100)

    def test_policy_model_returns_logits_and_attention(self):
        out, attn = Dignity(task="policy", input_size=9, hidden_size=64, n_actions=3)(
            torch.randn(2, 50, 9)
        )
        assert out.shape == (2, 3) and attn.shape == (2, 50)

    def test_forecast_model_takes_horizon_not_volume_price_shape(self):
        model = Dignity(
            task="forecast",
            input_size=9,
            hidden_size=128,
            n_layers=2,
            horizon=7,
        )
        predictions, attn = model(torch.randn(4, 100, 9))
        assert predictions.shape == (4, 7) and attn.shape == (4, 100)
        with pytest.raises(TypeError):
            Dignity(task="forecast", input_size=9, pred_len=5, num_features=3)

    def test_model_summary(self):
        """Test model summary."""
        model = Dignity(task="risk", input_size=9, hidden_size=128)
        summary = model.summary()

        assert "Dignity" in summary
        assert "128" in summary  # hidden size
        # Parameter count is formatted with commas, so just check it exists
        assert "Total parameters" in summary

    def test_predict(self):
        """Test inference mode."""
        model = Dignity(task="risk", input_size=9, hidden_size=64)
        x = torch.randn(2, 50, 9)

        predictions = model.predict(x)

        assert predictions.shape == (2, 1)
        assert not model.training  # Should be in eval mode

    def test_cnn_kernel_size_reaches_the_backbone(self):
        model = Dignity(
            task="risk", input_size=7, hidden_size=32, n_layers=1, cnn_kernel_size=5
        )
        kernels = [
            m.kernel_size
            for m in model.backbone.cnn.modules()
            if isinstance(m, nn.Conv1d)
        ]
        assert kernels == [(5,), (5,)]

    def test_misspelled_head_kwarg_is_rejected(self):
        with pytest.raises(TypeError, match="bogus_kwarg"):
            Dignity(task="risk", input_size=9, bogus_kwarg=1)

    def test_head_kwarg_for_another_task_is_rejected(self):
        with pytest.raises(TypeError, match="horizon"):
            Dignity(task="risk", input_size=9, horizon=5)
