"""Tests for models/agent_hybrid.py - model architecture and forward pass."""
import pytest
import torch
import torch.nn as nn


@pytest.mark.unit
class TestSharedEncoder:
    """Unit tests for SharedEncoder base class."""

    def test_shared_encoder_initialization(self, device):
        """Test SharedEncoder instantiation with various configs."""
        from models.agent_hybrid import SharedEncoder

        encoder = SharedEncoder(
            num_features=20,
            hidden_size_lstm=64,
            cnn_num_filters=32,
            attention_dim=64,
        ).to(device)

        assert encoder is not None
        assert hasattr(encoder, 'cnn')
        assert hasattr(encoder, 'lstm')

    def test_shared_encoder_forward_pass(self, sample_batch, device):
        """Test forward pass through SharedEncoder."""
        from models.agent_hybrid import SharedEncoder

        x, _ = sample_batch
        x = x.to(device)

        encoder = SharedEncoder(
            num_features=20,
            hidden_size_lstm=64,
            cnn_num_filters=32,
            attention_dim=64,
        ).to(device)

        encoder.eval()
        with torch.no_grad():
            output = encoder(x)

        # Should return embedding tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 4  # batch size

    def test_shared_encoder_output_shape(self, sample_batch, device):
        """Test that SharedEncoder output has expected shape."""
        from models.agent_hybrid import SharedEncoder

        x, _ = sample_batch
        x = x.to(device)

        hidden_size = 128
        encoder = SharedEncoder(
            num_features=20,
            hidden_size_lstm=hidden_size,
            cnn_num_filters=32,
        ).to(device)

        encoder.eval()
        with torch.no_grad():
            output = encoder(x)

        # Output should be (batch_size, embedding_dim)
        assert output.shape[0] == 4
        # Validate embedding dimension (should match hidden_size_lstm + cnn_num_filters)
        expected_embedding_dim = hidden_size + 32
        assert output.shape[1] == expected_embedding_dim


@pytest.mark.unit
class TestDignityModel:
    """Unit tests for main DignityModel (agent_hybrid.py)."""

    def test_dignity_model_initialization(self, mock_model_config, device):
        """Test DignityModel instantiation."""
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_dignity_model_forward_classification(self, sample_batch, mock_model_config, device):
        """Test forward pass in classification mode."""
        from models.agent_hybrid import DignityModel

        x, _ = sample_batch
        x = x.to(device)

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x)

        assert isinstance(outputs, dict)
        # Handle both possible output key names
        output_key = "direction_logits" if "direction_logits" in outputs else "primary"
        assert output_key in outputs, f"Expected either 'direction_logits' or 'primary' in outputs, got {outputs.keys()}"
        assert outputs[output_key].shape[0] == 4  # batch size

    def test_dignity_model_parameter_count(self, mock_model_config):
        """Test that model has reasonable number of parameters."""
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config)

        param_count = sum(p.numel() for p in model.parameters())

        # Should have at least some parameters
        assert param_count > 0
        # Should be reasonable (< 100M for this architecture)
        assert param_count < 100_000_000

    def test_dignity_model_outputs_in_correct_range(self, sample_batch, mock_model_config, device):
        """Test that model outputs are in expected ranges."""
        from models.agent_hybrid import DignityModel

        x, _ = sample_batch
        x = x.to(device)

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x)

        # Check for NaN/Inf
        for key, value in outputs.items():
            assert not torch.isnan(value).any(), f"{key} contains NaN"
            assert not torch.isinf(value).any(), f"{key} contains Inf"

    def test_dignity_model_gradient_flow(self, sample_batch, mock_model_config, device):
        """Test that gradients flow through entire model."""
        from models.agent_hybrid import DignityModel

        x, _ = sample_batch
        x = x.to(device)

        model = DignityModel(mock_model_config).to(device)
        model.train()

        outputs = model(x)
        loss = outputs["direction_logits"].sum()
        loss.backward()

        # Check that at least first layer has gradients
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in model parameters"

    def test_dignity_model_batch_sizes(self, mock_model_config, device):
        """Test model with various batch sizes."""
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 120, 20, device=device)

            with torch.no_grad():
                outputs = model(x)

            assert outputs["direction_logits"].shape[0] == batch_size

    def test_dignity_model_training_vs_eval_mode(self, sample_batch, mock_model_config, device):
        """Test difference between training and eval modes."""
        from models.agent_hybrid import DignityModel

        x, _ = sample_batch
        x = x.to(device)

        model = DignityModel(mock_model_config).to(device)

        # Training mode
        model.train()
        with torch.no_grad():
            outputs_train = model(x)

        # Eval mode
        model.eval()
        with torch.no_grad():
            outputs_eval = model(x)

        # Handle both possible output keys
        train_key = "direction_logits" if "direction_logits" in outputs_train else "primary"
        eval_key = "direction_logits" if "direction_logits" in outputs_eval else "primary"

        # Verify dropout causes different outputs (assert non-zero difference)
        diff = (outputs_train[train_key] - outputs_eval[eval_key]).abs().sum()
        assert diff > 0, "Expected dropout to cause different outputs in train vs eval mode"
        assert isinstance(outputs_train, dict)
        assert isinstance(outputs_eval, dict)


@pytest.mark.unit
class TestSignalPolicy:
    """Unit tests for signal_policy.py."""

    def test_signal_policy_initialization(self, device):
        """Test SignalModel initialization."""
        from models.signal_policy import SignalModel

        model = SignalModel(input_dim=20, hidden_dim=64, output_dim=1).to(device)

        assert model is not None
        assert isinstance(model, nn.Module)

    def test_signal_policy_forward(self, device):
        """Test forward pass through SignalModel."""
        from models.signal_policy import SignalModel

        x = torch.randn(4, 20, device=device)
        model = SignalModel(input_dim=20, hidden_dim=64, output_dim=1).to(device)
        model.eval()

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_execution_policy_initialization(self, device):
        """Test ExecutionPolicy initialization."""
        from models.signal_policy import ExecutionPolicy

        policy = ExecutionPolicy(state_dim=64, action_dim=1, hidden_dim=64).to(device)

        assert policy is not None


@pytest.mark.unit
class TestRegimeEncoder:
    """Unit tests for regime_encoder.py."""

    def test_regime_encoder_initialization(self, device):
        """Test RegimeEncoder initialization."""
        from models.regime_encoder import RegimeEncoder

        encoder = RegimeEncoder(input_dim=20, hidden_dim=64, num_regimes=3).to(device)

        assert encoder is not None

    def test_regime_encoder_forward(self, device):
        """Test forward pass through RegimeEncoder."""
        from models.regime_encoder import RegimeEncoder

        x = torch.randn(4, 120, 20, device=device)
        encoder = RegimeEncoder(input_dim=20, hidden_dim=64, num_regimes=3).to(device)
        encoder.eval()

        with torch.no_grad():
            output = encoder(x)

        # Should return regime encoding
        assert isinstance(output, torch.Tensor)


@pytest.mark.unit
class TestAgentMultitask:
    """Unit tests for agent_multitask.py."""

    def test_multitask_model_initialization(self, device):
        """Test multitask agent initialization."""
        from models.agent_multitask import AgentMultitask

        model = AgentMultitask(
            num_features=20,
            num_classes=3,
            num_auxiliary_tasks=3,
        ).to(device)

        assert model is not None

    def test_multitask_model_forward(self, sample_batch, mock_model_config, device):
        """Test forward pass through multitask model."""
        from models.agent_multitask import AgentMultitask

        x, _ = sample_batch
        x = x.to(device)

        model = AgentMultitask(
            num_features=20,
            num_classes=3,
            num_auxiliary_tasks=3,
        ).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x)

        # Should return dict with main task + auxiliary tasks
        assert isinstance(outputs, dict)
        assert "primary" in outputs or "direction_logits" in outputs

    def test_multitask_model_gradient_flow(self, sample_batch, device):
        """Test gradient flow through all task heads."""
        from models.agent_multitask import AgentMultitask

        x, _ = sample_batch
        x = x.to(device)

        model = AgentMultitask(
            num_features=20,
            num_classes=3,
            num_auxiliary_tasks=3,
        ).to(device)
        model.train()

        outputs = model(x)

        # Compute loss for each task and accumulate
        loss = 0
        for key, output in outputs.items():
            if isinstance(output, torch.Tensor):
                loss = loss + output.mean()

        loss.backward()

        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
