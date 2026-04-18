"""Tests for ONNX export — single-head and cascade paths."""

import os

import onnx
import pytest
import torch

from export.to_onnx import CASCADE_OUTPUT_NAMES, export_to_onnx


def _small_cascade():
    from models.dignity import Dignity
    return Dignity(task="cascade", input_size=16, hidden_size=32, n_layers=1)


def _small_risk():
    from models.dignity import Dignity
    return Dignity(task="risk", input_size=9, hidden_size=32, n_layers=1)


class TestCascadeOutputNames:
    def test_count_is_seven(self):
        assert len(CASCADE_OUTPUT_NAMES) == 7

    def test_contains_all_expected_names(self):
        expected = {
            "regime_probs", "var_estimate", "position_limit",
            "alpha_score", "action_logits", "value", "attention_weights",
        }
        assert expected == set(CASCADE_OUTPUT_NAMES)


class TestCascadeOnnxExport:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "cascade.onnx")
        export_to_onnx(_small_cascade(), out, input_shape=(1, 20, 16), verify=False)
        assert os.path.exists(out)

    def test_onnx_model_is_valid(self, tmp_path):
        out = str(tmp_path / "cascade.onnx")
        export_to_onnx(_small_cascade(), out, input_shape=(1, 20, 16), verify=False)
        onnx.checker.check_model(onnx.load(out))

    def test_output_names_in_exported_model(self, tmp_path):
        out = str(tmp_path / "cascade.onnx")
        export_to_onnx(_small_cascade(), out, input_shape=(1, 20, 16), verify=False)
        graph_outputs = {o.name for o in onnx.load(out).graph.output}
        for name in CASCADE_OUTPUT_NAMES:
            assert name in graph_outputs, f"Missing output: {name}"


class TestSingleHeadOnnxExport:
    """Existing risk-head export path must be unaffected."""

    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "risk.onnx")
        export_to_onnx(_small_risk(), out, input_shape=(1, 20, 9), verify=False)
        assert os.path.exists(out)

    def test_onnx_model_is_valid(self, tmp_path):
        out = str(tmp_path / "risk.onnx")
        export_to_onnx(_small_risk(), out, input_shape=(1, 20, 9), verify=False)
        onnx.checker.check_model(onnx.load(out))

    def test_output_names_are_predictions_and_attention(self, tmp_path):
        out = str(tmp_path / "risk.onnx")
        export_to_onnx(_small_risk(), out, input_shape=(1, 20, 9), verify=False)
        graph_outputs = {o.name for o in onnx.load(out).graph.output}
        assert "predictions" in graph_outputs
        assert "attention_weights" in graph_outputs
