"""ONNX export: verification result, exit codes, and shapes from the checkpoint."""

import importlib
import re
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler

from core.config import DignityConfig
from export import to_onnx
from export.to_onnx import export_to_onnx, get_onnx_model_info, verify_onnx_export
from models.dignity import Dignity
from train.checkpoint import (
    build_model_from_meta,
    load_checkpoint,
    make_meta,
    save_checkpoint,
)

FEATURES = [
    "volume",
    "fee_rate",
    "tx_count",
    "volatility",
    "momentum",
    "directional_change",
    "volume_volatility",
]


def _write_checkpoint(path):
    cfg = DignityConfig()
    cfg.model.hidden_size = 8
    cfg.model.n_layers = 1
    cfg.data.seq_len = 10
    meta = make_meta(cfg, FEATURES, np.zeros(7), np.ones(7))
    model = build_model_from_meta(meta)
    optimizer = torch.optim.AdamW(model.parameters())
    save_checkpoint(
        path,
        model,
        optimizer,
        GradScaler("cpu", enabled=False),
        meta,
        epoch=1,
        metrics={"val": {"loss": 0.5}},
        best_metric=0.5,
    )


class TestExport:
    def test_export_reports_successful_verification(self, tmp_path):
        model = Dignity(task="risk", input_size=4, hidden_size=8, n_layers=1)
        out = tmp_path / "m.onnx"
        assert export_to_onnx(model, str(out), input_shape=(1, 20, 4)) is True

    def test_verification_rejects_a_different_model(self, tmp_path):
        torch.manual_seed(0)
        a = Dignity(task="risk", input_size=4, hidden_size=8, n_layers=1)
        torch.manual_seed(1)
        b = Dignity(task="risk", input_size=4, hidden_size=8, n_layers=1)
        path = str(tmp_path / "a.onnx")
        export_to_onnx(a, path, input_shape=(1, 20, 4), verify=False)
        assert verify_onnx_export(b, path, torch.randn(1, 20, 4)) is False


class TestExportCli:
    def test_console_script_points_at_a_callable_main(self):
        text = (Path(__file__).parent.parent / "setup.py").read_text()
        module, attr = re.search(r"dignity-export=([\w.]+):(\w+)", text).groups()
        assert getattr(importlib.import_module(module), attr) is to_onnx.main

    def test_pre_v1_checkpoint_exits_3_with_a_clear_message(self, tmp_path, capsys):
        torch.save({"epoch": 1, "model_state_dict": {}}, tmp_path / "old.pt")
        rc = to_onnx.main(
            [
                "--checkpoint",
                str(tmp_path / "old.pt"),
                "--output",
                str(tmp_path / "x.onnx"),
            ]
        )
        assert rc == 3
        assert "unsupported checkpoint format" in capsys.readouterr().err

    def test_failed_verification_exits_nonzero(self, tmp_path, monkeypatch):
        # Seam test for the exit-code wiring only; real mismatch detection is
        # test_verification_rejects_a_different_model.
        _write_checkpoint(tmp_path / "c.pt")
        monkeypatch.setattr(to_onnx, "verify_onnx_export", lambda *a, **k: False)
        rc = to_onnx.main(
            [
                "--checkpoint",
                str(tmp_path / "c.pt"),
                "--output",
                str(tmp_path / "x.onnx"),
            ]
        )
        assert rc == 1

    def test_benchmark_uses_the_checkpoint_shape(self, tmp_path):
        _write_checkpoint(tmp_path / "c.pt")  # 7 features, seq_len 10: not (1,100,9)
        rc = to_onnx.main(
            [
                "--checkpoint",
                str(tmp_path / "c.pt"),
                "--output",
                str(tmp_path / "x.onnx"),
                "--benchmark",
            ]
        )
        assert rc == 0

    def test_cli_trained_checkpoint_exports_to_onnx(self, tiny_train_config, tmp_path):
        import train.cli

        train.cli.main(["--config", str(tiny_train_config())])
        best = tmp_path / "runs" / "dignity_risk_best.pt"
        meta = load_checkpoint(best)["meta"]
        assert meta.feature_names == FEATURES
        out = tmp_path / "risk.onnx"
        rc = to_onnx.main(
            ["--checkpoint", str(best), "--output", str(out), "--benchmark"]
        )
        assert rc == 0
        assert get_onnx_model_info(str(out))["inputs"][0]["shape"][2] == 7  # not 9
