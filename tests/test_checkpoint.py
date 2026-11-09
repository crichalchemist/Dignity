import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler

from core.config import DignityConfig
from train.checkpoint import (
    FORMAT_VERSION,
    UnsupportedCheckpointError,
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


def _parts():
    cfg = DignityConfig()
    cfg.model.hidden_size, cfg.model.n_layers, cfg.model.cnn_kernel_size = 16, 1, 5
    meta = make_meta(cfg, FEATURES, np.arange(7.0), np.ones(7))
    model = build_model_from_meta(meta)
    return cfg, meta, model, torch.optim.AdamW(model.parameters())


class TestCheckpoint:
    def test_checkpoint_alone_rebuilds_an_identical_model(self, tmp_path):
        _, meta, model, opt = _parts()
        scaler = GradScaler("cpu", init_scale=1024.0)
        save_checkpoint(
            tmp_path / "c.pt",
            model,
            opt,
            scaler,
            meta,
            epoch=3,
            metrics={"val": {"loss": 0.5}},
            best_metric=0.25,
        )
        ck = load_checkpoint(tmp_path / "c.pt")
        assert ck["meta"] == meta and ck["meta"].feature_names == FEATURES
        assert ck["meta"].input_size == 7
        assert ck["meta"].format_version == FORMAT_VERSION
        assert (ck["epoch"], ck["best_metric"]) == (3, 0.25)
        assert ck["grad_scaler_state"]["scale"] == 1024.0  # DGNTY-33
        rebuilt = build_model_from_meta(ck["meta"])
        rebuilt.load_state_dict(ck["model_state"])
        x = torch.randn(2, 10, 7)
        model.eval()
        rebuilt.eval()
        assert torch.equal(model(x)[0], rebuilt(x)[0])
        kernels = [m.kernel_size for m in rebuilt.modules() if isinstance(m, nn.Conv1d)]
        assert kernels == [(5,), (5,)]

    def test_pre_v1_checkpoint_is_rejected_with_a_clear_message(self, tmp_path):
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "metrics": {"val": {"loss": 0.1}},
            },
            tmp_path / "old.pt",
        )
        with pytest.raises(
            UnsupportedCheckpointError, match="unsupported checkpoint format"
        ):
            load_checkpoint(tmp_path / "old.pt")

    def test_unpicklable_legacy_checkpoint_is_rejected_not_crashed(self, tmp_path):
        torch.save({"metrics": {"loss": np.float64(0.1)}}, tmp_path / "np.pt")
        with pytest.raises(UnsupportedCheckpointError):
            load_checkpoint(tmp_path / "np.pt")
