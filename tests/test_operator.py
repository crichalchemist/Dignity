"""Operator-layer guarantees. See docs/THREAT-MODEL.md, Adversary B.

'Deniability' means exactly the three properties tested here: no network I/O in
the inference/export path, a self-contained ONNX artifact, nothing that phones home.
"""

import ast
import re
import socket
from pathlib import Path

import onnx
import torch

from export.to_onnx import export_to_onnx
from models.dignity import Dignity

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ("core", "data", "models", "export", "train")
NETWORK_MODULES = {
    "socket",
    "http",
    "urllib",
    "ssl",
    "requests",
    "aiohttp",
    "ccxt",
    "websocket",
    "websockets",
}
ABSOLUTE_PATH = re.compile(r"(^|\s)/(home|Users|tmp|var|opt|mnt)/")


def _imported_roots(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def _tiny_model() -> Dignity:
    return Dignity(task="risk", input_size=4, hidden_size=16, n_layers=1)


class TestOperatorLayer:
    def test_inference_path_imports_no_network_modules(self):
        offenders = []
        for pkg in INFERENCE_PATH:
            assert (ROOT / pkg).is_dir(), pkg
            for py in (ROOT / pkg).rglob("*.py"):
                for root in _imported_roots(py):
                    if root in NETWORK_MODULES:
                        offenders.append(f"{py.relative_to(ROOT)} imports {root}")
        assert offenders == []

    def test_onnx_export_is_self_contained(self, tmp_path):
        out = tmp_path / "tiny.onnx"
        export_to_onnx(_tiny_model(), str(out), input_shape=(1, 20, 4), verify=False)

        model = onnx.load(str(out), load_external_data=False)
        external = [
            t.name
            for t in model.graph.initializer
            if t.data_location == onnx.TensorProto.EXTERNAL
        ]
        assert external == [], "ONNX must not reference external data files"

        text = " ".join(
            [
                model.doc_string,
                model.graph.doc_string,
                *(p.value for p in model.metadata_props),
            ]
        )
        assert "://" not in text, "artifact metadata must not embed URLs"
        assert not ABSOLUTE_PATH.search(text), (
            "artifact metadata must not embed local paths"
        )

    def test_predict_succeeds_with_sockets_disabled(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("inference attempted to open a socket")

        monkeypatch.setattr(socket, "socket", refuse)
        out = _tiny_model().predict(torch.randn(2, 20, 4))
        assert out.shape[0] == 2

    def test_export_succeeds_with_sockets_disabled(self, monkeypatch, tmp_path):
        def refuse(*args, **kwargs):
            raise AssertionError("export attempted to open a socket")

        monkeypatch.setattr(socket, "socket", refuse)
        out = tmp_path / "tiny.onnx"
        export_to_onnx(_tiny_model(), str(out), input_shape=(1, 20, 4), verify=False)
        assert out.stat().st_size > 0
