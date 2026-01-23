import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_imports():
    """Test that core modules can be imported."""
    modules = [
        "data.gdelt_ingest",
        "run.training_pipeline",
        "models.agent_hybrid",
        "models.signal_policy",
        "features.agent_features",
        "config.constants",
    ]
    for mod in modules:
        importlib.import_module(mod)


def docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not docker_available(), reason="Docker/Testcontainers not available")
def test_testcontainers_echo():
    from testcontainers.core.container import DockerContainer

    with DockerContainer("alpine:3.18").with_command("echo ok") as container:
        output = container.get_logs().decode("utf-8")
        assert "ok" in output.lower()


def test_training_pipeline_help(tmp_path: Path):
    # Ensure the CLI can render help without accessing data.
    result = subprocess.run(
        [sys.executable, "run/training_pipeline.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage" in result.stdout.lower()
