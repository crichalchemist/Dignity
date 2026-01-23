"""Tests for TimesFM wrapper subprocess functionality."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.timesfm_wrapper import TimesFMWrapper, _call_timesfm_subprocess


def test_call_timesfm_subprocess_success():
    """Test successful subprocess execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock input and output files
        input_file = temp_path / "input.json"
        output_file = temp_path / "output.json"

        input_data = {"horizon": 10, "sequences": [[1.0, 2.0, 3.0]]}
        with open(input_file, "w") as f:
            json.dump(input_data, f)

        # Create output file (simulating successful script execution)
        output_data = {"forecasts": [[4.0, 5.0, 6.0]]}
        with open(output_file, "w") as f:
            json.dump(output_data, f)

        # Mock subprocess.run to simulate success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            python_bin = Path("/usr/bin/python3")
            script_path = temp_path / "script.py"

            # Should not raise any exception
            _call_timesfm_subprocess(python_bin, script_path, input_file, output_file)


def test_call_timesfm_subprocess_failure():
    """Test subprocess execution failure with non-zero exit code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_file = temp_path / "input.json"
        output_file = temp_path / "output.json"

        # Mock subprocess.run to simulate failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ImportError: Failed to import TimesFM"

        with patch("subprocess.run", return_value=mock_result):
            python_bin = Path("/usr/bin/python3")
            script_path = temp_path / "script.py"

            # Should raise RuntimeError with error details
            with pytest.raises(RuntimeError) as exc_info:
                _call_timesfm_subprocess(python_bin, script_path, input_file, output_file)

            assert "exit code 1" in str(exc_info.value)
            assert "ImportError: Failed to import TimesFM" in str(exc_info.value)


def test_call_timesfm_subprocess_failure_no_stderr():
    """Test subprocess failure without stderr output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_file = temp_path / "input.json"
        output_file = temp_path / "output.json"

        # Mock subprocess.run to simulate failure without stderr
        mock_result = MagicMock()
        mock_result.returncode = 127
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            python_bin = Path("/usr/bin/python3")
            script_path = temp_path / "script.py"

            # Should raise RuntimeError with exit code
            with pytest.raises(RuntimeError) as exc_info:
                _call_timesfm_subprocess(python_bin, script_path, input_file, output_file)

            assert "exit code 127" in str(exc_info.value)


def test_call_timesfm_subprocess_command_construction():
    """Test that the subprocess command is constructed correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        input_file = temp_path / "input.json"
        output_file = temp_path / "output.json"
        python_bin = Path("/path/to/python")
        script_path = Path("/path/to/script.py")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _call_timesfm_subprocess(python_bin, script_path, input_file, output_file)

            # Verify subprocess.run was called with correct arguments
            mock_run.assert_called_once()
            call_args = mock_run.call_args

            expected_command = [
                str(python_bin),
                str(script_path),
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ]

            assert call_args[0][0] == expected_command
            assert call_args[1]["capture_output"] is True
            assert call_args[1]["text"] is True


def test_timesfm_wrapper_init_valid_env():
    """Test TimesFMWrapper initialization with valid environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock virtual environment structure
        bin_dir = temp_path / "bin"
        bin_dir.mkdir()
        python_bin = bin_dir / "python"
        python_bin.touch()

        # Should initialize successfully
        wrapper = TimesFMWrapper(timesfm_env_path=temp_path)
        assert wrapper.timesfm_env == temp_path
        assert wrapper.python_bin == python_bin


def test_timesfm_wrapper_init_invalid_env():
    """Test TimesFMWrapper initialization with invalid environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Don't create the python binary
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            TimesFMWrapper(timesfm_env_path=temp_path)

        assert "TimesFM Python binary not found" in str(exc_info.value)


def test_timesfm_wrapper_forecast_naive_integration():
    """Test forecast_naive method with mocked subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create mock virtual environment structure
        bin_dir = temp_path / "bin"
        bin_dir.mkdir()
        python_bin = bin_dir / "python"
        python_bin.touch()

        wrapper = TimesFMWrapper(timesfm_env_path=temp_path)

        # Mock subprocess to simulate successful prediction
        def mock_subprocess_run(command, **kwargs):
            # Extract input and output file paths from command
            input_idx = command.index("--input") + 1
            output_idx = command.index("--output") + 1
            input_file = Path(command[input_idx])
            output_file = Path(command[output_idx])

            # Read input
            with open(input_file) as f:
                input_data = json.load(f)

            # Create mock output (simple echo of inputs)
            forecasts = []
            for seq in input_data["sequences"]:
                # Mock forecast: just append some values
                forecast = seq + [seq[-1] + 0.1 * i for i in range(1, input_data["horizon"] + 1)]
                forecasts.append(forecast[:input_data["horizon"]])

            output_data = {"forecasts": forecasts}
            with open(output_file, "w") as f:
                json.dump(output_data, f)

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            inputs = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
            forecasts = wrapper.forecast_naive(horizon=3, inputs=inputs)

            assert len(forecasts) == 1
            assert forecasts[0] is not None
            assert len(forecasts[0]) == 3
