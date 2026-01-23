"""
Wrapper for calling TimesFM from a separate virtual environment.

This module provides a clean interface for using TimesFM predictions
when it's installed in an isolated environment (e.g., .venvx_timesfm/).
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class TimesFMWrapper:
    """Wrapper for TimesFM model in separate environment."""

    def __init__(self, timesfm_env_path: str | Path = ".venvx_timesfm"):
        """
        Initialize wrapper.

        Args:
            timesfm_env_path: Path to TimesFM virtual environment directory
        """
        self.timesfm_env = Path(timesfm_env_path)
        self.python_bin = self.timesfm_env / "bin" / "python"

        if not self.python_bin.exists():
            raise FileNotFoundError(
                f"TimesFM Python binary not found at {self.python_bin}. "
                f"Ensure the environment exists and is properly set up."
            )

    def forecast_naive(
        self,
        horizon: int,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Forecast using TimesFM's naive method.

        Args:
            horizon: Number of steps to forecast
            inputs: List of input sequences (each is 1D numpy array)

        Returns:
            List of forecast arrays (one per input sequence)
        """
        # Create temporary directory for data exchange
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Serialize inputs
            input_file = temp_path / "inputs.json"
            output_file = temp_path / "outputs.json"

            serialized_inputs = {
                "horizon": horizon,
                "sequences": [seq.tolist() for seq in inputs],
            }

            with open(input_file, "w") as f:
                json.dump(serialized_inputs, f)

            # Call TimesFM subprocess
            script_path = Path(__file__).parent / "timesfm_predict.py"
            _call_timesfm_subprocess(
                self.python_bin,
                script_path,
                input_file,
                output_file,
            )

            # Deserialize outputs
            with open(output_file) as f:
                result = json.load(f)

            forecasts = [
                np.array(seq) if seq is not None else None
                for seq in result["forecasts"]
            ]

            return forecasts


def _call_timesfm_subprocess(
    timesfm_env_path: Path,
    script_path: Path,
    input_file: Path,
    output_file: Path,
) -> None:
    """
    Execute TimesFM prediction script in isolated environment.

    Args:
        timesfm_env_path: Path to Python binary in TimesFM environment
        script_path: Path to prediction script
        input_file: Path to JSON file with input data
        output_file: Path where predictions will be written

    Raises:
        RuntimeError: If subprocess execution fails
    """
    # Construct command: [python_path, script_path, "--input", input_file, "--output", output_file]
    command = [
        str(timesfm_env_path),
        str(script_path),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
    ]

    # Use subprocess.run() with capture_output=True
    result = subprocess.run(command, capture_output=True, text=True)

    # Check returncode and raise RuntimeError with stderr if non-zero
    if result.returncode != 0:
        error_msg = f"TimesFM subprocess failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f": {result.stderr}"
        raise RuntimeError(error_msg)
