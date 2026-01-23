"""
TimesFM prediction script - runs in isolated TimesFM environment.

This script is called by timesfm_wrapper.py via subprocess to perform
predictions using the TimesFM model.
"""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="TimesFM prediction worker")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    # Load inputs
    with open(args.input) as f:
        data = json.load(f)

    horizon = data["horizon"]
    sequences = [np.array(seq, dtype=np.float32) for seq in data["sequences"]]

    # Load TimesFM model (only import here, in TimesFM environment)
    try:
        from timesfm.timesfm_2p5 import timesfm_2p5_torch

        model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import TimesFM. Ensure it's installed in this environment: {e}"
        )

    # Run predictions
    forecasts = model.forecast_naive(horizon=horizon, inputs=sequences)

    # Serialize outputs (handle None values)
    serialized_forecasts = [
        forecast.tolist() if forecast is not None else None for forecast in forecasts
    ]

    result = {"forecasts": serialized_forecasts}

    # Write outputs
    with open(args.output, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
