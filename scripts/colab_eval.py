"""Run the risk evaluation on a Colab VM.

Driven by `colab exec -f scripts/colab_eval.py` (see docs/RESULTS.md). Reads the
env vars DIGNITY_SHA (commit to evaluate) and DIGNITY_SEED (generator seed).
"""

import importlib.util
import os
import subprocess
import sys

REPO_URL = "https://github.com/crichalchemist/Dignity"
REPO_DIR = "/content/Dignity"
# Never `pip install -r requirements.txt`: its numpy<2 pin downgrades Colab's stack.
MISSING_MODULE_PACKAGES = {"yaml": "pyyaml"}


def main():
    sha = os.environ["DIGNITY_SHA"]
    seed = os.environ["DIGNITY_SEED"]

    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", sha], check=True)

    for module, package in MISSING_MODULE_PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package], check=True
            )

    os.chdir(REPO_DIR)
    sys.path.insert(0, REPO_DIR)

    import numpy
    import pandas
    import torch

    from eval.run import main as eval_main

    print(f"numpy {numpy.__version__}")
    print(f"torch {torch.__version__}")
    print(f"pandas {pandas.__version__}")

    argv = ["--config", "config/synthetic_hard.yaml", "--seed", seed]
    assert eval_main([*argv, "--tasks", "risk,forecast,policy"]) == 0

    subprocess.run(
        [
            "tar",
            "-czf",
            f"/content/dignity_results_{seed}.tar.gz",
            "results/v1.0",
            "docs/img",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
