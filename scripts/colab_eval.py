"""Run the risk evaluation on a Colab VM.

Driven by `colab exec -f scripts/colab_eval.py` (see docs/RESULTS.md). Reads the
env vars DIGNITY_SHA (commit to evaluate), DIGNITY_SEED (generator seed; not needed
for a rehearsal), DIGNITY_DATA_SEED (optional tuning data seed) and DIGNITY_REHEARSAL
(optional: the arguments of scripts/rehearse.py, run instead of the final command).
"""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/crichalchemist/Dignity"
REPO_DIR = "/content/Dignity"
# Never `pip install -r requirements.txt`: its numpy<2 pin downgrades Colab's stack.
MISSING_MODULE_PACKAGES = {"yaml": "pyyaml"}


def main():
    sha = os.environ["DIGNITY_SHA"]
    rehearsal = os.environ.get("DIGNITY_REHEARSAL")
    seed = None if rehearsal else os.environ["DIGNITY_SEED"]
    data_seed = os.environ.get("DIGNITY_DATA_SEED")

    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--tags", "origin"], check=True)
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

    from eval.run import _dirty_frozen_paths
    from eval.run import main as eval_main

    print(f"numpy {numpy.__version__}")
    print(f"torch {torch.__version__}")
    print(f"pandas {pandas.__version__}")

    if rehearsal:
        from scripts.rehearse import main as rehearse_main

        tarball = "/content/dignity_rehearsal.tar.gz"
        paths = ["runs/rehearsal"]
    else:
        tarball = (
            f"/content/dignity_results_{seed}.tar.gz"
            if data_seed is None
            else f"/content/dignity_results_tuning_{data_seed}.tar.gz"
        )
        paths = ["results/v1.0", "docs/img"]
    try:
        if rehearsal:
            assert rehearse_main(shlex.split(rehearsal)) == 0
        else:
            argv = ["--config", "config/synthetic_hard.yaml", "--seed", seed]
            if data_seed is not None:
                argv += ["--data-seed", data_seed]
            tasks = ["--tasks", "risk,forecast,policy"]
            code = eval_main([*argv, *tasks, "--privacy-sweep"])
            if code == 1:
                print("EVAL_EXIT=1 (a training raise was recorded in the results)")
            else:
                assert code == 0
    finally:
        # informational: would the final run's clean-tree check pass on this VM;
        # it never masks the outcome of the run, so a helper failure prints unknown
        try:
            dirty = _dirty_frozen_paths(Path(REPO_DIR))
            print(f"FROZEN_DIRTY={','.join(dirty) or 'none'}")
        except Exception:  # noqa: BLE001 - informational line only
            print("FROZEN_DIRTY=unknown")
        # results already on disk are kept even when the sweep (last) fails
        subprocess.run(["tar", "-czf", tarball, *paths], check=True)


if __name__ == "__main__":
    main()
