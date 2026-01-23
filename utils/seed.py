"""Global random‑seed helper for reproducible experiments.

The repository runs stochastic components from ``random``, ``numpy`` and
``torch``.  A single function ``set_seed`` synchronises all three libraries
and also configures deterministic CUDA behaviour when a GPU is available.
"""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np
import torch

DEFAULT_SEED: Final[int] = 42


def set_seed(seed: int | None = None) -> int:
    """Set the seed for ``random``, ``numpy`` and ``torch`` for reproducibility.

    Parameters
    ----------
    seed : int or None
        Optional integer seed.  If ``None`` the function reads the
        ``SEQ_GLOBAL_SEED`` environment variable; if that is also unset a
        deterministic default of ``42`` is used.

    Returns
    -------
    int
        The seed value that was set (useful for logging).
    """
    if seed is None:
        seed = int(os.getenv("SEQ_GLOBAL_SEED", DEFAULT_SEED))

    # Set seeds for all stochastic libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set Python's hash seed for reproducible dict iteration
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic GPU kernels when using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
