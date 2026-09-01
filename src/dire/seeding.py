"""All randomness in the project is seeded through this module and nowhere else.

Code that needs a numpy Generator should derive it from the run seed
(``np.random.default_rng(seed)``) rather than pulling fresh OS entropy —
``set_all_seeds`` only controls the legacy numpy global state, python's
``random``, and torch.
"""

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> int:
    """Seed python, numpy, and torch from a single value. Returns the seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    # Only affects child processes; the current interpreter's str hashing is
    # fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # MLP-scale models lose nothing measurable to deterministic kernels, and the
    # paper's claims depend on runs being exactly repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
