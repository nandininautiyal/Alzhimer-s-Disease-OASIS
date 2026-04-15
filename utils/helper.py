"""
helper.py
─────────
Reproducibility and checkpoint utilities.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model