"""
scheduler.py
────────────
Learning rate scheduler — step decay (paper Table 4: constant or 10^-1 decay).
"""

import torch


def get_scheduler(optimizer, step_size=10, gamma=0.1):
    """
    StepLR: multiply lr by `gamma` every `step_size` epochs.
    Paper uses γ = 10^-1 = 0.1.
    """
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )