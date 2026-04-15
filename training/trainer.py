"""
trainer.py
──────────
Thin helper — the main training loop lives in train.py (existing code).
This file provides any extra utilities train.py might need.
"""

import torch


def get_optimizer(model, lr=1e-3, weight_decay=0.02):
    """Adam optimizer matching paper Table 4."""
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )