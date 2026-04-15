"""
losses.py
─────────
Focal Loss (paper §III-D-6, Eq. 4).

L_focal = -α_t (1 - p_t)^γ  log(p_t)

Settings from paper Table 4: α = 1.0, γ = 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Parameters
    ----------
    alpha       : float   class-balancing factor (1.0 = uniform)
    gamma       : float   focusing exponent (2.0 from paper)
    num_classes : int
    reduction   : str     'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha=1.0, gamma=2.0, num_classes=3, reduction="mean"):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.num_classes = num_classes
        self.reduction   = reduction

    def forward(self, logits, targets):
        """
        logits  : (B, num_classes)  raw model outputs
        targets : (B,)              integer class indices
        """
        # Standard cross-entropy gives log(p_t) per sample
        log_pt = F.log_softmax(logits, dim=-1)              # (B, C)
        log_pt = log_pt.gather(1, targets.unsqueeze(1))     # (B, 1)
        log_pt = log_pt.squeeze(1)                          # (B,)

        pt = log_pt.exp()                                   # (B,)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma

        loss = -self.alpha * focal_weight * log_pt          # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss