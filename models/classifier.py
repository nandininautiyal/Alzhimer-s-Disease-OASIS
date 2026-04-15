"""
classifier.py
─────────────
Dual-token classification head (paper §III-D-5).

Strategy:
  1. CLS token  → global context
  2. Attention-pooled patch tokens → region-aware summary
  3. Concatenate → 2-layer MLP → softmax (3 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Compute a weighted summary of patch tokens.
    Weights are learned via a small linear projection + softmax.
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.attn_w = nn.Linear(embed_dim, 1, bias=False)
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, C)  — exclude the CLS token (index 0)
        x = self.norm(patch_tokens)                     # (B, N, C)
        scores  = self.attn_w(x).squeeze(-1)            # (B, N)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, 1)
        pooled  = (weights * patch_tokens).sum(dim=1)   # (B, C)
        return pooled


class DualTokenClassifier(nn.Module):
    """
    Parameters
    ----------
    embed_dim   : int   transformer output dimension
    hidden_dim  : int   hidden layer size in MLP head
    num_classes : int   3 for AD/MCI/CN
    dropout     : float
    """

    def __init__(self, embed_dim, hidden_dim=256, num_classes=3, dropout=0.4):
        super().__init__()
        self.attn_pool = AttentionPooling(embed_dim)
        self.norm      = nn.LayerNorm(embed_dim * 2)   # after concat

        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens):
        """
        tokens : (B, 1+N, C)   first token is CLS
        returns: (B, num_classes) logits
        """
        cls_token    = tokens[:, 0, :]          # (B, C)
        patch_tokens = tokens[:, 1:, :]         # (B, N, C)
        pooled       = self.attn_pool(patch_tokens)   # (B, C)

        fused  = torch.cat([cls_token, pooled], dim=-1)  # (B, 2C)
        fused  = self.norm(fused)
        logits = self.head(fused)               # (B, num_classes)
        return logits