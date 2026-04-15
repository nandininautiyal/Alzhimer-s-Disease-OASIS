"""
transformer.py
──────────────
Transformer encoder blocks (paper §III-D-4).

Uses FlashAttention if available and config says so,
otherwise falls back to standard torch MultiheadAttention.

Each block:
  LayerNorm → merged QKV projection → FlashAttention → dropout
  LayerNorm → MLP (GELU) → dropout
  + DropPath (stochastic depth) on both sub-layers
  + LayerScale (learnable per-channel scalar)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── DropPath (stochastic depth) ───────────────────────────────────────────────

class DropPath(nn.Module):
    """Drop entire residual paths with probability `drop_prob` during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


# ── Attention ─────────────────────────────────────────────────────────────────

class FlashAttentionBlock(nn.Module):
    """
    Multi-head attention with merged QKV projection.
    Uses flash_attn if installed, otherwise falls back to torch SDPA.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, use_flash=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv     = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale   = math.sqrt(self.head_dim)

        # Try to import flash_attn
        self.use_flash = False
        if use_flash:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self._flash_fn = flash_attn_qkvpacked_func
                self.use_flash = True
                print("[transformer] Using FlashAttention ✓")
            except ImportError:
                print("[transformer] flash_attn not found, using torch SDPA fallback.")

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.use_flash:
            # flash_attn expects (B, N, 3, H, D) — already that shape
            out = self._flash_fn(qkv, dropout_p=self.dropout.p if self.training else 0.0)
            out = out.reshape(B, N, C)
        else:
            # Torch scaled dot-product attention (efficient built-in)
            q, k, v = qkv.unbind(dim=2)         # each (B, N, H, D)
            q = q.transpose(1, 2)               # (B, H, N, D)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One ViT-style transformer block with:
      - LayerNorm pre-norm
      - FlashAttention (or SDPA fallback)
      - LayerScale
      - DropPath (stochastic depth)
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4,
                 dropout=0.0, drop_path=0.0, use_flash=True,
                 layer_scale_init=1e-4):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = FlashAttentionBlock(embed_dim, num_heads, dropout, use_flash)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = MLP(embed_dim, mlp_ratio, dropout)
        self.drop1  = DropPath(drop_path)
        self.drop2  = DropPath(drop_path)

        # LayerScale — learnable per-channel scalar (paper §III-D-5)
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(embed_dim))

    def forward(self, x):
        x = x + self.drop1(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop2(self.gamma2 * self.mlp(self.norm2(x)))
        return x


# ── Transformer Encoder (stack of N blocks) ───────────────────────────────────

class TransformerEncoder(nn.Module):
    """
    Stack of `num_blocks` TransformerBlocks with linearly increasing
    drop-path rates (stochastic depth schedule).
    """

    def __init__(self, embed_dim, num_heads, num_blocks=8,
                 mlp_ratio=4, dropout=0.0, drop_path_rate=0.1, use_flash=True):
        super().__init__()
        # Linear schedule from 0 → drop_path_rate
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dp_rates[i],
                use_flash=use_flash,
            )
            for i in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)