"""
backbone.py
───────────
7-block 3D CNN backbone.
Block 6 does NOT pool (keeps spatial size at 2^3 minimum).
Exposes blocks 5 and 6 output for BiFPN (2 levels).

Spatial sizes with 128^3 input:
  block 0: pool → 64^3  (ch=8)
  block 1: pool → 32^3  (ch=16)
  block 2: pool → 16^3  (ch=32)
  block 3: pool → 8^3   (ch=64)
  block 4: pool → 4^3   (ch=128)
  block 5: pool → 2^3   (ch=256)  ← F0
  block 6: NO pool → 2^3 (ch=256) ← F1  (same spatial, different features)
"""

import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),   # GroupNorm works at any spatial size
            nn.LeakyReLU(0.1, inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Backbone3D(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [8, 16, 32, 64, 128, 256, 256]
        assert len(channels) == 7

        in_ch  = 1
        blocks = []
        for i, out_ch in enumerate(channels):
            # Last block: no pooling to avoid 1x1x1
            pool = (i < 6)
            blocks.append(ConvBlock3D(in_ch, out_ch, pool=pool))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Expose last 2 blocks → both are 2^3 spatial
        self.feature_indices = [5, 6]

    def forward(self, x):
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.feature_indices:
                features.append(x)
        return features