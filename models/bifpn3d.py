"""
bifpn3d.py
──────────
3D BiFPN — uses GroupNorm instead of BatchNorm so it works
at any spatial size including 2x2x2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(channels):
    """GroupNorm that works for any channel count."""
    groups = min(8, channels)
    # groups must divide channels evenly
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvAttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = gn(channels)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

        # Channel attention
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 1), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        w   = self.pool(out).flatten(1)
        w   = self.fc(w).view(-1, out.shape[1], 1, 1, 1)
        return out * w


class BiFPNLevel3D(nn.Module):
    def __init__(self, out_channels, num_levels):
        super().__init__()
        self.num_levels = num_levels

        self.td_w = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)
        ])
        self.bu_w = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(num_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            ConvAttentionBlock3D(out_channels) for _ in range(num_levels - 1)
        ])
        self.bu_convs = nn.ModuleList([
            ConvAttentionBlock3D(out_channels) for _ in range(num_levels - 1)
        ])

    @staticmethod
    def _resize(src, target):
        if src.shape[2:] == target.shape[2:]:
            return src
        return F.adaptive_avg_pool3d(src, target.shape[2:])

    @staticmethod
    def _fuse(x, y, weights):
        w = F.softmax(weights, dim=0)
        return w[0] * x + w[1] * y

    def forward(self, features):
        # Top-down
        td = [features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            up   = self._resize(td[-1], features[i])
            fuse = self._fuse(features[i], up, self.td_w[self.num_levels - 2 - i])
            td.append(self.td_convs[self.num_levels - 2 - i](fuse))
        td.reverse()

        # Bottom-up
        bu = [td[0]]
        for i in range(1, self.num_levels):
            down = self._resize(bu[-1], td[i])
            fuse = self._fuse(td[i], down, self.bu_w[i - 1])
            bu.append(self.bu_convs[i - 1](fuse))

        return bu


class BiFPN3D(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_levels, repeats=1):
        super().__init__()
        self.num_levels = num_levels

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_channels, kernel_size=1, bias=False),
                gn(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )
            for in_ch in in_channels_list
        ])

        self.bifpn_layers = nn.ModuleList([
            BiFPNLevel3D(out_channels, num_levels) for _ in range(repeats)
        ])

    def forward(self, features):
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        for layer in self.bifpn_layers:
            projected = layer(projected)
        return projected