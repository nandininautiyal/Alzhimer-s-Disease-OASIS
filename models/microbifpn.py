"""
microbifpn.py
─────────────
MicroBiFPN: Lightweight hybrid CNN + BiFPN + Attention
Proposed novelty for Alzheimer classification.

Works on:
  - OASIS dataset (2D slices, 4 classes)
  - ADNI dataset (can use 2D slices extracted from 3D MRI)

Input:  (B, 3, 224, 224)  — standard 2D image
Output: (B, num_classes)

Architecture:
  1. Lightweight CNN encoder (4 stages, MobileNet-style)
  2. BiFPN multi-scale fusion (top-down + bottom-up)
  3. Channel + spatial attention
  4. Dual-head classifier (global + local)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.losses import FocalLoss


# ── Building blocks ───────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv — 8x fewer params than standard conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, ch, ratio=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, max(ch // ratio, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // ratio, 8), ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(-1, x.shape[1], 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention — highlights important regions."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        att = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Convolutional Block Attention Module = SE + Spatial."""
    def __init__(self, ch):
        super().__init__()
        self.se      = SEBlock(ch)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.se(x))


# ── BiFPN for 2D ─────────────────────────────────────────────────────────────

class BiFPNBlock(nn.Module):
    """One BiFPN round over 3 scales."""
    def __init__(self, ch):
        super().__init__()
        # Fusion weights
        self.td_w = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(2)])
        self.bu_w = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(2)])

        # Post-merge convs
        self.td_convs = nn.ModuleList([ConvBNReLU(ch, ch) for _ in range(2)])
        self.bu_convs = nn.ModuleList([ConvBNReLU(ch, ch) for _ in range(2)])

    @staticmethod
    def _resize(src, target):
        if src.shape[2:] == target.shape[2:]:
            return src
        return F.interpolate(src, size=target.shape[2:],
                            mode='bilinear', align_corners=False)

    @staticmethod
    def _fuse(x, y, w):
        w = F.softmax(w, dim=0)
        return w[0] * x + w[1] * y

    def forward(self, features):
        f0, f1, f2 = features   # shallow→deep

        # Top-down: f2→f1, f1→f0
        f1_td = self.td_convs[0](self._fuse(f1, self._resize(f2, f1), self.td_w[0]))
        f0_td = self.td_convs[1](self._fuse(f0, self._resize(f1_td, f0), self.td_w[1]))

        # Bottom-up: f0→f1, f1→f2
        f1_bu = self.bu_convs[0](self._fuse(f1_td, self._resize(f0_td, f1_td), self.bu_w[0]))
        f2_bu = self.bu_convs[1](self._fuse(f2,    self._resize(f1_bu, f2),    self.bu_w[1]))

        return [f0_td, f1_bu, f2_bu]


# ── Full Model ────────────────────────────────────────────────────────────────

class MicroBiFPN(nn.Module):
    """
    MicroBiFPN: Lightweight 2D CNN + BiFPN + CBAM attention.

    Encoder stages (224→112→56→28→14):
      Stage 0: 3   → 32   (stride 2)
      Stage 1: 32  → 64   (stride 2) ← BiFPN level 0
      Stage 2: 64  → 128  (stride 2) ← BiFPN level 1
      Stage 3: 128 → 256  (stride 2) ← BiFPN level 2
    """

    def __init__(self, cfg):
        super().__init__()
        nc   = cfg["num_classes"]
        drop = cfg["dropout"]
        bifpn_ch = 128

        # Encoder
        self.stage0 = ConvBNReLU(3,   32,  s=2)   # 224→112
        self.stage1 = nn.Sequential(               # 112→56
            DepthwiseSeparableConv(32,  64,  stride=2),
            DepthwiseSeparableConv(64,  64),
        )
        self.stage2 = nn.Sequential(               # 56→28
            DepthwiseSeparableConv(64,  128, stride=2),
            DepthwiseSeparableConv(128, 128),
        )
        self.stage3 = nn.Sequential(               # 28→14
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256),
        )

        # Project all 3 BiFPN levels to same channel width
        self.proj1 = ConvBNReLU(64,  bifpn_ch, k=1, p=0)
        self.proj2 = ConvBNReLU(128, bifpn_ch, k=1, p=0)
        self.proj3 = ConvBNReLU(256, bifpn_ch, k=1, p=0)

        # BiFPN (2 rounds)
        self.bifpn = nn.Sequential(
            BiFPNBlock(bifpn_ch),
            BiFPNBlock(bifpn_ch),
        )

        # CBAM attention on fused features
        self.cbam = CBAM(bifpn_ch)

        # Global + local pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(bifpn_ch * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, nc),
        )

        self.loss_fn = FocalLoss(
            alpha=cfg["focal_alpha"],
            gamma=cfg["focal_gamma"],
            num_classes=nc,
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encode
        x0 = self.stage0(x)    # (B,32,112,112)
        x1 = self.stage1(x0)   # (B,64,56,56)
        x2 = self.stage2(x1)   # (B,128,28,28)
        x3 = self.stage3(x2)   # (B,256,14,14)

        # Project to BiFPN channel width
        p1 = self.proj1(x1)    # (B,128,56,56)
        p2 = self.proj2(x2)    # (B,128,28,28)
        p3 = self.proj3(x3)    # (B,128,14,14)

        # BiFPN multi-scale fusion
        p1, p2, p3 = self.bifpn[0]([p1, p2, p3])
        p1, p2, p3 = self.bifpn[1]([p1, p2, p3])

        # Use deepest fused feature + CBAM attention
        feat = self.cbam(p3)   # (B,128,14,14)

        # Dual pooling: avg + max → richer representation
        avg  = self.gap(feat).flatten(1)   # (B,128)
        mx   = self.gmp(feat).flatten(1)   # (B,128)
        fused = torch.cat([avg, mx], dim=1) # (B,256)

        logits = self.classifier(fused)     # (B, nc)
        return logits, fused

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


def create_model(cfg):
    return MicroBiFPN(cfg)