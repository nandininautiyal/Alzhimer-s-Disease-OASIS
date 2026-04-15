"""
bifpn3dvit.py — Paper architecture, memory-safe version
Uses GroupNorm throughout, 2 BiFPN levels, 4 transformer blocks.
"""

import torch
import torch.nn as nn

from models.backbone    import Backbone3D
from models.bifpn3d     import BiFPN3D
from models.transformer import TransformerEncoder
from models.classifier  import DualTokenClassifier
from training.losses    import FocalLoss


def gn(channels):
    g = min(8, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class BiFPN3DViT(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        C = cfg

        # 1. Backbone
        self.backbone = Backbone3D(channels=C["backbone_channels"])

        # Backbone exposes indices [5, 6] → both ch=256
        idx       = [5, 6]
        in_ch_list = [C["backbone_channels"][i] for i in idx]

        # 2. BiFPN
        self.bifpn = BiFPN3D(
            in_channels_list = in_ch_list,
            out_channels     = C["bifpn_out_channels"],
            num_levels       = C["bifpn_levels"],
            repeats          = C["bifpn_repeats"],
        )

        # 3. Patch tokeniser
        self.token_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(C["bifpn_out_channels"], C["embed_dim"], kernel_size=1),
                gn(C["embed_dim"]),
                nn.GELU(),
            )
            for _ in range(C["bifpn_levels"])
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, C["embed_dim"]))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding (lazy)
        self.register_buffer("pos_embed", None, persistent=False)
        self._pos_embed_dim = C["embed_dim"]

        # 4. Transformer
        self.transformer = TransformerEncoder(
            embed_dim      = C["embed_dim"],
            num_heads      = C["num_heads"],
            num_blocks     = C["num_transformer_blocks"],
            mlp_ratio      = C["mlp_ratio"],
            dropout        = C["dropout"],
            drop_path_rate = C["drop_path_rate"],
            use_flash      = C.get("use_flash_attention", False),
        )

        # 5. Classifier
        self.classifier = DualTokenClassifier(
            embed_dim   = C["embed_dim"],
            hidden_dim  = C["classifier_hidden"],
            num_classes = C["num_classes"],
            dropout     = C["dropout"],
        )

        # 6. Loss
        self.loss_fn = FocalLoss(
            alpha       = C["focal_alpha"],
            gamma       = C["focal_gamma"],
            num_classes = C["num_classes"],
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_pos_embed(self, n_tokens, device):
        if self.pos_embed is not None and \
           self.pos_embed.shape[1] == n_tokens + 1:
            return self.pos_embed
        pe = torch.zeros(1, n_tokens + 1, self._pos_embed_dim, device=device)
        self.pos_embed = pe
        return pe

    def forward(self, x):
        B = x.shape[0]

        # 1. Backbone → 2 feature maps
        features = self.backbone(x)

        # 2. BiFPN fusion
        fused = self.bifpn(features)

        # 3. Tokenise each scale → flatten → concat
        tokens = []
        for feat, proj in zip(fused, self.token_proj):
            t = proj(feat)                       # (B, E, d, h, w)
            t = t.flatten(2).transpose(1, 2)     # (B, N, E)
            tokens.append(t)
        tokens = torch.cat(tokens, dim=1)        # (B, total_N, E)

        # 4. Prepend CLS + positional embed
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._get_pos_embed(tokens.shape[1] - 1, x.device)

        # 5. Transformer encoder
        tokens = self.transformer(tokens)

        # 6. Dual-token classifier
        logits = self.classifier(tokens)

        return logits, tokens

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


def create_model(cfg: dict) -> BiFPN3DViT:
    return BiFPN3DViT(cfg)