"""Custom backbones referenced by the repository configs."""

from __future__ import annotations

import torch
import torch.nn as nn

from mmseg.models.backbones import MobileNetV2
from mmseg.registry import MODELS


@MODELS.register_module()
class MobileNetV2_BiSeNetAdapted(MobileNetV2):
    """Project MobileNetV2 features to the channels expected by BiSeNetV1."""

    def __init__(self, context_channels=(32, 160, 320), **kwargs):
        super().__init__(**kwargs)
        self.context_channels = context_channels
        self.project_layers = nn.ModuleList()

        with torch.no_grad():
            was_training = self.training
            self.eval()
            dummy_outs = super().forward(torch.zeros(1, 3, 128, 128))
            if was_training:
                self.train()

        real_channels = [feat.shape[1] for feat in dummy_outs[1:4]]
        for in_channels, out_channels in zip(real_channels, context_channels):
            if in_channels == out_channels:
                self.project_layers.append(nn.Identity())
            else:
                self.project_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        outs = list(super().forward(x))
        projected = [
            proj(feat) for proj, feat in zip(self.project_layers, outs[1:4])
        ]
        return (outs[0], *projected)
