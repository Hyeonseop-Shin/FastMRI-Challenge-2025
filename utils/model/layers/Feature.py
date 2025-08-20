import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeatureImage(NamedTuple):
    features: Tensor
    sens_maps: Optional[Tensor] = None
    crop_size: Optional[Tuple[int, int]] = None
    means: Optional[Tensor] = None
    variances: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    ref_kspace: Optional[Tensor] = None
    beta: Optional[Tensor] = None
    gamma: Optional[Tensor] = None


class FeatureEncoder(nn.Module):
    def __init__(self, in_chans: int = 2, feature_chans: int = 32, drop_prob: float = 0.0):
        super().__init__()
        self.feature_chans = feature_chans

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=feature_chans,
                kernel_size=5,
                padding='same',
                bias=True,
            ),
        )

    def forward(self, image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.encoder((image - means) * torch.rsqrt(variances))


class FeatureDecoder(nn.Module):
    def __init__(self, feature_chans: int = 32, out_chans: int = 2):
        super().__init__()
        self.feature_chans = feature_chans

        self.decoder = nn.Conv2d(
            in_channels=feature_chans,
            out_channels=out_chans,
            kernel_size=5,
            padding='same',
            bias=True,
        )

    def forward(self, features: Tensor, means: Tensor, variances: Tensor) -> Tensor:
        means = means.view(1, -1, 1, 1)
        variances = variances.view(1, -1, 1, 1)
        return self.decoder(features) * torch.sqrt(variances) + means

