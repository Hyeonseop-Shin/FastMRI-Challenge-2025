import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NormStats(nn.Module):
    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        # group norm
        batch, chans, _, _ = data.shape

        if batch != 1:
            raise ValueError("Unexpected input dimensions.")

        data = data.view(chans, -1)

        mean = data.mean(dim=1)
        variance = data.var(dim=1, unbiased=False)

        assert mean.ndim == 1
        assert variance.ndim == 1
        assert mean.shape[0] == chans
        assert variance.shape[0] == chans

        return mean, variance
