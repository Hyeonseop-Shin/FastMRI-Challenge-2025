import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.model.utils.fftc import fft2c, ifft2c
from utils.model.utils.math import complex_mul, complex_conj
from utils.model.utils.coil_combine import sens_expand, sens_reduce

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))


    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:

        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask.to(torch.bool), current_kspace - ref_kspace, zero) * self.dc_weight

        model_term = sens_expand(
            self.model(sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term
