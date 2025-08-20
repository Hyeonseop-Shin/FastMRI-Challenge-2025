import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.model.utils.fftc import fft2c, ifft2c
from utils.model.utils.math import complex_mul, complex_conj, complex_abs_sq
from utils.model.utils.transforms import chan_complex_to_last_dim, complex_to_chan_dim


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))



def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def sens_expand(x: Tensor, sens_maps: Tensor) -> Tensor:
    return fft2c(complex_mul(x, sens_maps))


def sens_reduce(x: Tensor, sens_maps: Tensor) -> Tensor:
    return complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)
    
