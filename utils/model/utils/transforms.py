import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def complex_to_chan_dim(x: Tensor) -> Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    assert c == 1
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: Tensor) -> Tensor:
    b, c2, h, w = x.shape
    assert c2 == 2
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()



def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    crop_width, crop_height = shape
    _, data_height, data_width = data.shape

    # Height padding
    if data_height < crop_height:
        pad_h1 = (crop_height - data_height) // 2
        pad_h2 = (crop_height - data_height) - pad_h1
        data = torch.nn.functional.pad(data, (0, 0, pad_h1, pad_h2), mode='constant', value=0)
        data_height = crop_height

    # Width padding
    if data_width < crop_width:
        pad_w1 = (crop_width - data_width) // 2
        pad_w2 = (crop_width - data_width) - pad_w1
        data = torch.nn.functional.pad(data, (pad_w1, pad_w2, 0, 0), mode='constant', value=0)
        data_width = crop_width

    h_from = (data_height - crop_height) // 2
    w_from = (data_width - crop_width) // 2
    h_to = h_from + crop_height
    w_to = w_from + crop_width

    return data[..., h_from:h_to, w_from:w_to]

def image_crop(image: Tensor, crop_size: Optional[Tuple[int, int]] = None) -> Tensor:
    if crop_size is None:
        return image
    return center_crop(image, crop_size).contiguous()


def _calc_uncrop(crop_height: int, in_height: int) -> Tuple[int, int]:
    pad_height = (in_height - crop_height) // 2
    if (in_height - crop_height) % 2 != 0:
        pad_height_top = pad_height + 1
    else:
        pad_height_top = pad_height

    pad_height = in_height - pad_height

    return pad_height_top, pad_height


def image_uncrop(image: Tensor, original_image: Tensor) -> Tensor:
    """Insert values back into original image."""
    in_shape = original_image.shape
    original_image = original_image.clone()

    if in_shape == image.shape:
        return image

    pad_height_top, pad_height = _calc_uncrop(image.shape[-2], in_shape[-2])
    pad_height_left, pad_width = _calc_uncrop(image.shape[-1], in_shape[-1])

    try:
        if len(in_shape) == 2:  # Assuming 2D images
            original_image[pad_height_top:pad_height, pad_height_left:pad_width] = image
        elif len(in_shape) == 3:  # Assuming 3D images with channels
            original_image[
                :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        elif len(in_shape) == 4:  # Assuming 4D images with batch size
            original_image[
                :, :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        else:
            raise RuntimeError(f"Unsupported tensor shape: {in_shape}")
    except RuntimeError:
        print(f"in_shape: {in_shape}, image shape: {image.shape}")
        raise

    return original_image