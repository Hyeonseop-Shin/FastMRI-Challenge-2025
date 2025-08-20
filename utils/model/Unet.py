import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as cp

from utils.model.layers.ConvBlock import ConvBlock, TransposeConvBlock
from utils.model.utils.transforms import chan_complex_to_last_dim, complex_to_chan_dim

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output



class NormUnet(nn.Module):
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):

        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        # attention_goes_here
        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = chan_complex_to_last_dim(x)

        return x


class UnetLevel(nn.Module):
    def __init__(
        self,
        child: Optional[nn.Module],
        in_planes: int,
        out_planes: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.left_block = ConvBlock(
            in_chans=in_planes, out_chans=out_planes, drop_prob=drop_prob
        )

        self.child = child

        if child is not None:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            if isinstance(child, UnetLevel):  # Ensure child is an instance of UnetLevel
                self.upsample = TransposeConvBlock(
                    in_chans=child.out_planes, out_chans=out_planes
                )
            else:
                raise TypeError("Child must be an instance of UnetLevel")

            self.right_block = ConvBlock(
                in_chans=2 * out_planes, out_chans=out_planes, drop_prob=drop_prob
            )

    def down_up(self, image: Tensor) -> Tensor:
        if self.child is None:
            raise ValueError("self.child is None, cannot call down_up.")
        downsampled = self.downsample(image)
        child_output = self.child(downsampled)
        upsampled = self.upsample(child_output)
        return upsampled

    def forward(self, image: Tensor) -> Tensor:
        image = self.left_block(image)

        if self.child is not None:
            image = self.right_block(torch.cat((image, self.down_up(image)), 1))

        return image
    

class Unet2d(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        output_bias: bool = False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_planes = out_chans
        self.factor = 2**num_pool_layers

        # Build from the middle of the UNet outwards
        planes = 2 ** (num_pool_layers)
        layer = None
        for _ in range(num_pool_layers):
            planes = planes // 2
            layer = UnetLevel(
                layer,
                in_planes=planes * chans,
                out_planes=2 * planes * chans,
                drop_prob=drop_prob,
            )

        self.layer = UnetLevel(
            layer, in_planes=in_chans, out_planes=chans, drop_prob=drop_prob
        )

        if output_bias:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
            )
        else:
            self.final_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=chans,
                    out_channels=out_chans,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

    def pad_input_image(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # pad image if it's not divisible by downsamples
        _, _, height, width = image.shape
        pad_height = (self.factor - (height - self.factor)) % self.factor
        pad_width = (self.factor - (width - self.factor)) % self.factor
        if pad_height != 0 or pad_width != 0:
            image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")

        return image, (height, width)

    def forward(self, image: Tensor) -> Tensor:
        image, (output_y, output_x) = self.pad_input_image(image)
        x = self.layer(image)
        x = self.final_conv(x)
        
        return x[:, :, :output_y, :output_x]
