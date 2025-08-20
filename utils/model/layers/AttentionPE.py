import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionPE(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        self.in_chans = in_chans

        self.norm = nn.InstanceNorm2d(in_chans)
        self.q = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_chans, in_chans, kernel_size=1, stride=1, padding=0
        )
        self.dilated_conv = nn.Conv2d(
            in_chans, in_chans, kernel_size=3, stride=1, padding=2, dilation=2
        )

    def reshape_to_blocks(self, x: Tensor, accel: int) -> Tensor:
        chans = x.shape[1]
        pad_total = (accel - (x.shape[3] - accel)) % accel
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        x = F.pad(x, (pad_left, pad_right, 0, 0), "reflect")
        return (
            torch.stack(x.chunk(chunks=accel, dim=3), dim=-1)
            .view(chans, -1, accel)
            .permute(1, 0, 2)
            .contiguous()
        )

    def reshape_from_blocks(
        self, x: Tensor, image_size: Tuple[int, int], accel: int
    ) -> Tensor:
        chans = x.shape[1]
        num_freq, num_phase = image_size
        x = (
            x.permute(1, 0, 2)
            .reshape(1, chans, num_freq, -1, accel)
            .permute(0, 1, 2, 4, 3)
            .reshape(1, chans, num_freq, -1)
        )
        padded_phase = x.shape[3]
        pad_total = padded_phase - num_phase
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        return x[:, :, :, pad_left : padded_phase - pad_right]

    def get_positional_encodings(
        self, seq_len: int, embed_dim: int, device: str
    ) -> Tensor:
        freqs = torch.tensor(
            [1 / (10000 ** (2 * (i // 2) / embed_dim)) for i in range(embed_dim)],
            device=device,
        )
        freqs = freqs.unsqueeze(0)
        positions = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        scaled = positions * freqs
        sin_encodings = torch.sin(scaled)
        cos_encodings = torch.cos(scaled)
        encodings = torch.cat([sin_encodings, cos_encodings], dim=1)[:, :embed_dim]
        return encodings

    def forward(self, x: Tensor, accel: int) -> Tensor:
        im_size = (x.shape[2], x.shape[3])
        h_ = x
        h_ = self.norm(h_)

        pos_enc = self.get_positional_encodings(x.shape[2], x.shape[3], h_.device.type)

        h_ = h_ + pos_enc

        q = self.dilated_conv(self.q(h_))
        k = self.dilated_conv(self.k(h_))
        v = self.dilated_conv(self.v(h_))

        # compute attention
        c = q.shape[1]
        q = self.reshape_to_blocks(q, accel)
        k = self.reshape_to_blocks(k, accel)
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = self.reshape_to_blocks(v, accel)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = self.reshape_from_blocks(h_, im_size, accel)

        h_ = self.proj_out(h_)

        return x + h_
