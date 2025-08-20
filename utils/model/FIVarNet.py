import math
from typing import List, NamedTuple, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from utils.model.layers.Feature import FeatureImage, FeatureEncoder, FeatureDecoder
from utils.model.layers.AttentionPE import AttentionPE
from utils.model.layers.NormStats import NormStats
from utils.model.layers.VarNetBlock import VarNetBlock
from utils.model.layers.AttentionFeatureVarNetBlock import AttentionFeatureVarNetBlock

from utils.model.utils.fftc import fft2c, ifft2c
from utils.model.utils.math import complex_abs
from utils.model.utils.coil_combine import rss, sens_expand, sens_reduce
from utils.model.utils.transforms import chan_complex_to_last_dim, complex_to_chan_dim
from utils.model.utils.transforms import center_crop

from utils.model.SensitivityModel import SensitivityModel
from utils.model.Unet import Unet2d, NormUnet


class FIVarNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        num_feature_cascades: int = 12,
        num_image_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 32,
        pools: int = 4,
        acceleration: int = 4,
        mask_center: bool = True,
        image_conv_cascades: Optional[List[int]] = None,
        kspace_mult_factor: float = 1e6,
        attn_stride: int = 0
    ):
        super().__init__()
        if image_conv_cascades is None:
            image_conv_cascades = [ind for ind in range(num_cascades) if ind % 3 == 0]

        self.image_conv_cascades = image_conv_cascades
        self.kspace_mult_factor = kspace_mult_factor
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.encoder = FeatureEncoder(in_chans=2, feature_chans=chans)
        self.decoder = FeatureDecoder(feature_chans=chans, out_chans=2)

        # Make attention application sequence
        attn_seq = list()
        if attn_stride == 0:
            attn_seq = [False for _ in range(num_feature_cascades)]
        else:
            for i in range(num_feature_cascades):
                attn_seq.append(i % attn_stride == 0)

        feature_cascades = []
        for ind in range(num_feature_cascades):
            use_image_conv = ind in self.image_conv_cascades
            use_attn = attn_seq[ind]

            feature_cascades.append(
                AttentionFeatureVarNetBlock(
                    encoder=self.encoder,
                    decoder=self.decoder,
                    acceleration=acceleration,
                    feature_processor=Unet2d(
                        in_chans=chans, out_chans=chans, num_pool_layers=pools
                    ),
                    attention_layer=AttentionPE(in_chans=chans) if use_attn else None,
                    use_extra_feature_conv=use_image_conv
                )
            )

        self.image_cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_image_cascades)]
        )

        self.decode_norm = nn.InstanceNorm2d(chans)
        self.feature_cascades = nn.ModuleList(feature_cascades)
        self.norm_fn = NormStats()
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def _decode_output(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            self.decode_norm(feature_image.features),
            means=feature_image.means,
            variances=feature_image.variances,
        )
        return sens_expand(chan_complex_to_last_dim(image), feature_image.sens_maps)

    def _encode_input(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        crop_size: Optional[Tuple[int, int]],
        num_low_frequencies: Optional[int],
    ) -> FeatureImage:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies)
        image = complex_to_chan_dim(sens_reduce(masked_kspace, sens_maps))
        # detect FLAIR 203
        if crop_size is not None and image.shape[-1] < crop_size[1]:
            crop_size = (image.shape[-1], image.shape[-1])
        means, variances = self.norm_fn(image)
        features = self.encoder(image, means=means, variances=variances)

        return FeatureImage(
            features=features,
            sens_maps=sens_maps,
            crop_size=crop_size,
            means=means,
            variances=variances,
            ref_kspace=masked_kspace,
            mask=mask,
        )


    def forward(
        self,
        masked_kspace: Tensor,
        mask: Tensor,
        num_low_frequencies: Optional[int] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        use_grad_ckpt: bool=True
    ) -> Tensor:
        masked_kspace = masked_kspace * self.kspace_mult_factor

        def apply_ckpt(fn, *args, use_grad_ckpt=True, **kwargs):
            if use_grad_ckpt:
                if kwargs:
                    fn = partial(fn, **kwargs)
                return checkpoint(fn, *args, use_reentrant=False)
            else:
                return fn(*args, **kwargs)

        feature_image = apply_ckpt(
            self._encode_input,
            masked_kspace,
            mask,
            crop_size,
            num_low_frequencies,
            use_grad_ckpt=use_grad_ckpt
        )

        for cascade in self.feature_cascades:
            feature_image = apply_ckpt(
                cascade,
                feature_image,
                use_grad_ckpt=use_grad_ckpt
                # use_grad_ckpt=False
            )

        kspace_pred = apply_ckpt(
            self._decode_output,
            feature_image,
            use_grad_ckpt=use_grad_ckpt
        )

        for cascade in self.image_cascades:
            kspace_pred = apply_ckpt(
                cascade,
                kspace_pred,
                feature_image.ref_kspace,
                mask,
                feature_image.sens_maps,
                use_grad_ckpt=use_grad_ckpt
                # use_grad_ckpt=False
            )

        kspace_pred = (
            kspace_pred / self.kspace_mult_factor
        )  
        
        result = rss(complex_abs(ifft2c(kspace_pred)), dim=1)
        cropped_result = center_crop(result, (384, 384)).contiguous()

        return cropped_result

