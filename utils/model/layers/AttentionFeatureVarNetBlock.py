import math
from typing import List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.model.layers.Feature import FeatureImage, FeatureEncoder, FeatureDecoder
from utils.model.Unet import Unet2d
from utils.model.layers.AttentionPE import AttentionPE

from utils.model.utils.coil_combine import sens_expand, sens_reduce
from utils.model.utils.transforms import chan_complex_to_last_dim, complex_to_chan_dim
from utils.model.utils.transforms import image_crop, image_uncrop

class AttentionFeatureVarNetBlock(nn.Module):
    def __init__(
        self,
        encoder: FeatureEncoder,
        decoder: FeatureDecoder,
        acceleration: int,
        feature_processor: Unet2d,
        attention_layer: None | AttentionPE,
        use_extra_feature_conv: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_processor = feature_processor
        self.attention_layer = attention_layer
        self.use_image_conv = use_extra_feature_conv
        self.dc_weight = nn.Parameter(torch.ones(1))
        feature_chans = self.encoder.feature_chans
        self.acceleration = acceleration

        self.input_norm = nn.InstanceNorm2d(feature_chans)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_extra_feature_conv:
            self.output_norm = nn.InstanceNorm2d(feature_chans)
            self.output_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    in_channels=feature_chans,
                    out_channels=feature_chans,
                    kernel_size=5,
                    padding=2,
                    bias=False,
                ),
                nn.InstanceNorm2d(feature_chans),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )

        self.zero: Tensor
        self.register_buffer("zero", torch.zeros(1, 1, 1, 1, 1))

    def encode_from_kspace(self, kspace: Tensor, feature_image: FeatureImage) -> Tensor:
        image = complex_to_chan_dim(sens_reduce(kspace, feature_image.sens_maps))

        return self.encoder(
            image, means=feature_image.means, variances=feature_image.variances
        )

    def decode_to_kspace(self, feature_image: FeatureImage) -> Tensor:
        image = self.decoder(
            feature_image.features,
            means=feature_image.means,
            variances=feature_image.variances,
        )

        return sens_expand(chan_complex_to_last_dim(image), feature_image.sens_maps)

    def compute_dc_term(self, feature_image: FeatureImage) -> Tensor:
        est_kspace = self.decode_to_kspace(feature_image)

        return self.dc_weight * self.encode_from_kspace(
            torch.where(
                feature_image.mask.to(torch.bool), est_kspace - feature_image.ref_kspace, self.zero
            ),
            feature_image,
        )

    def apply_model_with_crop(self, feature_image: FeatureImage) -> Tensor:
        if feature_image.crop_size is not None:
            features = image_uncrop(
                self.feature_processor(
                    image_crop(feature_image.features, feature_image.crop_size)
                ),
                feature_image.features.clone(),
            )
        else:
            features = self.feature_processor(feature_image.features)

        return features

    def forward(self, feature_image: FeatureImage) -> FeatureImage:

        feature_image = feature_image._replace(
            features=self.input_norm(feature_image.features)
        )

        # Data consistency
        new_features = feature_image.features - self.compute_dc_term(feature_image)
        """
        new_features_np = feature_image.features.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'new_features_before_{timestamp}.mat'
        savemat(file_name, {'new_features_before': new_features_np})

        new_ref_kspace = feature_image.ref_kspace.cpu().numpy()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'kspace_{timestamp}.mat'
        savemat(file_name, {'kspace_': new_ref_kspace})
        """

        # Apply attention
        if self.attention_layer:
            feature_image = feature_image._replace(
                features=self.attention_layer(feature_image.features, self.acceleration)
            )
        new_features = new_features - self.apply_model_with_crop(feature_image)

        if self.use_image_conv:
            new_features = self.output_norm(new_features)
            new_features = new_features + self.output_conv(new_features)

        return feature_image._replace(features=new_features)

