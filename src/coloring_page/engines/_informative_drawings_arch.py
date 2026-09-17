"""Generator architecture used by the optional Informative Drawings engine.

Vendored (with only cosmetic changes -- type hints, docstring formatting)
from https://github.com/carolineec/informative-drawings (``model.py``, the
``Generator``/``ResidualBlock`` classes only -- the file's other classes are
for training and geometry prediction and aren't needed for inference), used
under the MIT License, Copyright (c) 2022 Caroline Chan. See
``THIRD_PARTY_LICENSES.md`` at the repository root for the full license text
and for why weights are downloaded from this project's official source
rather than through a third-party mirror. The class/attribute names are kept
identical to the upstream project because they determine the ``state_dict``
keys that a community-provided pretrained weights file must match.

This module only contains the network definition -- no pretrained weights
are bundled with this project.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_norm_layer = nn.InstanceNorm2d


class ResidualBlock(nn.Module):
    """A single reflection-padded residual block."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            _norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            _norm_layer(in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = x + self.conv_block(x)
        return result


class Generator(nn.Module):
    """Encoder/residual-block/decoder line-drawing generator.

    Unlike Anime2Sketch's U-Net (which carries skip connections between
    matching encoder/decoder stages), this is a plain ResNet-style
    generator: a downsampling encoder, a stack of residual blocks, and
    an upsampling decoder, with no skip connections -- the architecture
    used by the Informative Drawings paper (Chan, Durand, Isola, CVPR
    2022).
    """

    def __init__(
        self, input_nc: int, output_nc: int, n_residual_blocks: int = 3, sigmoid: bool = True
    ) -> None:
        super().__init__()

        initial = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            _norm_layer(64),
            nn.ReLU(inplace=True),
        ]
        self.model0 = nn.Sequential(*initial)

        downsampling: list[nn.Module] = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            downsampling += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                _norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*downsampling)

        residual_blocks = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*residual_blocks)

        upsampling: list[nn.Module] = []
        out_features = in_features // 2
        for _ in range(2):
            upsampling += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                _norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*upsampling)

        final: list[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            final.append(nn.Sigmoid())
        self.model4 = nn.Sequential(*final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        result: torch.Tensor = self.model4(out)
        return result


def build_generator(n_residual_blocks: int = 3) -> Generator:
    """Construct the untrained Informative Drawings line-art generator.

    Matches the architecture the upstream project's pretrained
    checkpoints (``contour_style``, ``anime_style``, ``opensketch_style``
    -- all share this same generator shape) were trained against: 3
    input (RGB) channels, 1 output (grayscale line-art) channel.

    Parameters
    ----------
    n_residual_blocks : int, optional
        Number of residual blocks in the generator's bottleneck, by
        default 3, matching the upstream project's default checkpoint
        configuration.

    Returns
    -------
    Generator
        A freshly initialized (untrained) network, ready to have a
        matching ``state_dict`` loaded into it.
    """
    return Generator(3, 1, n_residual_blocks=n_residual_blocks, sigmoid=True)
