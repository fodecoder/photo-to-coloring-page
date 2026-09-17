"""U-Net generator architecture used by the optional Anime2Sketch engine.

Vendored (with only cosmetic changes -- type hints, docstring formatting)
from https://github.com/Mukosame/Anime2Sketch (``model.py``), used under
the MIT License, Copyright (c) 2021 Xiaoyu Xiang. See
``THIRD_PARTY_LICENSES.md`` at the repository root for the full license
text. The class/attribute names are kept identical to the upstream project
because they determine the ``state_dict`` keys that a community-provided
pretrained weights file must match.

This module only contains the network definition -- no pretrained weights
are bundled with this project.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn as nn

#: A normalization-layer constructor, e.g. `nn.BatchNorm2d` or
#: `functools.partial(nn.InstanceNorm2d, affine=False)`.
NormLayerFactory = Callable[..., nn.Module]


class UnetSkipConnectionBlock(nn.Module):
    """A single U-Net submodule with a skip connection around it."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: NormLayerFactory = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        model: list[nn.Module]
        down: list[nn.Module]
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up: list[nn.Module] = [uprelu, upconv, nn.Tanh()]
            assert submodule is not None
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            assert submodule is not None
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            result: torch.Tensor = self.model(x)
            return result
        # add skip connection
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """U-Net generator, built recursively from the innermost block outward."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer: NormLayerFactory = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.model(input)
        return result


def build_generator() -> UnetGenerator:
    """Construct the untrained Anime2Sketch "default" generator network.

    Matches the architecture the upstream project's pretrained ``netG.pth``
    weights were trained against: 3 input (RGB) channels, 1 output
    (grayscale sketch) channel, 8 downsampling stages, 64 base filters, and
    instance normalization.

    Returns
    -------
    UnetGenerator
        A freshly initialized (untrained) network, ready to have a matching
        ``state_dict`` loaded into it.
    """
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    return UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
