import torch
import torch.nn as nn
import numpy as np

from typing import List
from utils.constants import LATENT_SPACE_DIM


def upsample_block(in_channels: int,
                   out_channels: int,
                   normalize: bool = True,
                   activation: nn.Module = None) -> List[nn.Module]:
    layers = []

    # upsample
    layers.append(nn.ConvTranspose2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=4,
                                     stride=2,
                                     padding=1,
                                     bias=False))

    # normalization layer
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))

    # activation function
    layers.append(nn.ReLU() if activation is None else activation)

    return layers


def downsample_block(in_channels: int,
                     out_channels: int,
                     n_power_iterations: int,
                     normalize: bool = True,
                     activation: nn.Module = None,
                     padding: int = 1) -> List[nn.Module]:
    layers = []

    # downsample
    conv = nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=4,
                     stride=2,
                     padding=padding)

    # normalization layer
    if normalize:
        conv = nn.utils.spectral_norm(conv,
                                      n_power_iterations=n_power_iterations)
    layers.append(conv)

    # activation function
    layers.append(nn.LeakyReLU(0.2) if activation is None else activation)

    return layers


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [1024, 512, 256, 128, 3]
        self._input_shape = (channels[0], 4, 4)

        self._projector = nn.Linear(LATENT_SPACE_DIM,
                                    np.prod(self._input_shape))

        self._net = nn.Sequential(
            *upsample_block(channels[0], channels[1]),
            *upsample_block(channels[1], channels[2]),
            *upsample_block(channels[2], channels[3]),
            *upsample_block(channels[3], channels[4], False, nn.Tanh())
        )

    def forward(self, x):
        x_projected = self._projector(x)
        x_projected = x_projected.view(x.size(0), *self._input_shape)
        output = self._net(x_projected)

        return output


class Discriminator(nn.Module):
    def __init__(self, n_power_iterations):
        super().__init__()

        channels = [3, 128, 256, 512, 1024, 1]
        self._net = nn.Sequential(
            *downsample_block(channels[0], channels[1], n_power_iterations),
            *downsample_block(channels[1], channels[2], n_power_iterations),
            *downsample_block(channels[2], channels[3], n_power_iterations),
            *downsample_block(channels[3], channels[4], n_power_iterations),
            *downsample_block(channels[4], channels[5], n_power_iterations,
                              False, nn.Sigmoid(), 0)
        )

    def forward(self, x):
        output = self._net(x)

        return output