import torch
from torch import functional as F
from torch import nn


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, activation=None):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    ]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(True))
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True, init_zero_weights=False,
         activation=None):
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers = [conv_layer]

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(True))
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)
