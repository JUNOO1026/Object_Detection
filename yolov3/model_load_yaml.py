import os
from platform import architecture

import yaml
import torch

from torch import nn


path = './config/config.yaml'

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    architecture = config['model_architecture']

    return architecture

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.bn_act = bn_act

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.bn_act:
            x = self.act(self.bn(self.conv(x)))
        else:
            x = self.conv(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeats, use_residual=True, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        self.num_repeats = num_repeats

        for _ in range(self.num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0),
                    CNNBlock(in_channels//2, in_channels, kernel_size=3, stride=1, padding=1),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class Yolov3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.architecture = load_config(path) # load     model
        self.layer = self._create_block(self.architecture)


    def forward(self, x):


    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layers in architecture:
            if isinstance(layers, list) and len(layers) == 3:
                layers.append(CNNBlock(in_channels,
                                       layers[0],
                                       kernel_size=layers[1],
                                       stride=layers[2],
                                       padding=1 if layers[1]==3 else 0
                                       ))
                in_channels = layers[0]

            elif isinstance(layers, list) and len(layers) == 2:
                layers.append(ResidualBlock(in_channels, num_repeats=layers[1], use_residual=True))

            elif isinstance(layers, str) and layers == 'S':


            elif isinstance(layers, str) and layers == 'U':
                layers.append(nn.Upsample(scale_factor=2))



        return layers


model = Yolov3(in_channels=3, num_classes=80)
a = torch.randn(2, 3, 416, 416)




