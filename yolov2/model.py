import yaml
import torch
from torch import nn

path = './config/config.yaml'

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))


class YoloV2(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()

        self.model_architecture = load_config(path)
        self.in_channels = in_channels
        self.darknet19 = self._create_block(self.model_architecture)
        self.fcs = self._create_fcs


    def forward(self, x):
        x = self.darknet19(x)
        return self.fcs(torch.flatten(x), start_dim=1)

    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, list):
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]


        return nn.Sequential(*layers)

    def _create_fcs(self, split_size=7, ):
