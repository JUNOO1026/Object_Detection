import torch
from torch import nn, optim


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

Activations = {
    'Mish': nn.Mish(),
    "Swish": Swish(),
    'Leaky': nn.LeakyReLU(0.1)

}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = Activations['Swish'], **kwargs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

# class CSPBlock(nn.Module):
#     def __init__(self, in_channels, )
#         super().__init__()
#         self.conv1 = CNNBlock(in_channels, in_channels * 2, kernel_size=1, stride=1)
#         self.

# class CatConv(nn.Module):
#     def __init__(self, layer1, layer2):
        
    

