import torch
from torch import nn

class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)

        output = torch.cat([x1, x2, x3], dim=1)

        return output

class PANet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.up_conv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

    def forward(self, h_f, l_f):
        h_f = self.up_conv1(self.conv1(h_f))
        return torch.cat([h_f, l_f], dim=1)




