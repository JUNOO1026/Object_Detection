import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_repeat, projection, **kwargs):
#         super().__init__()
        
#         self.layers = nn.ModuleList([
#             nn.Sequential(
#                 CNNBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             ) 
#             for _ in range(num_repeat)
#         ])
#         self.projection = projection
#         self.act = nn.ReLU()

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x) # element-wise

#             if self.projection is not None:
#                 shorcut = self.projection(x)

#             else:
#                 shortcut = x
            
#             output = self.act(x + shortcut)

#         return output
    

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, num_repeat, projection, **kwargs):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                CNNBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
            for _ in range(num_repeat)
        ])
        
        self.projection = projection
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

            if self.projection is not None:
                shortcut = self.projection(x)
            else:
                shortcut = x

        output = self.act(x + shortcut)

        return output
    

    class ResNet(nn.Module):
        def __init__(self, in_channels, block, **kwargs):
            super().__init__()

            self.block = block # [3, 4, 6, 3]

            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.act1 = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            
            self.in_channels = 64

            self.module = self._create_block(self.block)


        def forward(self, x):
            pass

        def _create_block(self, block):
            in_channels = self.in_channels
            layers = []







    
