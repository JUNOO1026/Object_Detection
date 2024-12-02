import yaml
import torch

from torch import nn

def load_model(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    backbone = config['yolov8_backbone']
    head = config['yolov8_head']

    return backbone, head 

class pointwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()


    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNNBlock(nn.Module):
    '''
        CNNBlock is Original Convolution Layer. 

        params : in_channels, out_channels

        Output shape : (BatchSize, channel, height, width)
    
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()


    def forward(self, x) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
    
    
class CNNBlock_nonact(nn.Module):
    '''
        CNNBlock_nonact doesn't have activation function.
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        return self.bn(self.conv(x))
    
    
class Bottleneck(nn.Module):
    '''
        This class is Yolov8 Bottleneck.

        params : in_channels, out_channels, num_repeat, shortcut=(True or False)
    
    '''
    def __init__(self, in_channels, shortcut=None, **kwargs):
        super().__init__()

        self.module  = nn.Sequential(
            CNNBlock(in_channels, in_channels, **kwargs),
            CNNBlock(in_channels, in_channels, **kwargs), 
        )
        
        self.shortcut = shortcut
    
    def forward(self, x) -> torch.Tensor:
        residual = x

        return residual + self.module(x) if self.shortcut else self.module(x)


class c2f(nn.Module):
    
    def __init__(self, in_channels, out_channels, num_repeat, shortcut=None, **kwargs):
        super().__init__()

        self.f_conv_1x1 = pointwiseBlock(in_channels, in_channels // 2)
        self.bottleneck = nn.ModuleList(
            Bottleneck(in_channels // 2, shortcut, **kwargs) for _ in range(num_repeat)
        )

        self.l_shortcut_conv_1x1 = pointwiseBlock(in_channels // 2 * (num_repeat + 1), out_channels)
        self.l_conv_1x1 = pointwiseBlock(in_channels // 2, out_channels)

        
        self.shortcut = shortcut
        
    def forward(self, x) -> torch.Tensor:
        bottleneck = [self.f_conv_1x1(x)]
        
        if self.shortcut:
            for module in self.bottleneck:
                bottleneck.append(module(bottleneck[-1]))
            output = self.l_shortcut_conv_1x1(torch.cat(bottleneck, dim=1))

        else:
            for module in self.bottleneck:
                bottleneck[-1] = module(bottleneck[-1])
            output = self.l_conv_1x1(bottleneck[-1])

        return output
    

class SPPF(nn.Module):
    '''
        This class is Spatial Pyramid Pooling Fast Module.
        Use Maxpool2D with kernel_size = 5.

    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.ModuleList(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(3)
        )

        self.l_conv1x1 = CNNBlock(in_channels * 3, out_channels, 
                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        output = []

        for spp in self.maxpool:
            output.append(spp(x))

        outputs = self.l_conv1x1(torch.cat(output, dim=1))

        return outputs
    

class upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.upsample(x)
    
class concat(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def forward(self, x):
        return torch.cat([self.backbone, x], dim=1)
    
    
        
