import logging
import torch
from torch import nn

class CNNBlock(nn.Module):
    '''
        This class Original CNNblock.

        Activation function is ReLU() Not ReLU6.

        ReLU6 will behave the same way as ReLU if the input does not exceed 1.
    
    
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        '''
            forward pass

            Returns : output tensor of shape : (-1, channel, width, height)
        
        '''

        return self.act(self.bn(self.conv(x)))

class PointWiseBlock(nn.Module):
    '''
        This class is 1 * 1 Convolution Block.
        1*1 convolution reduces the amount of computation 
        and helps you learn various characteristics by growing channels.
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = CNNBlock(in_channels, out_channels, 
                              kernel_size=1, stride=1, padding=0)
        
    def forward(self, x) -> torch.Tensor:
        return self.layer(x)
    
class DepthWiseBlock(nn.Module):
    '''
        This class is Depthwise Convolution Block.

        Depthwise Convolution is independent and has the advantage 
        of enriching the expression of each channel. 
        It also has low computational power and easy real-time processing.
    '''
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = CNNBlock(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, groups=in_channels)
        
    def forward(self, x) -> torch.Tensor:
        return self.layers(x)
    

class InvertedResidualBlock(nn.Module):
    '''
        This class Inverted Residual Block.

        It is consist of [PointwiseBlock + DepthWiseBlock + PointwiseBlock].
    '''
    expansion = 6

    def __init__(self, in_channels, out_channels, stride, use_residual):
        super().__init__()

        self.block = nn.Sequential(
            PointWiseBlock(in_channels, in_channels * InvertedResidualBlock.expansion),
            DepthWiseBlock(in_channels * InvertedResidualBlock.expansion, in_channels * InvertedResidualBlock.expansion, stride),
            nn.Conv2d(in_channels * InvertedResidualBlock.expansion, out_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

        self.use_residual = use_residual

    def forward(self, x) -> torch.Tensor:
        shortcut = self.block(x)

        if self.use_residual:
            output = x + shortcut
        else:
            output = shortcut
        
        return output

    @classmethod
    def change_exp(cls, exp) -> int:
        cls.exp = exp

        if cls.exp == cls.expansion:
            print('Not change InvertedResidualBlock expansion. please check again')

        else:
            pass

class RetinaFace(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers1 = CNNBlock(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.layers2 = CNNBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.block1 = self._create_block(64, 2, 64, 5, 2)
        self.block2 = self._create_block(64, 2, 128, 1, 2)
        self.block3 = self._create_block(128, 4, 128, 6, 2)
        self.block4 = self._create_block(128, 2, 256, 1, 2)

    def forward(self, x) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.layers1(x)
        x = self.layers2(x)
        c3 = self.block1(x)
        c4 = self.block2(c3)
        c5 = self.block3(c4)
        x = self.block4(c5)
        return x, [c3, c4, c5]
    
    def _create_block(self, in_channels, expansion, out_channels, num_repeat, stride):
        layers = []
        use_residual = None

        InvertedResidualBlock.change_exp(expansion)

        for idx, _ in enumerate(range(num_repeat)):
            if idx == 0:
                stride = stride
                use_residual = False
            else:
                stride = 1
                use_residual = True

            layers.append(InvertedResidualBlock(in_channels, out_channels, stride, use_residual))

        return nn.Sequential(*layers)
    

a = torch.randn(1, 3, 640, 640)
model = RetinaFace()

print(model(a)[0].shape)
print(model(a)[1][0].shape)
for i in range(3):
    print(model(a)[1][i].shape)