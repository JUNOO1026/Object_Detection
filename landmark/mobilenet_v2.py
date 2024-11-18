import yaml
import torch
from torch import nn

path = './landmark/config/mobilenetV2.yaml'

def load_model(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    model = config['model_architecture']

    return model

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.layers = CNNBlock(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               groups=in_channels, 
                               bias=False,
                               **kwargs)
        

    def forward(self, x):
        return self.layers(x)
    
class PointwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear=None, **kwargs):
        super().__init__()
        
        self.R_conv = CNNBlock(in_channels, out_channels,kernel_size=1, bias=False, **kwargs)
        self.L_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.nonlinear = nonlinear

    def forward(self, x):
        if self.nonlinear:
            return self.R_conv(x)
        else:
            return self.L_conv(x)
        
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, repeat, stride):
        super().__init__()

        layers = []
        
        self.residual_check = (stride == 1 and in_channels == out_channels)

        for i in range(repeat):
            current_stride = stride if i == 0 else 1
            current_padding = 0 if i == 0 else 1 
            layers.append(
                nn.Sequential(
                    PointwiseBlock(in_channels, in_channels * expansion, nonlinear=True),
                    DepthwiseBlock(in_channels * expansion, in_channels * expansion, 
                                   stride=current_stride, 
                                   padding=current_padding),
                    PointwiseBlock(in_channels * expansion, out_channels, nonlinear=False)
                )
            )
            in_channels = out_channels

        self.bottleneck = nn.ModuleList(layers)
 
        if not self.residual_check:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.residual = None

    def forward(self, x):
        for idx, layer in enumerate(self.bottleneck):
            out = layer(x)

            if idx == 0 and self.residual:
                x = self.residual(out)
                print("residual : ",x.shape)
                x = x + out
                
            else:
                x = out
        return x
    
class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, **kwargs):
        super().__init__()

        self.in_channels = 32
        self.conv1 = CNNBlock(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.architecture = load_model(path)
        self.layer = self._create_block(self.architecture)

        self.conv2 = PointwiseBlock(320, 1280, nonlinear=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l_conv = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print('이것도 안나와?')
        print(x.shape)
        x = self.layer(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.l_conv(x)
        return x

    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for config in architecture:
            t, c, n, s = config
            layers.append(Bottleneck(in_channels, c, t, n, s))
            in_channels = c

        return nn.Sequential(*layers)


if __name__ == "__main__": 
    a = torch.randn(1, 3, 224, 224)
    model = MobileNetV2()
    print(model(a).shape)