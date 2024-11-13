import os
import yaml
import torch
from torch import nn
from torchinfo import summary

# print(os.getcwd())

path = './landmark/config/mobilenetV1.yaml'


def load_model(path):
    with  open(path, 'r') as f:
        config = yaml.safe_load(f)

    model = config['model_architecture']

    return model


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.DW_conv = CNNBlock(in_channels, 
                                out_channels, 
                                kernel_size=3, 
                                groups=in_channels,
                                padding=1,
        )
        
    def forward(self, x):
        return self.DW_conv(x)
    

class PointWiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.PW_conv = CNNBlock(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
        )

    def forward(self, x):
        return self.PW_conv(x)
    

class MobileNetV1(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        self.architecture = load_model(path)
        self.in_channels = 32
        self.layers = self._create_block(self.architecture)
 
        
        self.f_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.ReLU()
        self.l_conv = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.act(self.bn(self.f_conv(x)))
        x = self.layers(x)
        x = self.l_conv(x)
        x = self.pool(x)
        print("x.shape : ", x.shape)
        print(x.size(0))
        x = x.view(x.size(0), -1)
        print("x.shape : ", x.shape)
        x = self.fc(x)

        return x

    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, list) and len(x) == 2:
                layer1 = x[0]
                layer2 = x[1]
                layers += [
                    DepthWiseBlock(in_channels, layer1[0], kernel_size=layer1[1], stride=layer1[2]),
                    PointWiseBlock(layer1[0], layer2[0], kernel_size=layer2[1], stride=layer2[2]),
                ]
                in_channels = layer2[0]

            elif isinstance(x, list) and len(x) == 3:
                num_repeat = x[0]
                layer1 = x[1]
                layer2 = x[2]
                
                for _ in range(num_repeat):
                    layers += [
                        DepthWiseBlock(in_channels, layer1[0], kernel_size=layer1[1], stride=layer1[2]),
                        PointWiseBlock(layer1[0], layer2[0], kernel_size=layer2[1], stride=layer2[2])
                    ]
                    in_channels = layer2[0]


        return nn.Sequential(*layers)
    

a = torch.randn(1, 3, 224, 224)

model = MobileNetV1(in_channels=3, num_classes=1000)
print(model(a).shape)