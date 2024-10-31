import os
import yaml
import torch

from torch import nn
print(os.getcwd())

config_path = 'config/yolov1.yaml'


def load_architecture_config(path):
    with open(path, 'r') as f:
        architecture = yaml.safe_load(f)

    return architecture

print(load_architecture_config(config_path))

class CNNBlock(nn.Module):
    '''
        This class original CNNBlock.
        (in_channels, out_channels, **kwargs)
        bias is False
    '''
    def __init__(self, in_channels, out_channels, bias=False, *args, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    '''
        This class Yolov1 of Object Detection.
        Model is Darknet Architecture.
        Model Architecture take a yaml config.
    '''
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()

        self.architecture = load_architecture_config(config_path)
        self.in_channels = in_channels
        self.darknet = self._create_block(self.architecture)
        self.fcs = self._create_fcs(**kwargs)


    def forward(self, x):
        x = self.darknet(x)
        print('x.shape : ', x.shape)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture['architecture_config']:
            print('come in?')
            if isinstance(x, list) and len(x) == 4:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif isinstance(x, list) and len(x) == 3:
                conv1 = x[0]
                conv2 = x[1]
                repeat = x[2]
                for i in range(repeat):
                    layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
                             nn.Linear(S * S * 1024, 496),
                             nn.Dropout(0.2),
                             nn.LeakyReLU(0.1),
                             nn.Linear(496, S * S * (B * 5 + C)))

model = Yolov1(split_size=7, num_boxes=2, num_classes=3)
x = torch.randn(4, 3, 448, 448)
print(model(x).shape)