import yaml
import torch
from torch import nn

path = './config/config.yaml'

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_model(path):
    with open(path, 'r') as f:
        model = yaml.safe_load(f)

    first_half = model['YoloV2_first']
    second_half = model['YoloV2_second']
    fcn_layers_channels = model['FCN_layers_channels']

    return first_half, second_half, fcn_layers_channels

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

        self.model_first, self.model_second, self.fcn_layers_channels = load_model(path)
        self.in_channels = in_channels
        self.conv1 = self._create_block(self.model_first, self.in_channels)
        self.conv2 = self._create_block(self.model_second, 512)
        self.fcs = self._create_fcs(self.fcn_layers_channels)


    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        _, _, height, width = x.size()

        part1 = x[..., :height // 2, :width // 2] # (BATCH_SIZE, 512, 13, 13)
        print('part1 : ', part1.shape)
        part2 = x[..., :height // 2, width // 2:] # (BATCH_SIZE, 512, 13, 13)
        part3 = x[..., height // 2:, :width // 2] # (BATCH_SIZE, 512, 13, 13)
        part4 = x[..., height // 2:, width // 2:] # (BATCH_SIZE, 512, 13, 13)
        residual = torch.cat((part1, part2, part3, part4), dim=1) # (BATCH_SIZE, 2048, 13, 13)
        print(residual.shape)

        x = torch.cat((self.conv2(x), residual), dim=1)
        print(x.shape)
        x = self.fcs(x)
        print(x.shape)
        return

    def _create_block(self, model, in_channels):
        layers = []

        for x in model:
            if isinstance(x, list):
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            elif isinstance(x, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)


    def _create_fcs(self, in_channels, num_of_anchor=5, num_of_classes=3):
        layers = []
        out_channels = num_of_anchor * (num_of_classes + 5)

        layers = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        final_layers = nn.Sequential(layers)
        final_out = final_layers.permute(0, 2, 3, 1).contiguous()

        return final_out.view(final_out(0), final_out(1), final_out(2), num_of_anchor, num_of_classes + 5)




x = torch.randn(4, 3, 416, 416)
model = YoloV2()
print(model(x))