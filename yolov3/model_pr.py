import torch
from torch import nn
from triton.language.extra.cuda import num_threads

# filters, kernel_size, stride
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.bn_act = bn_act
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)


    def forward(self, x):
        if self.bn_act:
            x = self.act(self.bn(self.conv(x)))

        else:
            x = self.conv(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        print("Residual_block")
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x
class ScalePredictionBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels*2, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes


    def forward(self, x):
        return (self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2))

class YoloV3(nn.Module):
    def __init__(self, in_channels, num_classes=80, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer = self._create_block()

    def forward(self, x):
        outputs = []
        route_connections = []

        for layers in self.layer:
            if isinstance(layers, ScalePredictionBlock):
                outputs.append(layers)
                continue

            x = layers(x)

            elif isinstance(layers, ResidualBlock) and layers.num_repeats == 8:



    def _create_block(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for x in config:
            if isinstance(x, tuple):
                out_channels, kernel_size, stride = x

                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size==3 else 0)
                )
                in_channels = out_channels

            elif isinstance(x, list):
                repeats = x[1]
                for _ in range(repeats):
                    layers.append(
                        ResidualBlock(in_channels, num_repeat=repeats)
                    )

            elif isinstance(x, str) and x == 'S':
                layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeat=1),
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ScalePredictionBlock(in_channels // 2, num_classes=self.num_classes),
                ]

                in_channels = in_channels // 2

            elif isinstance(x, str) and x == 'U':
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels * 3


        return layers



