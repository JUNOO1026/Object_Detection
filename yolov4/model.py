import yaml
import torch
from torch import nn
from torchinfo import summary

path = 'yolov4/config/config.yaml'


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    architecture = config['model_architecture']
    return architecture


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeat):
        super().__init__()
        self.conv1 = CNNBlock(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = CNNBlock(in_channels, out_channels // 2, kernel_size=1)
        self.residual = ResidualBlock(out_channels // 2, num_repeat)
        self.conv3 = CNNBlock(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(self.conv2(x))
        output = torch.cat([x1, x2], dim=1)
        return self.conv3(output)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeat):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                CNNBlock(in_channels // 2, in_channels, kernel_size=3, padding=1)
            ) for _ in range(num_repeat)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class SPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv = CNNBlock(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv(x)
        return x


class PANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.conv1 = CNNBlock(1024, 512, kernel_size=1)
        self.conv2 = nn.Sequential(
            CNNBlock(1024, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=1, stride=1, padding=0),
            CNNBlock(1024, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 1024, kernel_size=1, stride=1, padding=0),
            CNNBlock(1024, 512, kernel_size=3, stride=1 ,padding=1),
        )
        
        
       
        self.conv3 = CNNBlock(512, 256, kernel_size=1)
        self.conv4 = nn.Sequential(
            CNNBlock(512, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 256, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, spp_out, route_connections):
        output_big = spp_out
        x = self.conv1(spp_out)
        x = self.upsample(x) # (1, 512, 26, 26)
        x = torch.cat([x, route_connections[-1]], dim=1) # (1, 1024, 26, 26)
        output_medium = self.conv2(x) # (1, 512, 26, 26)

        # print('out_big : ', output_big.shape)
        x = self.conv3(output_medium) # (1, 256, 26, 26)
        x = self.upsample(x)       # (1, 256, 52, 52)
        x = torch.cat([x, route_connections[-2]], dim=1) # (1, 512, 52, 52)
        output_small = self.conv4(x)  # (1, 256, 52, 52)

        print("output_small.shape : ", output_small.shape)
        print("output_medium.shape : ", output_medium.shape)
        print("output_big.shape : ", output_big.shape)

        return output_small, output_medium, output_big


class CSPDarknet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.architecture = load_config(path)
        self.layers = self._create_block(self.architecture)

    def forward(self, x):
        route_connections = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in [5, 7]:  
                route_connections.append(x)

        return x, route_connections

    def _create_block(self, arch):
        in_channels = self.in_channels
        layers = []
        for x in arch:
            if isinstance(x, list) and len(x) == 4:
                layers.append(CNNBlock(in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3]))
                in_channels = x[0]
            elif isinstance(x, list) and len(x) == 2:
                layers.append(CSPBlock(in_channels, in_channels, num_repeat=x[1]))
        return nn.Sequential(*layers)


class YoloV4Head(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.small_output = nn.Conv2d(256, (num_classes + 5) * len(anchors), kernel_size=1)
        self.medium_output = nn.Conv2d(512, (num_classes + 5) * len(anchors), kernel_size=1)
        self.big_output = nn.Conv2d(1024, (num_classes + 5) * len(anchors), kernel_size=1)

    def forward(self, small, medium, big):
        small_out = self.small_output(small)
        medium_out = self.medium_output(medium)
        big_out = self.big_output(big)
        return [small_out, medium_out, big_out]


class YoloV4(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.backbone = CSPDarknet(in_channels)
        self.spp = SPP(1024)
        self.panet = PANet()
        self.head = YoloV4Head(num_classes, anchors=[(10, 13), (16, 30), (33, 23)])

    def forward(self, x):
        x, route_connections = self.backbone(x)
        x = self.spp(x)
        small, medium, big = self.panet(x, route_connections)
        outputs = self.head(small, medium, big)
        return outputs

# Test
model = YoloV4()
x = torch.randn(1, 3, 416, 416)
print(model(x)[0].shape)
print(model(x)[1].shape)
print(model(x)[2].shape)

# summary(model, (1, 3, 416, 416))
