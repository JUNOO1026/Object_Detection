import os
import yaml
import torch
from torch import nn, optim


path = 'yolov4/config/config.yaml'

# print(os.getcwd())
def load_model(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config['model_architecture']

# print(load_model(path))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

Activations = {
    'Mish': nn.Mish(),
    "Swish": Swish(),
    'Leaky': nn.LeakyReLU(0.1)

}

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = Activations['Swish'], **kwargs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeat=1, **kwargs):
        super().__init__()

        self.conv1 = CNNBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = CNNBlock(in_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.residual = ResidualBlock(out_channels // 2, num_repeat)
        self.conv3 = CNNBlock(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.act = Swish()


    def forward(self, x):
        l_layer = self.act(self.conv1(x))
        print(l_layer.shape)
        r_layer = self.act(self.residual(self.conv2(x)))
        cat_layer = torch.cat([l_layer, r_layer], dim=1)

        output = self.conv3(cat_layer)

        return output



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_repeat=1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                CNNBlock(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
                CNNBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1)
            )
            for _ in range(num_repeat)
        ])

    
    def forward(self, x):
        for layer in self.layers:
            print("x: ", x.shape)
            print('layer(x) : ', layer(x).shape)
            x = x + layer(x) # element-wise

        return x

class CSPDarknet53(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        
        self.in_channels = in_channels
        self.architecture = load_model(path)
        self.layer = self._create_conv(self.architecture)

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layer(x)
        x = torch.mean(x, dim=[2, 3])
        
        return self.classifier(x)
        

    def _create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, list) and len(x) == 4:
                layers.append(CNNBlock(in_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3]))
                in_channels = x[0]

            elif isinstance(x, list) and len(x) == 2:
                layers.append(CSPBlock(in_channels, in_channels, num_repeat=x[1]))
                in_channels = in_channels

        return nn.Sequential(*layers)
        
model = CSPDarknet53(num_classes=80)
x = torch.randn(1, 3, 416, 416)
output = model(x)
print(output.shape) 
        
# a = torch.randn(2, 64, 416, 416)

# model = ResidualBlock(64, 2)
# print(model(a).shape)

# for layer in load_model(path):
#     print(layer)
