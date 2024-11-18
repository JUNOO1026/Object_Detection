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
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = CNNBlock(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               groups=in_channels, 
                               bias=False,
                               padding=1 if stride == 1 else 0)
        

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
        
# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, expansion, repeat, stride):
#         super().__init__()

#         layers  = []
#         self.residual_check = (stride == 1 and in_channels == out_channels)
#         for i in range(repeat):
#             stride = stride if i == 0 else 1
#             layers.append(
#                 nn.Sequential(
#                     PointwiseBlock(in_channels, in_channels*expansion, nonlinear=True),
#                     DepthwiseBlock(in_channels*expansion, in_channels*expansion, stride),
#                     PointwiseBlock(in_channels*expansion, out_channels, nonlinear=False)
#                 ) 
#             )
#             in_channels = out_channels
        
#         self.bottleneck = nn.ModuleList(layers)
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         if not self.residual_check:
#             self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         else:
#             self.residual = None

#     def forward(self, x):    
#         for idx, layer in enumerate(self.bottleneck):
#             out = layer(x)
        
#             if idx == 0 and self.residual:
#                 x = self.residual(x)
#                 x = x + out
#             else:
#                 x = out
#         return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, repeat, stride):
        super().__init__()

        layers = []
        self.initial_in_channels = in_channels  # 초기 입력 채널 저장
        self.residual_check = (stride == 1 and in_channels == out_channels)

        for i in range(repeat):
            current_stride = stride if i == 0 else 1
            layers.append(
                nn.Sequential(
                    PointwiseBlock(in_channels, in_channels * expansion, nonlinear=True),
                    DepthwiseBlock(in_channels * expansion, in_channels * expansion, current_stride),
                    PointwiseBlock(in_channels * expansion, out_channels, nonlinear=False)
                )
            )
            in_channels = out_channels

        self.bottleneck = nn.ModuleList(layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Residual 연결이 필요한 경우 Conv Layer 추가
        if not self.residual_check:
            self.residual = nn.Conv2d(self.initial_in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.residual = None

    def forward(self, x):
        for idx, layer in enumerate(self.bottleneck):
            out = layer(x)
            print("이거 안나오지?: ", layer(x).shape)
            

            # Residual Connection 적용
            if idx == 0 and self.residual:
                x = self.residual(x)
                x = x + out
            else:
                x = out
        return x
    
# class MobileNetV2(nn.Module):
#     def __init__(self, in_channels=3, num_classes=1000, **kwargs):
#         super().__init__()

#         self.in_channels = 32
#         self.conv1 = CNNBlock(in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False)
#         self.architecture = load_model(path)
#         self.layer = self._create_block(self.architecture)

#         self.conv2 = PointwiseBlock(320, 1280, nonlinear=True)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.l_conv = nn.Linear(1280, num_classes)


#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.layer(x)
#         x = self.conv2(x)
#         x = self.avgpool(x)
#         print(x.shape)
#         x = x.view(x.size(0), -1)
#         x = self.l_conv(x)
#         return x                


#     def _create_block(self, architecture):
#         layers = []
#         in_channels = self.in_channels

#         for config in architecture:
#             t, c, n, s = config
#             layers.append(Bottleneck(in_channels, c, t, n, s))
#             print("stride : ", s)
#             in_channels = c

#         return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)  # Flatten
        x = self.l_conv(x)
        return x

    def _create_block(self, architecture):
        layers = []
        in_channels = self.in_channels

        for config in architecture:
            t, c, n, s = config
            layers.append(Bottleneck(in_channels, c, t, n, s))
            print(f"Creating Bottleneck: expansion={t}, out_channels={c}, repeats={n}, stride={s}")
            in_channels = c  # 채널 업데이트

        return nn.Sequential(*layers)
a = torch.randn(1, 3, 224, 224)
model = MobileNetV2()
print(model(a).shape)