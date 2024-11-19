import torch
from torch import nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.layers(x)

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, **kwargs):
        super(DepthwiseConv, self).__init__()

        self.DW_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=in_channels, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.DW_conv(x)

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear=None):
        super(PointwiseConv, self).__init__()

        self.R_PW_conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1)
        self.L_PW_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        self.nonlinear = nonlinear

    def forward(self, x):
        if self.nonlinear:
            return self.R_PW_conv(x)
        else:
            return self.L_PW_conv(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, expansion, out_channels, stride, residual=None):
        super(Bottleneck, self).__init__()
        print('ssss', stride)
        self.layers = nn.Sequential(
            PointwiseConv(in_channels, in_channels*expansion, nonlinear=True),
            DepthwiseConv(in_channels * expansion, in_channels * expansion, stride),
            PointwiseConv(in_channels * expansion, out_channels, nonlinear=False),
        )

        self.residual = residual

    def forward(self, x):
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class PFLDBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super(PFLDBackbone, self).__init__()
        # input(-1, 3, 112, 112)
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6()
        )
        self.DW_conv1_layer = DepthwiseConv(64, 64, stride=1, padding=1)  # output : [1, 64, 56, 56]

        self.Bottleneck_1_1 = Bottleneck(64, 6, 64, 2, residual=False)
        self.Bottleneck_1_2 = Bottleneck(64, 6, 64, 1, residual=True)
        self.Bottleneck_1_3 = Bottleneck(64, 6, 64, 1, residual=True)
        self.Bottleneck_1_4 = Bottleneck(64, 6, 64, 1, residual=True)
        self.Bottleneck_1_5 = Bottleneck(64, 6, 64, 1, residual=True)  # output : [1, 64, 28, 28]

        self.Bottleneck_2_1 = Bottleneck(64, 6, 128, 2, residual=False)  # output : [1, 128, 14, 14]

        self.Bottleneck_3_1 = Bottleneck(128, 6, 128, 1, residual=False)
        self.Bottleneck_3_2 = Bottleneck(128, 6, 128, 1, residual=True)
        self.Bottleneck_3_3 = Bottleneck(128, 6, 128, 1, residual=True)
        self.Bottleneck_3_4 = Bottleneck(128, 6, 128, 1, residual=True)
        self.Bottleneck_3_5 = Bottleneck(128, 6, 128, 1, residual=True)
        self.Bottleneck_3_6 = Bottleneck(128, 6, 128, 1, residual=True)  # output : [1, 128, 14, 14]

        self.Bottleneck_4_1 = Bottleneck(128, 6, 16, 1, residual=False)

        self.conv2_layer = CNNBlock(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3_layer = nn.Conv2d(32, 128, kernel_size=7, stride=1)
        self.bn_l = nn.BatchNorm2d(128)

        self.avgpool1 = nn.AvgPool2d(14)
        self.avgpool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)


    def forward(self, x):
        x = self.conv1_layer(x)
        x = self.DW_conv1_layer(x)

        x = self.Bottleneck_1_1(x)

        x = self.Bottleneck_1_2(x)
        x = self.Bottleneck_1_3(x)
        x = self.Bottleneck_1_4(x)
        out1 = self.Bottleneck_1_5(x)
        print(out1.shape)

        x = self.Bottleneck_2_1(out1)


        x = self.Bottleneck_3_1(x)
        x = self.Bottleneck_3_2(x)
        x = self.Bottleneck_3_3(x)
        x = self.Bottleneck_3_4(x)
        x = self.Bottleneck_3_5(x)
        x = self.Bottleneck_3_6(x)

        x = self.Bottleneck_4_1(x)

        s1 = self.avgpool1(x)
        s1 = s1.view(s1.size(0), -1)


        x = self.conv2_layer(x)

        s2 = self.avgpool2(x)
        s2 = s2.view(s2.size(0), -1)

        s3 = self.conv3_layer(x)

        s3 = s3.view(s3.size(0), -1)

        cat = torch.cat([s1, s2, s3], dim=1)
        landmarks = self.fc(cat)

        return out1, landmarks
    
# [-1, 64, 28, 28]    
class AuxiliaryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CNNBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = CNNBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = CNNBlock(128, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 128, kernel_size=7, stride=1, padding=0)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

a = torch.randn(1, 3, 112, 112)
model = PFLDBackbone(in_channels=3)
out1, landmark = model(a)

print('out:', out1.shape)
print('landmark : ', landmark.shape)

b = torch.randn(1, 64, 28, 28)
model = AuxiliaryBlock()
print(model(b).shape)