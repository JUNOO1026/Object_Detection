import torch
import torch.nn.functional as F
from torch import nn

# Original Convolution Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=0,**kwargs): 
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=leaky)

    def forward(self, x) -> torch.tensor:
        '''
            forward pass 

            Returns : output tensof of shape  = (1, 3, 256, 256)
        
        '''

        return self.act(self.bn(self.conv(x)))
    

class CNNBlockNonact(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        return self.bn(self.conv(x))
    

class SSH(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        assert out_channels % 4 == 0

        leaky = 0
        if out_channels <= 64:
            leaky = 0.1

        self.conv3x3 = CNNBlock(in_channels, out_channels // 2, stride=1, padding=1)

        self.conv5x5_1 = CNNBlock(in_channels, out_channels // 4, stride=1, padding=1, leaky=leaky)
        self.conv5x5_2 = CNNBlockNonact(out_channels // 4, out_channels // 4, stride=1, padding=1)

        self.conv7x7_1 = CNNBlock(out_channels //4, out_channels // 4, stride=1, padding=1, leaky=leaky)
        self.conv7x7_2 = CNNBlockNonact(out_channels//4, out_channels//4, stride=1, padding=1)

    def forward(self, x) -> torch.Tensor:
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5_2 = self.conv5x5_2(conv5x5_1)

        conv7x7_1 = self.conv7x7_1(conv5x5_1)
        conv7x7_2 = self.conv7x7_2(conv7x7_1)

        output = torch.cat([conv3x3, conv5x5_2, conv7x7_2], dim=1)
        
        return F.relu(output)         
        

class FPN(nn.Module):
    def __init__(self, feature_list, out_channels):
        super().__init__()

        assert out_channels % 4 == 0

        leaky = 0
        if out_channels <= 64:
            leaky = 0.1


        self.lateral_connections = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) for in_channels in feature_list
        ])        

        self.output_conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1) for in_channels in feature_list
        ])


    def TopDownPath(self, lateral) -> torch.Tensor:

        l_feature = [lateral[-1]]

        for i in range(len(lateral) - 2, -1, -1):
            upsampled = F.interpolate(
                l_feature[0], size=lateral[i].shape[-2:], mode='nearest'
            )
            l_feature.insert(0, lateral[i] + upsampled)


        return l_feature
    
    def forward(self, feature):

        lateral_conv = [
            lateral_conv(feature) for feature, lateral_conv in zip(feature, self.lateral_connections)
        ]

        Top_down = self.TopDownPath(lateral_conv)

        output = [
            output_conv(Top_down) for Top_down, output_conv in zip(Top_down, self.output_conv)
        ]
        
        return output

if __name__ == '__main__':
    # user_input = torch.randn(1, 256, 640, 640)
    # model = SSH(in_channels=256, out_channels=512)
    # print(model(user_input).shape)

    # Simulate feature maps from a backbone
    C3 = torch.randn(1, 256, 64, 64)  # ResNet C3
    C4 = torch.randn(1, 512, 32, 32)  # ResNet C4
    C5 = torch.randn(1, 1024, 16, 16) # ResNet C5

    # Instantiate FPN
    fpn = FPN(feature_list=[256, 512, 1024], out_channels=256)

    # Forward pass
    outputs = fpn([C3, C4, C5])

    # Print output shapes
    for i, output in enumerate(outputs):
        print(f"Output P{i+3}: shape = {output.shape}")