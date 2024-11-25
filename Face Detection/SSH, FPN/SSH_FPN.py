import torch
from torch import nn


# Original Convolution Block
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): 
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x) -> torch.tensor:
        '''
            forward pass 

            Returns : output tensof of shape  = (1, 3, 256, 256)
        
        '''

        return self.act(self.bn(self.conv(x)))
    

class SSH(nn.Module): 

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv1X1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 
                      kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1)
        )

        self.conv3X3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.conv5X5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4,
                      kernel_size=3, stride=1, padding=2, dilation=2),

            nn.LeakyReLU(0.1)
        )

    def forward(self, x) -> torch.tensor:
        '''
            forward pass

            Input : input tensor of shape : [ x1: (1, 256, 640, 640), 
                                              x2: (1, 256, 640, 640), 
                                              x3: (1, 128, 640, 640)] -> toch.tensor

            Returns : output tensor of shape : [conv1X1, conv3X3, conv5X5] -> torch.tensor (1, 640, 640, 640)
        '''
        x1 = self.conv1X1(x)
        print("x1.shape: ",x1.shape)
        x2 = self.conv3X3(x)
        print("x2.shape: ",x2.shape)
        x3 = self.conv5X5(x)
        print("x3.shape: ",x3.shape)
        x_sum = torch.cat([x1, x2, x3], dim=1)

        return x_sum



if __name__ == '__main__':
    user_input = torch.randn(1, 256, 640, 640)
    model = SSH(in_channels=256, out_channels=512)
    print(model(user_input).shape)