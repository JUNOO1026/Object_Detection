import yaml
import torch
from torch import nn

path = './landmark/config/mobilenetV2.yaml'

def load_model(path) -> list:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    model = config['pfld_architecture']

    return model

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        '''
            forward pass

            Returns : output tensor of shape -> (-1, channel, width, height)

        '''

        return self.act(self.bn(self.conv(x)))
    

class CNNBlockNonact(nn.Module):
    '''
        This class doesn't have activation function.
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x) -> torch.Tensor:
        '''
            forward pass

            Returns : output tensor of shape : (-1, channel, width, height)
        
        '''

        return self.bn(self.conv(x))
    

class DepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.Dw_conv = nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=3, stride=stride, 
                                 padding=1, groups=in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        '''
            forward pass

            Returns : output tensor of shape : (-1, channel, width, height)
        
        '''

        return self.act(self.bn(self.Dw_conv(x)))


class InvertedResidualBlock(nn.Module):
    '''
        This class is PFlD Backbone with MobilenetV2.
    '''
    expansion = 2

    def __init__(self, in_channels, out_channels, stride, use_residual):
        super().__init__()

        self.layers = nn.Sequential(
            CNNBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            DepthwiseBlock(in_channels, in_channels * InvertedResidualBlock.expansion, stride=stride),
            CNNBlockNonact(in_channels * InvertedResidualBlock.expansion, out_channels,  
                           kernel_size=1, stride=1, padding=0)
        )
        self.act = nn.ReLU()

        self.use_residual = use_residual

    def forward(self, x):
        shortcut = self.layers(x)

        if self.use_residual:
            x = self.act(x + shortcut)
        else:
            x = self.act(shortcut)

        return x
    
    @classmethod
    def change_exp(cls, exp) -> int:
        '''
            This funcion is change the InvertedResidualBlock variable.

            Returs : your input inteager value
        '''
        cls.expansion = exp


class PFLDBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        '''
            This class is PFLD BackBone.

            input_shape : (-1, 3, 112, 112)
        
        '''
        self.architecture = load_model(path)

        self.conv1 = CNNBlock(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = CNNBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.in_channels = 64
        
        self.layers1 = self._create_block(64, 2, 64, 5, 2)
        self.layers2 = self._create_block(64, 2, 128, 1, 2)
        self.layers3 = self._create_block(128, 4, 128, 6, 1)
        self.layers4 = self._create_block(128, 2, 16, 1, 1)

        self.conv3 = CNNBlockNonact(in_channels=16, out_channels=32, 
                              kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=128,
                               kernel_size=7, stride=1, padding=0)
        
        self.avgpool1 = nn.AvgPool2d(14)
        self.avgpool2 = nn.AvgPool2d(7)

        self.fcs = nn.Linear(176, 196) # landmark

        



    def forward(self, x):
        x = self.conv1(x) # (-1, 64, 56, 56) 
        x = self.conv2(x) # (-1, 64, 56, 56)
        out1 = self.layers1(x) # (-1, 64, 28, 28)
        x = self.layers2(out1) # (-1, 128, 14, 14)
        x = self.layers3(x) # (-1, 128, 14, 14)
        x = self.layers4(x) # (-1, 16, 14, 14)
        s1 = self.avgpool1(x) # (-1, 16, 1, 1)
        s1 = s1.view(s1.size(0), -1)
        x = self.conv3(x) # (-1, 32, 7, 7)
        s2 = self.avgpool2(x) # (-1, 32, 1, 1)
        s2 = s2.view(s2.size(0), -1)

        s3 = self.conv4(x) # (-1, 128, 1, 1)
        s3 = s3.view(s3.size(0), -1)

        
        features = torch.cat([s1, s2, s3], dim=1)

        landmark = self.fcs(features)

        return out1, landmark

    @staticmethod
    def _create_block(in_channels, expansion, out_channels, repeat, stride):
        layers = []
        use_residual = None

        InvertedResidualBlock.change_exp = expansion

        for idx, _ in enumerate(range(repeat)):
            if idx == 0:
                stride = stride
                use_residual = False

            else:
                stride = 1
                use_residual = True

            layers.append(InvertedResidualBlock(in_channels, out_channels, stride, use_residual))
            in_channels = out_channels

        return nn.Sequential(*layers)
    


class AuxiliaryBlock(nn.Module):
    '''
        This class just calcualte euler_angles: pitch, yaw, raw
    
    '''
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 128, kernel_size=7, stride=1, padding=0)

        self.linear1 = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, 3)


    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        
        x = self.linear1(x)
        euler_angle = self.linear2(x)

        return euler_angle
    

# user_input = torch.randn(1, 3, 112, 112)
# model = PFLDBackbone()
# print(model(user_input)[0].shape, model(user_input)[1].shape)


# user_input2 = torch.randn(1, 64, 28, 28)
# model2 = AuxiliaryBlock()
# print(model2(user_input2))

from torchinfo import summary

# model1 = PFLDBackbone()
# print(summary(model1, (1, 3, 112, 112)))



model2 = AuxiliaryBlock()
print(summary(model2, (1, 64, 28, 28)))