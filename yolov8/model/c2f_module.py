import torch
from torch import nn


class CNNBlock(nn.Module):
    '''
        This class is Original CNN Block.

        **kwargs : kernel_size, stride, padding
    
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()


    def forward(self, x) -> torch.Tensor:
        '''
            Returns : output tensor of shape (BATCH_SIZE, channel, width, height)
        
        '''
        return self.act(self.bn(self.conv(x)))
    

class Bottleneck(nn.Module):
    '''
        This class is Bottleneck in c2f Block.

        param : in_channels, out_channels, shorcut
    
    '''
    def __init__(self, in_channels, out_channels, shortcut=None):
        super().__init__()

        self.block = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
        )

        self.shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x) -> torch.Tensor:
        '''
            If shortcut is True -> residual + Bottlenck output
               shortcut is False -> Bottlenck output

            Element Wise calculate.
        '''
    
        residual = x

        return residual + self.block(x) if self.shortcut else self.block(x) # elemenet wise
    

class C2f(nn.Module):
    '''
        This class is C2f Module.

        paramas : in_channels, out_channels, repeat(Bottleneck num repeat) 
    
    '''
    def __init__(self, in_channels, out_channels, repeat, shortcut=None):
        super().__init__()

        self.f_conv1x1 = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.bottleneck = nn.ModuleList(
            Bottleneck(out_channels, out_channels, shortcut) for _ in range(repeat)
        )

        self.l_conv1x1 = CNNBlock(out_channels * (repeat + 1), out_channels, 
                                  kernel_size=1, stride=1, padding=0)
        

    def forward(self, x) -> torch.Tensor:
        '''
            Use Concatenate N channel.    
        
            Returns : output tensor of shape -> (BATCHSIZE, channel, width, height)
            
        '''
        outputs = [self.f_conv1x1(x)]
        print('outputs1. shape : ', outputs[0].shape)
        
        for block in self.bottleneck:
            outputs.append(block(outputs[-1]))  # outputs : [x1, x2, x3, x4]

        outputs = self.l_conv1x1(torch.cat(outputs, dim=1)) # stack

        return outputs


if __name__ == "__main__":
    u_input = torch.randn(1, 3, 640, 640)
    model = C2f(in_channels=3, out_channels=64, repeat=3)
    print(model(u_input).shape)
         


        