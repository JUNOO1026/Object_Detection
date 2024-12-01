import yaml
import torch

from torch import nn



def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    model = config['yolov8_architecture']

    return model


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
    

class Upsample(nn.Module):
    '''
        This class Upsample Class.

        scale factor is 2 

        input size = (-1, 3, 20, 20)
        output size = (-1, 3, 40, 40)
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.upsample(x)


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
        
        for block in self.bottleneck:
            outputs.append(block(outputs[-1]))  # outputs : [x1, x2, x3, x4]

        outputs = self.l_conv1x1(torch.cat(outputs, dim=1)) # stack

        return outputs


class SPPF(nn.Module):
    '''
        This class is Spatial Pyramid Pooling Fast Module.
        Use Maxpool2D with kernel_size = 5.

    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.ModuleList(
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(3)
        )

        self.l_conv1x1 = CNNBlock(in_channels * 3, out_channels, 
                                  kernel_size=1, stride=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        output = []

        for spp in self.maxpool:
            output.append(spp(x))

        outputs = self.l_conv1x1(torch.cat(output, dim=1))

        return outputs
    
    
class Backbone(nn.Module):
    '''
        This class is yolov8 Backbone.
        Backbone Module consist of (conv + c2f Module)
    '''
    def __init__(self, in_channels, path):
        super().__init__()

        self.in_channels = in_channels
        self.architecture = load_config(path)
        self.block = self._create_block(self.architecture)
        

    def forward(self, x) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = [x]
        
        for block in self.block:
            outputs.append(block(outputs[-1]))
        
        return [outputs[4], outputs[6], outputs[9]]
            

    def _create_block(self, architecture):

        in_channels = self.in_channels
        layers = nn.ModuleList()

        for config in architecture:
            if len(config) == 2:
                out_channels = config[1]
                block = nn.Sequential(
                    CNNBlock(in_channels, out_channels, 
                             kernel_size=3, stride=2, padding=1)
                )
                layers.append(block)
                in_channels = out_channels
                

            else:
                out_channels = config[1]
                repeat = config[2]
                block = nn.Sequential(
                    C2f(in_channels, out_channels, repeat=repeat, shortcut=True)
                )
                layers.append(block)
                in_channels = out_channels
        
        layers.append(nn.Sequential(SPPF(in_channels, in_channels)))

        return layers

class Neck(nn.Module):
    '''
        This class is Neck in Yolov8 model architecture

        Parmas : in_channels, out_channels

        input tensor shape  is ([1, 512, 20, 20])

    '''
    def __init__(self, in_channels, out_channels, repeats=2, **kwargs):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f = C2f(in_channels, out_channels, repeat=3, shortcut=False)


    def forward(self, x) -> list[torch.Tensor, torch.Tensor]:
        p_dish = []

        for idx in range(2, 0, -1):
            upsample = self.upsample(x[idx])
            c_path = torch.cat([upsample, x[idx - 1]], dim=1)
            p_dish.append(c_path)

        return p_dish
        

class Detect_c2f(nn.Module):
    '''
        Detect_bottlneck doesn't have shortcut.

        shorcut default False.
    
    '''
    def __init__(self, in_channels, out_channels, repeat):
        super().__init__()

        self.f_conv1x1 = CNNBlock(in_channels, in_channels, kernel_size=1, stride=1)

        self.block = nn.Sequential(
            CNNBlock(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            CNNBlock(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1),
        ) 


        self.l_conv1x1 = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1)
        self.repeat = repeat

    def forward(self, x):
        x = self.f_conv1x1(x)

        for _ in range(self.repeat):
            x = self.block(x)

        output = self.l_conv1x1(x)

        return output 


class Detect(nn.Module):
    '''
        Yolov8 has Backbone - Detect design.

    
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.block1 = Detect_c2f(in_channels=1024, out_channels=512, repeat=3)
        self.block2 = Detect_c2f(in_channels=768, out_channels=256, repeat=3)
        

    def forward(self, x) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:

        '''
            refactoring essential!!!!!
        
        
        '''
        outputs = [x[2]]
        output = self.upsample(x[2])  
        output = torch.cat([output, x[1]], dim=1)
        output = self.block1(output)
        outputs.insert(0, output)
        output = self.upsample(output)
        output = torch.cat([output, x[0]], dim=1)
        output = self.block2(output)
        outputs.insert(0, output)
        
        # print(outputs.shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)

        return outputs


if __name__ == '__main__':
    a = torch.randn(1, 3, 640, 640)
    backbone = Backbone(in_channels=3, path='./yolov8/config/config.yaml')
    output1 = backbone(a)

    print(output1[0].shape)
    print(output1[1].shape)
    print(output1[2].shape)

    detect = Detect()
    output2 = detect(output1)
    print(output2[0].shape)
    print(output2[1].shape)
    print(output2[2].shape)



        # self.module = nn.ModuleList(
        #     nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode='nearest'), # [1, 512, 40, 40]
        #         C2f(in_channels, out_channels, repeat=3, shortcut=False),
        #     ) for _ in range(repeats)
        # ) 

# if __name__ == '__main__':
#     a = torch.randn(1, 3, 640, 640)
#     model = Backbone(in_channels=3, path='./yolov8/config/config.yaml')
#     b = model(a) 
#     model2 = Neck(in_channels=512, out_channels=256)
#     c = model2(b)

#     print(b[0].shape)
#     print(b[1].shape)
#     print(b[2].shape)

#     print(c[0].shape)
#     print(c[1].shape)
# # if __name__ == "__main__":
#     u_input = torch.randn(1, 3, 640, 640)
#     model = C2f(in_channels=3, out_channels=64, repeat=3)
#     print(model(u_input).shape)
         


        