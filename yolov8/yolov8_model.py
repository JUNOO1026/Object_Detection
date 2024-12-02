import yaml
import torch

from torch import nn
from common import *


class YOLOv8(nn.Module):
    def __init__(self, path, in_channels=3, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load configurations
        self.backbone_arch, self.head_arch = load_model(path)

        # Create Backbone and Head
        self.backbone = self._create_backbone_block(self.backbone_arch)
        self.head = self._create_head_block(self.head_arch, backbone_outputs=[])

    def forward(self, x):
        # Backbone forward pass
        backbone_outputs = []
        for layer in self.backbone:
            x = layer(x)
            backbone_outputs.append(x)

        # Head forward pass
        head_outputs = self.forward_head(backbone_outputs, self.head)
        print(head_outputs.shape)

        return head_outputs


    def _create_backbone_block(self, architecture):
        in_channels = self.in_channels
        layers = nn.ModuleList()

        for block in architecture:
            module, cfg, repeat, shortcut = block
            print('-' * 50)
            print(module, cfg, repeat, shortcut)
            print('-' * 50)
            print("Input channels before block: ", in_channels)

            if module in ('conv', 'conv_nonact'):
                layers.append(
                    CNNBlock(in_channels=in_channels,
                            out_channels=cfg[0],
                            kernel_size=cfg[1],
                            stride=cfg[2],
                            padding=1)
                )
                in_channels = cfg[0]  # Update in_channels after Conv layer
                print("Output channels after Conv: ", in_channels)

            elif module in ('c2f'):
                layers.append(
                    c2f(in_channels=in_channels,
                        out_channels=cfg[0],
                        num_repeat=repeat,
                        shortcut=shortcut,
                        kernel_size=cfg[1],
                        stride=cfg[2],
                        padding=1)
                )
                in_channels = cfg[0]  # Update in_channels after c2f block
                print("Output channels after c2f: ", in_channels)

        layers.append(SPPF(in_channels=in_channels, out_channels=512))
        print("Final SPPF output channels: 512")
        return layers
    
    def _create_head_block(self, architecture, backbone_outputs):
 
        in_channels = backbone_outputs[-1].shape[1]  # Start with the last backbone output channels
        layers = nn.ModuleList()
        outputs = []

        for idx, block in enumerate(architecture):
            module = block[0]

            print('-' * 50)
            print(f"Processing Head Block {idx}: {block}")
            print("Input channels before block:", in_channels)

            if module == 'upsample':
                # Upsample the current feature map
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
                print("Upsample layer added.")

            elif module == 'concat':
                # Concatenate with the corresponding backbone output
                layers.append('concat')  # Placeholder to signal concatenation
                print("Concat operation placeholder added.")

            elif module in ('conv', 'conv_nonact'):
                cfg = block[1]
                layers.append(
                    CNNBlock(in_channels=in_channels,
                            out_channels=cfg[0],
                            kernel_size=cfg[1],
                            stride=cfg[2],
                            padding=1)
                )
                in_channels = cfg[0]
                print("Output channels after Conv:", in_channels)

            elif module == 'c2f':
                cfg = block[1]
                layers.append(
                    c2f(in_channels=in_channels,
                        out_channels=cfg[0],
                        num_repeat=block[2],
                        shortcut=block[3] == 'True',
                        kernel_size=cfg[1],
                        stride=cfg[2],
                        padding=1)
                )
                in_channels = cfg[0]
                print("Output channels after c2f:", in_channels)

        return layers

    def forward_head(self, features, head_layers):
    
        outputs = [features[-1]]  # Start with the largest feature map (P5)

        for idx, layer in enumerate(head_layers):
            if isinstance(layer, nn.Upsample):
                # Perform upsampling
                outputs.append(layer(outputs[-1]))

            elif layer == 'concat':
                # Concatenate with the corresponding backbone output
                outputs[-1] = torch.cat([outputs[-1], features[-len(outputs)]], dim=1)

            else:
                # Process with a standard layer (e.g., Conv, c2f)
                outputs[-1] = layer(outputs[-1])

        return outputs[-3:]  # Return small, medium, and large detection maps

        


# a, b = load_model(path='./yolov8/config/config.yaml')
# for i in b:
#     print(i[0])
#     break




    # def _create_backbone_block(self, architecture):
    #     in_channels = self.in_channels
    #     output = []
    #     layers = nn.ModuleList()
        
    #     for block in architecture:
    #         module, cfg, repeat, shortcut = block
    #         print('-'*50)
    #         print(module, cfg, repeat, shortcut)
    #         print('-'*50)
    #         print("check : ", in_channels)
    #         if module in ('conv', 'conv_nonact'):

    #             output = nn.Sequential(
    #                 CNNBlock(in_channels=self.in_channels,
    #                          out_channels=cfg[0],
    #                          kernel_size=cfg[1],
    #                          stride=cfg[2], padding=1)
    #             )
    #             layers.append(output)
    #             in_channels = cfg[0]
    #             print(in_channels)

    #         elif module in ('c2f'):
    #             output = nn.Sequential(
    #                 c2f(in_channels=in_channels,
    #                     out_channels=cfg[0],
    #                     num_repeat=repeat,
    #                     shortcut=shortcut,
    #                     kernel_size=cfg[1],
    #                     stride=cfg[2], padding=1)
    #             )
    #             layers.append(output)
    #             in_channels = cfg[0]

    #     output = nn.Sequential(SPPF(in_channels, out_channels=512))
    #     layers.append(output)

    #     return layers
                
    

if __name__ == '__main__':
    a = torch.randn(1, 3, 640, 640)
    model = YOLOv8(path='./yolov8/config/config.yaml', in_channels=3, num_classes=10)
    print(model(a).shape)



# if __name__ == '__main__':
#     a = torch.randn(1, 128, 80, 80)
#     model = c2f(in_channels=128, out_channels=128, num_repeat=6, shortcut=True, kernel_size=3, stride=1, padding=1)
#     print(model(a).shape)
            

        


        

