import torch

from torch import nn
from common import *

path = './yolov8/config/config.yaml'

class Yolov8(nn.Module):
    def __init__(self, num_classes, path):
        super().__init__()
        self.in_channels = 3
    
        self.backbone = Backbone(self.in_channels, path)
        # 두 개의 module을 설계해야 됨.
        self.neck = []
        self.head = []
        

    def forward(self, x):
        c2f_1, c2f_2, sppf = self.backbone(x)

        
        return 1
    





