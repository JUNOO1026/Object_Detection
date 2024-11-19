import math
import torch

from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PFLDloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmark, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes, axis=0)
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device)
        weight_attribute = torch.sum(attributes.mul(mat_ratio), axis=1)

        l2 = torch.sum(
            (landmark_gt - landmark) * (landmark_gt - landmark), axis=1)
        
        return torch.mean(weight_angle * weight_attribute * l2), torch.mean(l2)
    