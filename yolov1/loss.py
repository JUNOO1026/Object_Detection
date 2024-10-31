import torch
from torch import nn

from utils import intersection_over_union

class YoloV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # 마지막 모델 출력단에서 flatten()을 사용했기 때문에, reshape을 통해 복원이 가능해짐.
        # flatten()은 배열의 순서를 유지하여 가능한 것.
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # predictions의 출력도 (배치사이즈, 7, 7, 13)인 것을 확인하라.

        # dataset.py를 보면 출력값은 다음과 같다.
        # [cls1, cls2, cls3, obj1, x_center, y_center, width_cell, height_cell, ob2, x_center, y_center, width_cell, height_cell]
        # 정답은 하나이니 두개의 바운딩 박스 모두 intersection 작업이 필요함.

        iou_b1 = intersection_over_union(predictions[..., self.C + 1 : self.C + 5], targets[..., self.C + 1 : self.C + 5])
        iou_b2 = intersection_over_union(predictions[..., self.C + 6: self.C + 10], targets[..., self.C + 1 : self.C + 5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box = torch.max(ious, dim=0)
        # iou_maxes는 큰 값을 반환, best_box는 인덱스 값 반환
        exists_box = targets[..., self.C].unsqueeze(3)

        predictions_box = exists_box * (
            best_box * predictions[..., self.C+6:self.C+10]
            + (1 - best_box) * predictions[..., self.C+1:self.C+5]
        )

        target_box = exists_box * targets[..., self.C+1:self.C+5]


        # coordinate loss (x, y)
        box_loss_xy = self.mse(
            torch.flatten(predictions_box[..., 0:2], end_dim=-2),
            torch.flatten(target_box[..., 0:2], end_dim=-2),
        )

        # coordinate loss (w, h)
        prediction_box = torch.sign(predictions_box[..., 2:4] * torch.sqrt(torch.abs(predictions_box[..., 2:4])))
        target_box = torch.sqrt(target_box[..., 2:4])

        box_loss_wh = self.mse(
            torch.flatten(prediction_box, end_dim=-2), # -> shape : (-1, 49, 2)
            torch.flatten(target_box, end_dim=-2),     # -> shape : (-1, 49, 2)
        )

        # obj loss
        pred_conf = (best_box * predictions[..., self.C+5:self.C+6]) + ( 1 - best_box) * predictions[..., self.C:self.C+1]

        conf_loss = self.mse(
            torch.flatten(exists_box * pred_conf, start_dim=1),                        # shape : (-1, 49)
            torch.flatten(exists_box * targets[..., self.C:self.C+1], start_dim=1),    # shape : (-1, 49)
        )

        ## noobj loss

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),  # shape : (-1, 49)
            torch.flatten((1 - exists_box) * targets[..., self.C:self.C+1], start_dim=1),      # shape : (-1, 49)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),  # shape : (-1, 49)
            torch.flatten((1 - exists_box) * targets[..., self.C:self.C+1], start_dim=1),        # shape : (-1, 49)
        )

        # class loss

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),     # shape : (-1, 49, 2)
            torch.flatten(exists_box * targets[..., :self.C], end_dim=-2)          # shape : (-1, 49, 2)
        )

        loss = (
                self.lambda_coord * box_loss_xy
                + self.lambda_coord * box_loss_wh
                + conf_loss
                + self.lambda_noobj * no_object_loss
                + class_loss
        )

        return loss





