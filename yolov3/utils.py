import os
import torch


def intersection_over_union(predictions, ground_truth, box_format='midpoint'):
    if box_format == 'midpoint':
        box1_x1 = predictions[..., 0:1] - (predictions[..., 2:3] / 2)
        box1_y1 = predictions[..., 1:2] - (predictions[..., 3:4] / 2)
        box1_x2 = predictions[..., 0:1] + (predictions[..., 2:3] / 2)
        box1_y2 = predictions[..., 1:2] + (predictions[..., 3:4] / 2)

        box2_x1 = ground_truth[..., 0:1] - (ground_truth[..., 2:3] / 2)
        box2_y1 = ground_truth[..., 1:2] - (ground_truth[..., 3:4] / 2)
        box2_x2 = ground_truth[..., 0:1] + (ground_truth[..., 2:3] / 2)
        box2_y2 = ground_truth[..., 1:2] + (ground_truth[..., 3:4] / 2)

    elif box_format == 'corners':
        box1_x1 = predictions[..., 0:1]
        box1_y1 = predictions[..., 1:2]
        box1_x2 = predictions[..., 2:3]
        box1_y2 = predictions[..., 3:4]

        box2_x1 = ground_truth[..., 0:1]
        box2_y1 = ground_truth[..., 1:2]
        box2_x2 = ground_truth[..., 2:3]
        box2_y2 = ground_truth[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection_over_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection_over_area / (box1_area + box2_area - intersection_over_area)

def iou_width_height(boxes1, boxes2):
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union
