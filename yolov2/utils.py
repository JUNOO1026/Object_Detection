import torch



def convert_to_corners(boxes):

    x_center, y_center, width, height = boxes.unbind(1)
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)

    return torch.stack((x_min, y_min, x_max, y_max), dim=1)


def match_anchor_box(w, h, to_exclude=[], anchor_boxes=None):
    iou = []
    for idx, box in enumerate(anchor_boxes):
        if idx in to_exclude:
            iou.append(0)
            continue

        intersection_width = min(box[0], w)
        intersection_height = min(box[1], h)
        I = intersection_width * intersection_height
        IOU = I / (w * h + box[0] * box[1] - I)
        iou.append(IOU)

    iou = torch.tensor(iou)

    return torch.argmax(iou, dim=0).item()

def intersection_over_union(bbox1, bbox2):
