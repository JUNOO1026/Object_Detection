import os
import torch
import yaml

from PIL import Image
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision import tv_tensors
from utils import match_anchor_box

path = './config/config.yaml'

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    split_size, n_anchor, n_class, anchor_boxes  = config['split_size'], config['num_of_anchor_box'], config['num_of_class'], config['ANCHOR_BOXES']

    return split_size, n_anchor, n_class, anchor_boxes

class AfricaDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_path = []
        self.transform = transform

        self.S, self.A, self.C, self.ANCHOR_BOXES = load_config(path)

        class_names = os.listdir(self.path) # directory set (buffalo, zebra, etc)
        for dir in class_names:
            files = os.listdir(os.path.join(self.path, dir))

            self.image_path += [os.path.join(dir, file) for file in files if files.endswith('.jpg')]



    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        sample = self._make_sample(idx)
        img, labels, bboxes = sample['image'], sample['labels'], sample['bboxes']
        _, height, width = img.size()


        # 여기서 넘어가는 bboxes의 값들은 이미 0~1사이의 값으로 이루어져 있음.
        # x_center, y_center, b_width, b_height
        target = self._make_target(bboxes, labels, width, height)

        return img, target


    def _make_sample(self, idx):
        img_path = os.path.join(self.path, self.image_path[idx])
        box_info_path = os.path.splitext(img_path)[0] + '.txt'
        img = read_image(img_path)
        _, width, height = img.size()

        bbox = []
        labels = []

        with open(box_info_path, 'r') as f:
            data = f.readlines()
            for line in data:
                values = line.split()
                labels.append(int(values[0]))
                temp_bbox = [float(val) for val in values[1:]]

                x, y = temp_bbox[0] * width, temp_bbox[1] * height
                box_width, box_height = temp_bbox[2] * width, temp_bbox[3] * height

                bbox.append([x, y, box_width, box_height])

        bboxes = tv_tensors.BoundingBoxes(bbox, format='CXCYWH', canvas_size=img.shape[1:])
        sample = {
            'image' : img,
            'labels' : labels,
            'bbox' : bboxes
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


    def _make_target(self, bboxes, labels, width, height):
        to_exclude = []
        target = torch.zeros(self.S, self.S, self.A, 1 + 4 + self.C) # (conf, x,y,w,h, cls1, cls2, cls3, cls4)

        for bbox, label in zip(bboxes, labels):
            # 픽셀 기준으로 박스의 중심상대좌표를 구하기 위함임.
            cx, cy = bbox[0] / 32, bbox[1] / 32 # grid * 32 == 416

            pos = (int(cx * 13) , int(cy * 13))
            pos = min(pos[0], 12), min(pos[1], 12)

            # 해당 그리드 셀 내에서 얼마나 떨어져 있는지 알 수 있음.
            bx, by = cx - int(cx), cy - int(cy)
            box_width, box_height = bbox[2] / 32, bbox[3] / 32

            assigned_anchor_box = match_anchor_box(box_width, box_height, to_exclude)
            anchor_box = self.ANCHOR_BOXES[assigned_anchor_box]

            bw_by_pw, bh_by_ph = box_width / anchor_box[0], box_height / anchor_box[1]

            target[pos[0], pos[1], assigned_anchor_box, 0:5] = torch.tensor([1, bx, by, bw_by_pw, bh_by_ph])
            target[pos[0], pos[1], assigned_anchor_box, 5 + int(label)] = 1

            to_exclude.append(assigned_anchor_box)

        return target


path = './file.txt'

with open(path, 'r') as f:
    data = f.readlines()
labels = []
for d in data:
    values = d.split()
    print(float(values[0]))


