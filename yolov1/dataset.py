import os
import yaml
import torch
import xml.etree.ElementTree as ET

from PIL import Image

config_path = 'config/config.yaml'

def load_config(path):
    with open(path, 'r') as f:
        info = yaml.safe_load(f)

    classes = info['classes']

    return classes

class FruitDataset(torch.utils.data.Dataset):
    '''
        This class is FruitDataset.
        image, label, class_map, cell_x, cell_y

    '''

    def __init__(self, df, annotation, split_size=7, num_boxes=2, num_classes=3, transform=None):
        self.dir_path = df
        self.annotation = annotation
        self.transform = transform
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

    def __len__(self):
        return len(self.dir_path)

    def __getitem__(self, idx):
        data = os.path.join(self.dir_path, self.annotation.iloc[idx, 1])
        tree = ET.parse(data)
        root = tree.getroot()
        boxes = []

        label_map = {cls: idx for idx, cls in enumerate(load_config(config_path))}

        if int(root.find('size').find('height').text) == 0:
            file_name = root.find('filename').text
            file_path = os.path.join(self.dir_path + '/' + file_name)
            image = Image.open(file_path)
            width, height = image.size

            for member in root.find('object'):
                classes = member.find('name').text
                class_label_map = label_map[classes]

                # bouning boxes coordinate
                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                center_x = ((xmin + xmax) / 2) / width
                center_y = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                boxes.append([class_label_map, center_x, center_y, box_width, box_height])


        elif int(root.find('size').find('height').text) != 0:
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            for member in root.find('object'):
                classes = member.find('name').text
                class_label_map = label_map[classes]

                # bouning boxes coordinate
                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)

                center_x = ((xmin + xmax) / 2) / width
                center_y = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                boxes.append([class_label_map, center_x, center_y, box_width, box_height])

        boxes = torch.tensor(boxes)
        image = Image.open(self.dir_path + '/' + self.annotation.iloc[idx, 0])
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        grid = torch.zeros((self.S, self.S, self.B * 5 + self.C)) # (7 ,7 ,13)

        for box in boxes:
            class_label, x_c, y_c, b_width, b_height = box.tolist()
            if isinstance(class_label, str):
                class_label = int(class_label)

            i, j = self.S * y_c, self.S * x_c # cell relative coordinate
            x_cell, y_cell = self.S * x_c - j, self.S * y_c - i # understand again
            # 셀 내의 박스 중심 상대 좌표를 의미함.
            # 예를 들어 i,j는 현재 바운딩 박스가 어떤 셀에 있는지를 판단하는지 뜻하고
            # x_cell, y_cell의 경우에는 해당 셀에 바운딩 박스의 중심 좌표가 어디있는지를 알려줌.

            width_cell, height_cell = b_width * self.S, b_height * self.S

            for b in range(self.B):
                if grid[i, j, self.C + b * 5] == 0:
                    grid[i, j, self.C + b * 5] = 1

                start_idx = self.C + b * 5 + 1
                box_coordinate = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                grid[i, j, start_idx:start_idx+4] = box_coordinate
                grid[i, j, class_label] = 1

        return image, grid
