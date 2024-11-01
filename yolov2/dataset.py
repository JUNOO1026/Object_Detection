import os
import torch

from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

class AfricaDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_path = []
        self.transform = transform

        class_names = os.listdir(self.path) # directory set (buffalo, zebra, etc)
        for dir in class_names:
            files = os.listdir(os.path.join(self.path, dir))

            self.image_path += [os.path.join(dir, file) for file in files if files.endswith('.jpg')]



    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        sample = self._make_sample(idx)


    def _make_sample(self, idx):
        img_path = os.path.join(self.path, self.image_path[idx])
        box_info_path = os.path.splitext(img_path)[0] + '.txt'
        img = Image.open(img_path)
        width, height = img.size

        bbox = []
        labels = []

        with open(box_info_path, 'r') as f:
            data = f.readlines()

