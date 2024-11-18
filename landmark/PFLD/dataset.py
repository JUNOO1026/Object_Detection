import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class WLFWDataset(Dataset):
    def __init__(self, d_path, f_path, transform=None):
        self.line = None
        self.path = None
        self.lanmark = None
        self.attribute = None
        self.euler_angle = None
        self.image = None
        self.transform = transform
        self.dataset_path = d_path

        with open(f_path, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        self.line = self.lines[idx].strip().split()
        self.landmark = np.asarray(self.line[:196], dtype=float)
        self.attribute = np.asarray(self.line[200:206], dtype=int)
        self.euler_angle = np.asarray(self.line[206:209], dtype=float)
        self.image = cv2.imread(self.dataset_path + self.line[210:])

        if self.transform:
            self.img = self.transform(self.image)

        return (self.img, self.lanmark, self.attribute, self.euler_angle)
    
    def __len__(self):
        return len(self.lines)


